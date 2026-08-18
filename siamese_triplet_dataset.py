"""
siamese_triplet_dataset.py — датасет для triplet-обучения сиамской сети.

Генерирует триплеты (anchor, positive, negative) на лету:
  - anchor: кадр камеры (патч из карты с масштабом + аугментации)
  - positive: тайл карты без аугментаций (эталон)
  - negative: другой тайл (hard negative — соседний или случайный)

Ключевое отличие от старой версии:
  Масштаб моделируется ЧЕРЕЗ КАРТУ — читается патч size×scale пикселей
  из GeoTIFF и ресемплинг в tile_size. Это честная имитация кадра камеры
  с высоты: больше земли видно → больше патч → меньше масштаб объектов.
  Старая версия ресайзила сам тайл (зум), что не моделировало высоту.

Использование:
  from siamese_triplet_dataset import TripletDataset
  dataset = TripletDataset(
      map_path='map_cache/region_google.tif',
      coords_path='training_data/region_dataset/positive_coords.npy',
      tile_size=512
  )
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.windows import Window
from PIL import Image
import random

Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augmentations import apply_camera_conditions


class TripletDataset(Dataset):
    """
    Он-лет датасет для triplet-обучения.

    Для каждого элемента:
      1. Берём тайл (positive) — 512×512 из карты
      2. Читаем патч из карты с масштабом (anchor) — честная имитация высоты
      3. Применяем аугментации к anchor (поворот, перспектива, погода, шум)
      4. Берём другой тайл → negative
    """

    def __init__(self, map_path: str, coords_path: str,
                 tile_size: int = 512, hard_neg_prob: float = 0.5,
                 aug_level: int = 0):
        """
        Args:
            map_path: путь к GeoTIFF карте
            coords_path: путь к файлу с координатами тайлов (.npy)
            tile_size: размер тайла (512)
            hard_neg_prob: вероятность выбора negative из соседних тайлов
            aug_level: curriculum-этап (0=масштаб, 1=+поворот, 2=полный)
        """
        self.map_path = map_path
        self.tile_size = tile_size
        self.hard_neg_prob = hard_neg_prob
        self.aug_level = aug_level

        print(f"[TripletDataset] Открываем карту: {map_path}")
        with rasterio.open(map_path) as src:
            self.width = src.width
            self.height = src.height
        self._src = None
        print(f"[TripletDataset] Карта: {self.width}x{self.height} px")

        print(f"[TripletDataset] Загружаем координаты: {coords_path}")
        self.coords = np.load(coords_path)
        self.num_tiles = len(self.coords)
        print(f"[TripletDataset] Тайлов: {self.num_tiles}")

    def __len__(self):
        return self.num_tiles

    def _ensure_src(self):
        """Открывает rasterio-дескриптор в текущем процессе (воркере)."""
        if self._src is None:
            self._src = rasterio.open(self.map_path)

    def _read_tile(self, x, y):
        """Читает тайл tile_size×tile_size из GeoTIFF. (x, y) — верхний левый угол."""
        win = Window(int(x), int(y), self.tile_size, self.tile_size)
        data = self._src.read(window=win)
        arr = np.transpose(data, (1, 2, 0))
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        h, w = arr.shape[:2]
        if h < self.tile_size or w < self.tile_size:
            padded = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
            padded[:h, :w] = arr
            arr = padded
        return arr

    def _read_scaled_patch(self, x, y, scale):
        """
        Читает патч из карты с масштабом scale относительно тайла.
        Имитирует кадр камеры с высоты: scale=2.0 → виден участок 2× больше.

        Читает patch_size = tile_size * scale пикселей из карты,
        ресемплинг в tile_size. (x, y) — центр патча.
        """
        ts = self.tile_size
        patch_size = int(ts * scale)
        x1 = int(x - patch_size / 2)
        y1 = int(y - patch_size / 2)

        win = Window(x1, y1, patch_size, patch_size)
        try:
            data = self._src.read(window=win)
            arr = np.transpose(data, (1, 2, 0))
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
        except Exception:
            return np.zeros((ts, ts, 3), dtype=np.uint8)

        h, w = arr.shape[:2]
        if h < patch_size or w < patch_size:
            padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
            padded[:h, :w] = arr
            arr = padded

        # Ресемплинг к tile_size (как камера: 4K → сеть 512)
        img = Image.fromarray(arr)
        img = img.resize((ts, ts), Image.BILINEAR)
        return np.array(img)

    def _normalize(self, arr):
        """Нормализация ImageNet."""
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        out = arr.astype(np.float32) / 255.0
        out = (out - mean) / std
        return torch.from_numpy(out.transpose(2, 0, 1))

    def __getitem__(self, idx):
        self._ensure_src()

        # Positive: тайл карты без аугментаций
        px, py = self.coords[idx]
        pos_tile = self._read_tile(px, py)

        # Anchor: кадр камеры — патч из карты с масштабом + аугментации
        # Масштаб моделируется через карту (честная имитация высоты)
        from augmentations import CURRICULUM_LEVELS
        cfg = CURRICULUM_LEVELS.get(self.aug_level, CURRICULUM_LEVELS[2])
        scale = random.uniform(cfg['scale_range'][0], cfg['scale_range'][1])

        # Центр тайла на карте
        cx = px + self.tile_size / 2
        cy = py + self.tile_size / 2

        # Читаем патч с масштабом (кадр камеры)
        anchor_tile = self._read_scaled_patch(cx, cy, scale)

        # Применяем остальные аугментации (поворот, перспектива, погода, шум)
        # apply_camera_conditions с level, но БЕЗ масштаба (уже сделан через карту)
        img = Image.fromarray(anchor_tile)
        anchor_tile = np.array(apply_camera_conditions(img, level=self.aug_level, skip_scale=True))

        # Negative: другой тайл
        if random.random() < self.hard_neg_prob:
            offset = random.choice([1, 2, 3, -1, -2, -3])
            neg_idx = (idx + offset) % self.num_tiles
        else:
            neg_idx = random.randint(0, self.num_tiles - 1)
            while neg_idx == idx:
                neg_idx = random.randint(0, self.num_tiles - 1)

        nx, ny = self.coords[neg_idx]
        neg_tile = self._read_tile(nx, ny)

        return {
            'anchor': self._normalize(anchor_tile),
            'positive': self._normalize(pos_tile),
            'negative': self._normalize(neg_tile),
        }

    def close(self):
        if self._src:
            self._src.close()