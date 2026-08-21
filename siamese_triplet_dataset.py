"""
siamese_triplet_dataset.py — датасет для triplet-обучения сиамской сети
на основе правила многоуровневых индексов.

Ключевая идея: каждый триплет привязан к уровню индекса (высоте полёта).
  - positive: патч уровня (patch_size → resize в 512) — то, что лежит в индексе
  - anchor:   тот же участок + аугментации (поворот, перспектива, сезоность, шум)
              — то, что видит камера
  - negative: патч другого участка того же уровня

Масштаб (высота) заложен в размере патча уровня, а не в аугментации.
Модель учится: «кадр камеры с высоты h ≈ патч уровня h + искажения».

Уровни (из multi_level_index.py):
  L0: патч 512×512   (h≈0-550 м,   scale=1.0)
  L1: патч 1024×1024 (h≈550-950 м, scale=2.0)
  L2: патч 1792×1792 (h≈950-1500 м, scale=3.5)

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
import math

Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augmentations import apply_camera_conditions, CURRICULUM_LEVELS
from multi_level_index import LEVELS, altitude_to_level, ALT_STEP, ALT_MAX

TILE_SIZE = 512


class TripletDataset(Dataset):
    """
    Он-лет датасет для triplet-обучения по правилу многоуровневых индексов.

    Для каждого элемента:
      1. Случайно выбираем уровень индекса (высоту)
      2. positive: патч уровня из карты → resize в 512 (эталон индекса)
      3. anchor: тот же центр, патч уровня → resize в 512 + аугментации (кадр камеры)
      4. negative: патч уровня из другого участка → resize в 512
    """

    def __init__(self, map_path: str, coords_path: str,
                 tile_size: int = 512, hard_neg_prob: float = 0.5,
                 aug_level: int = 0,
                 level_weights: tuple = None):
        """
        Args:
            map_path: путь к GeoTIFF карте
            coords_path: путь к файлу с координатами тайлов (.npy)
            tile_size: размер тайла/выхода (512)
            hard_neg_prob: вероятность выбора negative из соседних тайлов
            aug_level: curriculum-этап (0=фотометрия, 1=+геометрия, 2=полный)
            level_weights: вероятности выбора каждого уровня индекса.
                           По умолчанию — равномерно по 13 уровням.
        """
        self.map_path = map_path
        self.tile_size = tile_size
        self.hard_neg_prob = hard_neg_prob
        self.aug_level = aug_level
        if level_weights is None:
            level_weights = tuple(1.0 / len(LEVELS) for _ in LEVELS)
        self.level_weights = level_weights

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
        print(f"[TripletDataset] Уровней индекса: {len(LEVELS)}")
        for i, lvl in enumerate(LEVELS):
            print(f"  L{i}: патч {lvl['patch_size']}×{lvl['patch_size']} "
                  f"(h={lvl['alt_min']}-{lvl['alt_max']} м), "
                  f"вероятность={level_weights[i]}")

    def __len__(self):
        return self.num_tiles

    def _ensure_src(self):
        """Открывает rasterio-дескриптор в текущем процессе (воркере)."""
        if self._src is None:
            self._src = rasterio.open(self.map_path)

    def _read_patch(self, cx, cy, patch_size):
        """
        Читает патч patch_size×patch_size из GeoTIFF с центром (cx, cy),
        ресемплинг в tile_size (512).
        """
        ts = self.tile_size
        x1 = int(cx - patch_size / 2)
        y1 = int(cy - patch_size / 2)

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

        img = Image.fromarray(arr)
        img = img.resize((ts, ts), Image.BILINEAR)
        return np.array(img)

    def _choose_level(self):
        """Случайный выбор уровня индекса по весам."""
        return random.choices(range(len(LEVELS)), weights=self.level_weights)[0]

    def _normalize(self, arr):
        """Нормализация ImageNet."""
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        out = arr.astype(np.float32) / 255.0
        out = (out - mean) / std
        return torch.from_numpy(out.transpose(2, 0, 1))

    def __getitem__(self, idx):
        self._ensure_src()

        # Выбор уровня индекса (высоты)
        lvl_idx = self._choose_level()
        patch_size = LEVELS[lvl_idx]['patch_size']

        # Positive: патч уровня из карты → resize в 512 (эталон индекса)
        px, py = self.coords[idx]
        cx = px + TILE_SIZE / 2
        cy = py + TILE_SIZE / 2
        pos_tile = self._read_patch(cx, cy, patch_size)

        # Anchor: тот же центр, патч уровня → resize в 512 + аугментации
        # (имитация кадра камеры с высоты данного уровня)
        anchor_tile = self._read_patch(cx, cy, patch_size)
        anchor_img = Image.fromarray(anchor_tile)
        anchor_tile = np.array(apply_camera_conditions(anchor_img, level=self.aug_level))

        # Negative: патч уровня из другого участка
        if random.random() < self.hard_neg_prob:
            offset = random.choice([1, 2, 3, -1, -2, -3])
            neg_idx = (idx + offset) % self.num_tiles
        else:
            neg_idx = random.randint(0, self.num_tiles - 1)
            while neg_idx == idx:
                neg_idx = random.randint(0, self.num_tiles - 1)

        nx, ny = self.coords[neg_idx]
        ncx = nx + TILE_SIZE / 2
        ncy = ny + TILE_SIZE / 2
        neg_tile = self._read_patch(ncx, ncy, patch_size)

        return {
            'anchor': self._normalize(anchor_tile),
            'positive': self._normalize(pos_tile),
            'negative': self._normalize(neg_tile),
            'level': lvl_idx,
        }

    def close(self):
        if self._src:
            self._src.close()
