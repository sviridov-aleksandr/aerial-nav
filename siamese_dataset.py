"""
SiameseDataset — PyTorch Dataset для он-лет обучения сиамской сети.

Читает тайлы из GeoTIFF по мере необходимости, не загружая карту в память.

Использование:
  from siamese_dataset import SiameseDataset
  from torch.utils.data import DataLoader
  
  dataset = SiameseDataset(
      map_path='/home/alex/aerial-nav/map_cache/antiuav_route_strip.tif',
      coords_path='/home/alex/aerial-nav/training_data/route_dataset/positive_coords.npy',
      tile_size=512,
      neg_multiplier=3
  )
  
  loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.windows import Window
from PIL import Image, ImageEnhance, ImageFilter
import random
from io import BytesIO

# Убираем лимит PIL
Image.MAX_IMAGE_PIXELS = None


class SiameseDataset(Dataset):
    """
    Он-лет датасет для обучения сиамской сети.
    
    Генерирует пары (map_tile, camera_frame) на лету:
    - Positive: один и тот же тайл, camera_frame с аугментациями
    - Negative: разные тайлы, camera_frame без аугментаций
    """

    def __init__(self, map_path: str, coords_path: str,
                 tile_size: int = 512, neg_multiplier: int = 3,
                 augment: bool = True):
        """
        Args:
            map_path: путь к GeoTIFF карте
            coords_path: путь к файлу с координатами тайлов (.npy)
            tile_size: размер тайла (512)
            neg_multiplier: количество negative на 1 positive
            augment: применять аугментации к camera_frame
        """
        self.map_path = map_path
        self.tile_size = tile_size
        self.neg_multiplier = neg_multiplier
        self.augment = augment

        # Открываем карту только для чтения метаданных.
        # Сам файл открывается лениво в __getitem__ (в каждом воркере свой
        # дескриптор) — при fork общий дескриптор rasterio ломает чтение.
        print(f"[SiameseDataset] Открываем карту: {map_path}")
        with rasterio.open(map_path) as src:
            self.width = src.width
            self.height = src.height
        self._src = None
        print(f"[SiameseDataset] Карта: {self.width}x{self.height} px")

        # Загружаем координаты
        print(f"[SiameseDataset] Загружаем координаты: {coords_path}")
        self.coords = np.load(coords_path)
        self.num_tiles = len(self.coords)
        print(f"[SiameseDataset] Тайлов: {self.num_tiles}")

        # Кэш для негативных тайлов (чтобы не читать один и тот же тайл дважды)
        self._neg_cache = {}
        self._cache_size = 1000

    def __len__(self):
        # Каждый элемент — это (positive_idx, neg_idx)
        # Для каждого positive генерируем neg_multiplier negative
        return self.num_tiles * (1 + self.neg_multiplier)

    def _ensure_src(self):
        """Открывает rasterio-дескриптор в текущем процессе (воркере)."""
        if self._src is None:
            self._src = rasterio.open(self.map_path)

    def __getitem__(self, idx):
        self._ensure_src()

        # Определяем, positive или negative
        pos_idx = idx // (1 + self.neg_multiplier)
        is_negative = (idx % (1 + self.neg_multiplier)) > 0

        # Получаем координаты positive тайла
        pos_x, pos_y = self.coords[pos_idx]

        # Читаем positive тайл
        pos_window = Window(pos_x, pos_y, self.tile_size, self.tile_size)
        pos_tile = self._src.read(window=pos_window)
        pos_array = self._tile_to_array(pos_tile)

        if is_negative:
            # Генерируем negative тайл
            neg_idx = random.randint(0, self.num_tiles - 1)
            while neg_idx == pos_idx:
                neg_idx = random.randint(0, self.num_tiles - 1)

            neg_x, neg_y = self.coords[neg_idx]
            neg_window = Window(neg_x, neg_y, self.tile_size, self.tile_size)
            neg_tile = self._src.read(window=neg_window)
            neg_array = self._tile_to_array(neg_tile)

            # Map — это negative тайл, camera — positive с аугментациями
            map_array = neg_array
            camera_array = self._augment(pos_array) if self.augment else pos_array
            label = 0.0
        else:
            # Map — это positive тайл, camera — positive с аугментациями
            map_array = pos_array
            camera_array = self._augment(pos_array) if self.augment else pos_array
            label = 1.0

        # Нормализация (ImageNet)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        map_tensor = (map_array.astype(np.float32) / 255.0 - mean) / std
        camera_tensor = (camera_array.astype(np.float32) / 255.0 - mean) / std

        # Переворачиваем для PyTorch (C, H, W)
        map_tensor = torch.from_numpy(map_tensor.transpose(2, 0, 1))
        camera_tensor = torch.from_numpy(camera_tensor.transpose(2, 0, 1))

        return {
            'map': map_tensor,
            'camera': camera_tensor,
            'label': torch.tensor(label, dtype=torch.float32),
            'pos_idx': pos_idx
        }

    def _tile_to_array(self, tile: np.ndarray) -> np.ndarray:
        """Конвертирует тайл из rasterio в numpy (H, W, C)."""
        if tile.shape[0] == 3:
            return np.transpose(tile, (1, 2, 0))
        else:
            return np.transpose(tile[:3], (1, 2, 0))

    def _augment(self, frame: np.ndarray) -> np.ndarray:
        """Применяет аугментации к кадру."""
        img = Image.fromarray(frame)

        # Яркость (±20%)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))

        # Контраст (±20%)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))

        # Цветовой тон (±10%)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.9, 1.1))

        # Гамма (0.8-1.2)
        img = np.array(img)
        gamma = random.uniform(0.8, 1.2)
        img = np.power(img / 255.0, 1.0 / gamma) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Шум (σ=5-15)
        sigma = random.uniform(5, 15)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Размытие (0-2 пикселя)
        blur = random.randint(0, 2)
        if blur > 0:
            img = Image.fromarray(img)
            img = img.filter(ImageFilter.GaussianBlur(blur))
            img = np.array(img)

        # Сжатие JPEG (качество 70-95)
        quality = random.randint(70, 95)
        buf = BytesIO()
        img = Image.fromarray(img)
        img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        img = Image.open(buf)
        img = np.array(img)

        return img

    def close(self):
        """Закрывает rasterio источник."""
        if self._src:
            self._src.close()


def main():
    """Тестовый запуск датасета."""
    dataset = SiameseDataset(
        map_path='/home/alex/aerial-nav/map_cache/antiuav_route_strip.tif',
        coords_path='/home/alex/aerial-nav/training_data/route_dataset/positive_coords.npy',
        tile_size=512,
        neg_multiplier=3,
        augment=True
    )

    print(f"\n[Test] Dataset size: {len(dataset)}")
    print(f"[Test] Loading sample item...")

    item = dataset[0]
    print(f"[Test] Map shape: {item['map'].shape}")
    print(f"[Test] Camera shape: {item['camera'].shape}")
    print(f"[Test] Label: {item['label']}")
    print(f"[Test] Pos idx: {item['pos_idx']}")

    dataset.close()
    print(f"\n[Test] OK")


if __name__ == '__main__':
    main()
