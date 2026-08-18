"""
Загрузка и управление тайлами карты местности.
Поддерживает загрузку спутниковых снимков из файлов или кэша.
"""

import os
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field


@dataclass
class MapTile:
    """Один тайл карты."""
    image: np.ndarray  # HxWx3, RGB
    lat: float  # широта центра
    lon: float  # долгота центра
    resolution: float  # метров на пиксель
    tile_size: int = 512  # размер тайла в пикселях


@dataclass
class MapRegion:
    """Регион карты, состоящий из тайлов."""
    tiles: Dict[Tuple[int, int], MapTile] = field(default_factory=dict)
    min_lat: float = 0.0
    max_lat: float = 0.0
    min_lon: float = 0.0
    max_lon: float = 0.0
    resolution: float = 0.1  # м/пиксель по умолчанию


class MapLoader:
    """Загружает и управляет тайлами карты местности."""

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser('~'), '.aerial_nav_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self._region: Optional[MapRegion] = None

    def load_from_image(self, image_path: str, lat: float, lon: float,
                        resolution: float = 0.1) -> MapRegion:
        """
        Загрузить карту из одного изображения.

        Args:
            image_path: путь к изображению карты
            lat: широта центра карты
            lon: долгота центра карты
            resolution: метров на пиксель

        Returns:
            MapRegion с загруженным тайлом
        """
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)

        tile = MapTile(
            image=img_array,
            lat=lat,
            lon=lon,
            resolution=resolution,
            tile_size=img_array.shape[0]
        )

        self._region = MapRegion(
            tiles={(0, 0): tile},
            min_lat=lat - 0.01,
            max_lat=lat + 0.01,
            min_lon=lon - 0.01,
            max_lon=lon + 0.01,
            resolution=resolution
        )
        return self._region

    def load_from_directory(self, dir_path: str, resolution: float = 0.1) -> MapRegion:
        """
        Загрузить карту из директории с тайлами.

        Файлы должны называться по шаблону: tile_{col}_{row}.png

        Args:
            dir_path: путь к директории с тайлами
            resolution: метров на пиксель

        Returns:
            MapRegion с загруженными тайлами
        """
        tiles = {}
        min_lat, max_lat = float('inf'), float('-inf')
        min_lon, max_lon = float('inf'), float('-inf')

        for filename in os.listdir(dir_path):
            if not filename.endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                continue

            # Парсим имя файла tile_{col}_{row}.png
            base = filename.rsplit('.', 1)[0]
            parts = base.split('_')
            if len(parts) >= 3 and parts[0] == 'tile':
                try:
                    col, row = int(parts[1]), int(parts[2])
                except ValueError:
                    continue

                img_path = os.path.join(dir_path, filename)
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)

                # Вычисляем координаты центра тайла
                tile_size = img_array.shape[0]
                center_lat = 55.7550 + row * 0.001  # пример: Москва
                center_lon = 37.6173 + col * 0.001

                tiles[(col, row)] = MapTile(
                    image=img_array,
                    lat=center_lat,
                    lon=center_lon,
                    resolution=resolution,
                    tile_size=tile_size
                )

                min_lat = min(min_lat, center_lat - 0.0005)
                max_lat = max(max_lat, center_lat + 0.0005)
                min_lon = min(min_lon, center_lon - 0.0005)
                max_lon = max(max_lon, center_lon + 0.0005)

        if not tiles:
            raise ValueError(f"Не найдено тайлов в {dir_path}")

        self._region = MapRegion(
            tiles=tiles,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            resolution=resolution
        )
        return self._region

    def get_tile_at(self, lat: float, lon: float) -> Optional[MapTile]:
        """Получить тайл по координатам."""
        if self._region is None:
            return None

        for (col, row), tile in self._region.tiles.items():
            tile_lat = tile.lat
            tile_lon = tile.lon
            half_size = tile.tile_size * self._region.resolution / 2

            if (abs(lat - tile_lat) < 0.001 and
                    abs(lon - tile_lon) < 0.001):
                return tile

        return None

    def get_region(self) -> Optional[MapRegion]:
        """Получить текущий регион карты."""
        return self._region

    def get_map_crop(self, lat: float, lon: float,
                     width_meters: float, height_meters: float) -> Optional[np.ndarray]:
        """
        Получить фрагмент карты вокруг заданных координат.

        Args:
            lat: широта центра
            lon: долгота центра
            width_meters: ширина в метрах
            height_meters: высота в метрах

        Returns:
            Срез карты в виде numpy массива
        """
        if self._region is None:
            return None

        resolution = self._region.resolution
        width_pixels = int(width_meters / resolution)
        height_pixels = int(height_meters / resolution)

        # Находим ближайший тайл
        tile = self.get_tile_at(lat, lon)
        if tile is None:
            # Создаём заглушку (серый фон)
            return np.zeros((height_pixels, width_pixels, 3), dtype=np.uint8)

        # Вычисляем центр тайла в пикселях
        center_px_x = tile.tile_size // 2
        center_px_y = tile.tile_size // 2

        # Вычисляем центр запрашиваемой области в пикселях карты
        # Разница в координатах → разница в пикселях
        lat_diff = lat - tile.lat
        lon_diff = lon - tile.lon

        # Конвертируем разницу координат в пиксели
        # 1 градус ≈ 111000 метров
        meters_per_pixel = resolution
        pixels_per_degree_lat = 111000.0 / meters_per_pixel
        pixels_per_degree_lon = (111000.0 * np.cos(np.radians(tile.lat))) / meters_per_pixel

        center_crop_x = center_px_x + int(lon_diff * pixels_per_degree_lon)
        center_crop_y = center_px_y + int(lat_diff * pixels_per_degree_lat)

        # Вычисляем границы выреза
        x_start = center_crop_x - width_pixels // 2
        y_start = center_crop_y - height_pixels // 2
        x_end = x_start + width_pixels
        y_end = y_start + height_pixels

        h, w = tile.image.shape[:2]

        # Создаём результат
        crop = np.zeros((height_pixels, width_pixels, 3), dtype=np.uint8)

        # Определяем области пересечения
        # src — координаты в исходном тайле
        # dst — координаты в результате
        dst_x_start = max(0, x_start)
        dst_y_start = max(0, y_start)
        src_x_start = max(0, -x_start)
        src_y_start = max(0, -y_start)

        # Копируем только перекрывающуюся область
        copy_w = min(w - src_x_start, width_pixels - dst_x_start)
        copy_h = min(h - src_y_start, height_pixels - dst_y_start)

        if copy_w > 0 and copy_h > 0:
            crop[dst_y_start:dst_y_start + copy_h,
                 dst_x_start:dst_x_start + copy_w] = \
                tile.image[src_y_start:src_y_start + copy_h,
                           src_x_start:src_x_start + copy_w]

        return crop

    def generate_synthetic_map(self, size: int = 1024,
                               resolution: float = 0.1) -> MapRegion:
        """
        Сгенерировать синтетическую карту для тестирования.

        Args:
            size: размер карты в пикселях
            resolution: метров на пиксель

        Returns:
            MapRegion с сгенерированной картой
        """
        # Создаём синтетическую карту с "зданиями" и "дорогами"
        img = np.zeros((size, size, 3), dtype=np.uint8)

        # Зелёная основа (трава)
        img[:, :] = [34, 139, 34]  # forest green

        # Серые "здания" (квадраты)
        np.random.seed(42)
        for _ in range(50):
            x = np.random.randint(50, size - 50)
            y = np.random.randint(50, size - 50)
            w = np.random.randint(20, 80)
            h = np.random.randint(20, 80)
            img[y:y+h, x:x+w] = [128, 128, 128]  # серый

        # Белые "дороги"
        for i in range(0, size, 100):
            img[i:i+5, :] = [255, 255, 255]
            img[:, i:i+5] = [255, 255, 255]

        # Синие "водоёмы"
        for _ in range(5):
            cx, cy = np.random.randint(100, size-100, 2)
            r = np.random.randint(30, 80)
            y, x = np.ogrid[:size, :size]
            mask = (x - cx)**2 + (y - cy)**2 < r**2
            img[mask] = [30, 144, 255]

        tile = MapTile(
            image=img,
            lat=55.7550,
            lon=37.6173,
            resolution=resolution,
            tile_size=size
        )

        self._region = MapRegion(
            tiles={(0, 0): tile},
            min_lat=55.7500,
            max_lat=55.7600,
            min_lon=37.6100,
            max_lon=37.6200,
            resolution=resolution
        )
        return self._region
