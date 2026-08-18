"""
Загрузка реальных спутниковых снимков из открытых источников.
Поддерживает: OpenStreetMap (OSM), Mapbox, Google Maps (через тайлы).
"""

import os
import math
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Tuple, Optional
from map_loader import MapLoader, MapRegion, MapTile


class TileDownloader:
    """
    Скачивает тайлы карт из различных источников.
    """

    # Тайлинг схемы (Web Mercator / EPSG:3857)
    ZOOM_LEVELS = {
        'low': 12,      # ~300м/пиксель
        'medium': 14,   # ~75м/пиксель
        'high': 16,     # ~18м/пиксель
        'ultra': 18,    # ~4м/пиксель
    }

    SOURCES = {
        'osm': 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
        'carto': 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}_retina.png',
        'esri': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    }

    def __init__(self, source: str = 'esri', cache_dir: str = None):
        """
        Args:
            source: 'osm' (дороги), 'carto' (светлая), 'esri' (спутник)
            cache_dir: директория для кэша тайлов
        """
        self.source = source
        self.url_template = self.SOURCES.get(source, self.SOURCES['esri'])
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser('~'), '.aerial_nav_tiles')
        os.makedirs(self.cache_dir, exist_ok=True)
        self._downloaded = 0

    def _tile_to_latlon(self, tile_x: int, tile_y: int, zoom: int) -> Tuple[float, float]:
        """Конвертирует координаты тайла в широту/долготу."""
        n = 2.0 ** zoom
        lon_deg = tile_x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n)))
        lat_deg = math.degrees(lat_rad)
        return lat_deg, lon_deg

    def _lonlat_to_tile(self, lon: float, lat: float, zoom: int) -> Tuple[int, int]:
        """Конвертирует широту/долготу в координаты тайла."""
        n = 2.0 ** zoom
        tile_x = int((lon + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat)
        tile_y = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
        return tile_x, tile_y

    def _get_cache_path(self, tile_x: int, tile_y: int, zoom: int) -> str:
        """Получает путь к закэшированному тайлу."""
        cache_path = os.path.join(self.cache_dir, f'{self.source}_{zoom}_{tile_x}_{tile_y}.png')
        return cache_path

    def download_tile(self, tile_x: int, tile_y: int, zoom: int) -> Optional[np.ndarray]:
        """
        Скачивает один тайл.

        Returns:
            numpy array (HxWx3) или None при ошибке
        """
        cache_path = self._get_cache_path(tile_x, tile_y, zoom)

        # Проверяем кэш
        if os.path.exists(cache_path):
            img = Image.open(cache_path).convert('RGB')
            return np.array(img)

        # Скачиваем
        url = self.url_template.format(z=zoom, x=tile_x, y=tile_y)

        try:
            headers = {
                'User-Agent': 'AerialNav/1.0 (educational project)'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            img = Image.open(BytesIO(response.content)).convert('RGB')
            img.save(cache_path)
            self._downloaded += 1

            return np.array(img)
        except Exception as e:
            print(f"[TileDownloader] Ошибка загрузки тайла ({tile_x},{tile_y},{zoom}): {e}")
            return None

    def download_region(self, lat: float, lon: float,
                        radius_km: float = 1.0,
                        zoom: int = None) -> Tuple[np.ndarray, float]:
        """
        Скачивает регион карты вокруг заданных координат.

        Args:
            lat: широта центра
            lon: долгота центра
            radius_km: радиус региона в км
            zoom: уровень детализации (если None, подбирается автоматически)

        Returns:
            (изображение региона, разрешение в м/пиксель)
        """
        if zoom is None:
            zoom = self.ZOOM_LEVELS['medium']

        # Вычисляем радиус в тайлах
        # На экваторе 1 градус ≈ 111 км
        degrees_radius = radius_km / 111.0
        tile_x_center, tile_y_center = self._lonlat_to_tile(lon, lat, zoom)

        # Определяем диапазон тайлов
        # На уровне zoom 14 один тайл ≈ 75м, на zoom 16 ≈ 18м
        tile_size_deg = 360.0 / (2 ** zoom)
        tiles_radius = int(math.ceil(degrees_radius / tile_size_deg))

        # Скачиваем все тайлы региона
        tiles = []
        for dx in range(-tiles_radius, tiles_radius + 1):
            row = []
            for dy in range(-tiles_radius, tiles_radius + 1):
                tx = tile_x_center + dx
                ty = tile_y_center + dy

                # Проверяем границы тайлов
                if ty < 0 or ty >= 2 ** zoom:
                    row.append(None)
                    continue

                tile = self.download_tile(tx, ty, zoom)
                row.append(tile)
            tiles.append(row)

        # Склеиваем тайлы в одно изображение
        if not tiles or not tiles[0]:
            return None, 0.0

        # Определяем размер
        tile_h = tiles[0][0].shape[0] if tiles[0][0] is not None else 256
        tile_w = tiles[0][0].shape[1] if tiles[0][0] is not None else 256
        rows = len(tiles)
        cols = len(tiles[0])

        full_img = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

        for r, row_tiles in enumerate(tiles):
            for c, tile in enumerate(row_tiles):
                if tile is not None:
                    full_img[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w] = tile

        # Вычисляем разрешение
        # На экваторе на уровне zoom: 1 пиксель ≈ 154 / 2^zoom километров
        resolution_km = 154.0 / (2 ** zoom)
        resolution = resolution_km * 1000.0  # конвертируем в метры

        print(f"[TileDownloader] Скачано {self._downloaded} тайлов, разрешение: {resolution:.2f} м/пиксель")

        return full_img, resolution

    def clear_cache(self):
        """Очищает кэш тайлов."""
        if os.path.exists(self.cache_dir):
            for f in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, f))
            print("[TileDownloader] Кэш очищен")


class RealMapLoader(MapLoader):
    """
    Расширенный загрузчик карт с поддержкой реальных спутниковых снимков.
    """

    def __init__(self, tile_source: str = 'esri', cache_dir: str = None):
        super().__init__(cache_dir)
        self.downloader = TileDownloader(source=tile_source, cache_dir=cache_dir)

    def load_real_map(self, lat: float, lon: float,
                      radius_km: float = 2.0,
                      zoom: int = None) -> MapRegion:
        """
        Загрузить реальную карту из интернета.

        Args:
            lat: широта центра
            lon: долгота центра
            radius_km: радиус региона в км
            zoom: уровень детализации

        Returns:
            MapRegion с загруженной картой
        """
        print(f"[RealMapLoader] Загрузка карты: lat={lat}, lon={lon}, radius={radius_km}км")

        img, resolution = self.downloader.download_region(lat, lon, radius_km, zoom)

        if img is None:
            raise ValueError("Не удалось загрузить карту")

        tile = MapTile(
            image=img,
            lat=lat,
            lon=lon,
            resolution=resolution,
            tile_size=img.shape[0]
        )

        self._region = MapRegion(
            tiles={(0, 0): tile},
            min_lat=lat - radius_km / 111.0,
            max_lat=lat + radius_km / 111.0,
            min_lon=lon - radius_km / 111.0,
            max_lon=lon + radius_km / 111.0,
            resolution=resolution
        )

        print(f"[RealMapLoader] Карта загружена: {img.shape[1]}x{img.shape[0]} пикселей, "
              f"{resolution:.2f} м/пиксель")
        return self._region

    def save_map_to_file(self, output_path: str):
        """Сохраняет загруженную карту в файл."""
        if self._region is None:
            raise ValueError("Карта не загружена")

        tile = list(self._region.tiles.values())[0]
        img = Image.fromarray(tile.image)
        img.save(output_path)
        print(f"[RealMapLoader] Карта сохранена в {output_path}")
