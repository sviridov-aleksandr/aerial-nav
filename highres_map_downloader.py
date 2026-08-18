"""
Скачивание карты высокого разрешения для навигации.
Использует ESRI World Imagery (бесплатно, без API ключа).
"""

import os
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import math
import time


class HighResMapDownloader:
    """
    Скачивает карты высокого разрешения из открытых источников.
    
    Источники:
    - ESRI World Imagery (бесплатно)
    - OpenStreetMap (бесплатно)
    """

    def __init__(self, cache_dir='/home/alex/aerial-nav/map_cache/highres'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # ESRI World Imagery URL template
        self.esri_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
        
        # Заголовки для обхода rate limiting (ESRI требует User-Agent)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/120.0 Safari/537.36',
            'Referer': 'https://www.arcgis.com/',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def latlon_to_tile(self, lat: float, lon: float, zoom: int) -> tuple:
        """Конвертация lat/lon в tile координаты."""
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat)
        y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
        return x, y, zoom

    def download_esri_tile(self, x: int, y: int, zoom: int, 
                           output_path: str = None) -> np.ndarray:
        """Скачивает тайл с ESRI World Imagery с ретраями."""
        url = self.esri_url.format(zoom=zoom, y=y, x=x)
        
        if output_path and os.path.exists(output_path):
            return np.array(Image.open(output_path))

        # Ретраи при ошибках (400, 429, 5xx)
        for attempt in range(5):
            try:
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    img = img.convert('RGB')
                    array = np.array(img)
                    
                    if output_path:
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        img.save(output_path)
                    
                    return array
                elif response.status_code in (400, 403, 429):
                    # Rate limited — ждём дольше
                    wait = 2 ** attempt  # 1, 2, 4, 8, 16 сек
                    print(f"  ⚠ HTTP {response.status_code} (x={x}, y={y}), "
                          f"ожидание {wait}с...")
                    time.sleep(wait)
                else:
                    print(f"  ⚠ HTTP {response.status_code} (x={x}, y={y})")
                    time.sleep(1)
            except Exception as e:
                print(f"  ⚠ Ошибка: {e}")
                time.sleep(2 ** attempt)
        
        return None

    def download_region(self, lat: float, lon: float,
                        size_km: float = 2.0,
                        zoom: int = 18) -> np.ndarray:
        """
        Скачивает карту региона высокого разрешения.
        
        Args:
            lat, lon: центр региона
            size_km: размер области в км
            zoom: zoom level (17-19 для детализации 0.5-2 м/пиксель)
        
        Returns:
            Карта (H, W, 3) RGB
        """
        # Расчёт количества тайлов
        # На zoom 18: 1 тайл = ~150 м
        tiles_per_km = 7
        tiles_needed = int(size_km * tiles_per_km)
        tile_size = 256  # ESRI tile size
        
        print(f"[MapDownloader] Скачивание карты:")
        print(f"  Центр: ({lat:.6f}, {lon:.6f})")
        print(f"  Размер: {size_km*1000:.0f}×{size_km*1000:.0f} м")
        print(f"  Zoom: {zoom}")
        print(f"  Тайлов: {tiles_needed}×{tiles_needed}")

        x_center, y_center, z = self.latlon_to_tile(lat, lon, zoom)
        
        # Собираем тайлы в словарь
        tiles_dict = {}
        downloaded = 0

        for dy in range(-tiles_needed//2, tiles_needed//2 + 1):
            for dx in range(-tiles_needed//2, tiles_needed//2 + 1):
                x = x_center + dx
                y = y_center + dy
                
                if y < 0 or y >= 2**z:
                    continue

                tile_path = os.path.join(
                    self.cache_dir,
                    f"esri_z{z}_{x}_{y}.png"
                )

                tile = self.download_esri_tile(x, y, z, tile_path)
                
                if tile is not None:
                    tiles_dict[(dx, dy)] = tile
                    downloaded += 1
                
                time.sleep(0.2)  # Rate limiting (0.2с между запросами)

        if downloaded == 0:
            raise Exception("Не удалось скачать ни одного тайла")

        # Создаём итоговую карту
        total_size = tiles_needed * tile_size
        map_array = np.zeros((total_size, total_size, 3), dtype=np.uint8)

        for (dx, dy), tile in tiles_dict.items():
            map_y = (dy + tiles_needed//2) * tile_size
            map_x = (dx + tiles_needed//2) * tile_size
            
            # Проверяем границы
            y_end = min(map_y + tile.shape[0], total_size)
            x_end = min(map_x + tile.shape[1], total_size)
            y_start = max(0, map_y)
            x_start = max(0, map_x)
            
            if y_end > y_start and x_end > x_start:
                tile_y_start = max(0, -map_y)
                tile_x_start = max(0, -map_x)
                map_array[y_start:y_end, x_start:x_end] = \
                    tile[tile_y_start:tile_y_start+(y_end-y_start), 
                         tile_x_start:tile_x_start+(x_end-x_start)]

        # Сохраняем
        output_path = os.path.join(
            self.cache_dir, 
            f"highres_{lat:.4f}_{lon:.4f}_z{zoom}.png"
        )
        Image.fromarray(map_array).save(output_path)

        print(f"[MapDownloader] ✓ Карта сохранена: {output_path}")
        print(f"[MapDownloader] Размер: {map_array.shape}")
        print(f"[MapDownloader] Скачано тайлов: {downloaded}")

        return map_array

    def download_moscow_map(self) -> np.ndarray:
        """Скачивает карту Москвы высокого разрешения (5×5 км)."""
        return self.download_region(
            lat=55.7550,
            lon=37.6173,
            size_km=5.0,
            zoom=18
        )


if __name__ == '__main__':
    print("=" * 60)
    print("СКАЧИВАНИЕ КАРТЫ ВЫСОКОГО РАЗРЕШЕНИЯ")
    print("=" * 60)
    
    downloader = HighResMapDownloader()
    
    print("\nСкачивание карты Москвы (zoom 18)...")
    try:
        map_data = downloader.download_moscow_map()
        print(f"\n✓ Карта готова: {map_data.shape}")
    except Exception as e:
        print(f"\n✗ Ошибка: {e}")
        print("\nПопробуйте другой zoom (17-19) или регион.")