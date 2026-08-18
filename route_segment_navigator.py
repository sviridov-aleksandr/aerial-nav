"""
Сегментный навигатор — разбивает маршрут на участки по 300м.
Каждый сегмент обрабатывается отдельно: загружается только нужный участок карты,
после обработки память освобождается.

Использование:
  python3 route_segment_navigator.py \
    --route 46.264929,33.372986 46.279537,33.371246 ... \
    --segment 300 \
    --zoom 18
"""

import numpy as np
import math
import os
import sys
import argparse
import gc
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from PIL import Image
from io import BytesIO
import requests
import time


# ======================== УТИЛИТЫ ========================

def haversine(lat1, lon1, lat2, lon2):
    """Расстояние в метрах между двумя GPS-точками."""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


def gps_to_pixels(lat, lon, center_lat, center_lon, resolution, map_size):
    """GPS → пиксели относительно центра карты."""
    lat_m_per_deg = 111320.0
    lon_m_per_deg = 111320.0 * np.cos(np.radians(lat))
    dy = (center_lat - lat) * lat_m_per_deg
    dx = (lon - center_lon) * lon_m_per_deg
    center_px = map_size // 2
    x = int(center_px + dx / resolution)
    y = int(center_px - dy / resolution)
    return x, y


def meters_to_pixels(meters, resolution):
    """Метры → пиксели."""
    return int(meters / resolution)


# ======================== ТАЙЛ-МЕНЕДЖЕР ========================

class TileManager:
    """
    Загружает и кэширует тайлы ESRI для конкретного сегмента маршрута.
    После обработки сегмента — освобождает память.
    """

    def __init__(self, cache_dir='/home/alex/aerial-nav/map_cache/highres'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._cache = {}  # (x, y, z) → np.ndarray
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            'Referer': 'https://www.arcgis.com/',
        })

    def latlon_to_tile(self, lat, lon, zoom):
        """GPS → tile координаты."""
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat)
        y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
        return x, y

    def download_tile(self, x, y, zoom):
        """Скачать один тайл ESRI с кэшированием."""
        key = (x, y, zoom)
        if key in self._cache:
            return self._cache[key]

        # Проверяем файл на диске
        tile_path = os.path.join(self.cache_dir, f"esri_z{zoom}_{x}_{y}.png")
        if os.path.exists(tile_path):
            try:
                arr = np.array(Image.open(tile_path).convert('RGB'))
                self._cache[key] = arr
                return arr
            except Exception:
                pass

        # Скачиваем
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
        for attempt in range(3):
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 200:
                    img = Image.open(BytesIO(resp.content)).convert('RGB')
                    arr = np.array(img)
                    self._cache[key] = arr
                    # Сохраняем в кэш
                    img.save(tile_path)
                    return arr
                elif resp.status_code in (400, 403, 429):
                    wait = 2 ** attempt
                    print(f"    ⚠ Rate limit, ждём {wait}с...")
                    time.sleep(wait)
            except Exception as e:
                print(f"    ⚠ Ошибка: {e}")
                time.sleep(2)

        return None

    def load_segment_map(self, center_lat, center_lon, size_m, zoom=18):
        """
        Загрузить карту для сегмента маршрута.
        
        Args:
            center_lat, center_lon: центр сегмента
            size_m: размер сегмента в метрах (квадрат)
            zoom: zoom level
        
        Returns:
            np.ndarray (H, W, 3) RGB
        """
        # Размер в пикселях
        size_px = meters_to_pixels(size_m, 0.5)  # 0.5 м/пиксель на zoom 18
        half_px = size_px // 2

        # Центр в tile-координатах
        cx, cy = self.latlon_to_tile(center_lat, center_lon, zoom)

        # Определяем тайлы, которые покрывают сегмент
        # На zoom 18: 1 тайл = ~150м, шаг ~30м в пикселях
        tile_size_px = 256
        margin_px = int(tile_size_px * 2)  # запас 2 тайла

        # Смещение центра сегмента относительно центра тайла
        center_tile_x = cx - margin_px
        center_tile_y = cy - margin_px

        # Диапазон тайлов
        min_tx = center_tile_x - margin_px // tile_size_px
        max_tx = center_tile_x + margin_px // tile_size_px
        min_ty = center_tile_y - margin_px // tile_size_px
        max_ty = center_tile_y + margin_px // tile_size_px

        # Собираем тайлы
        tiles = {}
        for ty in range(min_ty, max_ty + 1):
            for tx in range(min_tx, max_tx + 1):
                if ty < 0 or ty >= 2**zoom:
                    continue
                tile = self.download_tile(tx, ty, zoom)
                if tile is not None:
                    tiles[(tx, ty)] = tile

        if not tiles:
            print(f"    ⚠ Не удалось скачать тайлы для ({center_lat:.6f}, {center_lon:.6f})")
            return None

        # Собираем в одну карту
        total_w = (max_tx - min_tx + 1) * tile_size_px
        total_h = (max_ty - min_ty + 1) * tile_size_px
        map_arr = np.zeros((total_h, total_w, 3), dtype=np.uint8)

        for (tx, ty), tile in tiles.items():
            mx = (tx - min_tx) * tile_size_px
            my = (ty - min_ty) * tile_size_px
            map_arr[my:my+tile.shape[0], mx:mx+tile.shape[1]] = tile

        return map_arr

    def clear_cache(self):
        """Освободить кэш тайлов."""
        self._cache.clear()
        gc.collect()


# ======================== СЕГМЕНТ МАРШРУТА ========================

@dataclass
class RouteSegment:
    """Один сегмент маршрута (300м)."""
    index: int
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    center_lat: float
    center_lon: float
    distance_m: float
    cumulative_m: float  # накопленная дистанция от начала маршрута


def split_route_into_segments(route_gps: List[Tuple[float, float]],
                               segment_length_m: float = 300.0) -> List[RouteSegment]:
    """
    Разбить маршрут на сегменты по N метров.
    
    Args:
        route_gps: список (lat, lon)
        segment_length_m: длина сегмента в метрах
    
    Returns:
        Список RouteSegment
    """
    if len(route_gps) < 2:
        return []

    # 1. Вычисляем накопленные расстояния между точками
    cumulative = [0.0]
    for i in range(len(route_gps) - 1):
        lat1, lon1 = route_gps[i]
        lat2, lon2 = route_gps[i + 1]
        d = haversine(lat1, lon1, lat2, lon2)
        cumulative.append(cumulative[-1] + d)

    total_distance = cumulative[-1]

    # 2. Интерполируем точки маршрута с шагом ~10м для точности
    step = 10.0  # метр
    interp_lats = [route_gps[0][0]]
    interp_lons = [route_gps[0][1]]
    interp_cum = [0.0]

    for i in range(len(route_gps) - 1):
        lat1, lon1 = route_gps[i]
        lat2, lon2 = route_gps[i + 1]
        seg_len = cumulative[i + 1] - cumulative[i]

        if seg_len < 1.0:
            continue

        num_steps = max(2, int(seg_len / step))
        for s in range(1, num_steps + 1):
            t = s / num_steps
            lat = lat1 + t * (lat2 - lat1)
            lon = lon1 + t * (lon2 - lon1)
            interp_lats.append(lat)
            interp_lons.append(lon)
            interp_cum.append(cumulative[i] + t * seg_len)

    # 3. Разбиваем на сегменты
    segments = []
    seg_start_idx = 0
    seg_index = 0

    while seg_start_idx < len(interp_lats):
        target_cum = interp_cum[seg_start_idx] + segment_length_m

        # Находим ближайшую точку, не превышающую target
        seg_end_idx = seg_start_idx
        for j in range(seg_start_idx + 1, len(interp_lats)):
            if interp_cum[j] <= target_cum:
                seg_end_idx = j
            else:
                break

        # Если не смогли набрать segment_length_m (конец маршрута)
        if seg_end_idx == seg_start_idx:
            seg_end_idx = min(seg_start_idx + 1, len(interp_lats) - 1)

        start_lat, start_lon = interp_lats[seg_start_idx], interp_lons[seg_start_idx]
        end_lat, end_lon = interp_lats[seg_end_idx], interp_lons[seg_end_idx]
        center_lat = (start_lat + end_lat) / 2
        center_lon = (start_lon + end_lon) / 2
        dist = interp_cum[seg_end_idx] - interp_cum[seg_start_idx]

        segments.append(RouteSegment(
            index=seg_index,
            start_lat=start_lat,
            start_lon=start_lon,
            end_lat=end_lat,
            end_lon=end_lon,
            center_lat=center_lat,
            center_lon=center_lon,
            distance_m=dist,
            cumulative_m=interp_cum[seg_start_idx]
        ))

        seg_start_idx = seg_end_idx
        seg_index += 1

    return segments


# ======================== СЕГМЕНТНЫЙ НАВИГАТОР ========================

class RouteSegmentNavigator:
    """
    Навигатор, обрабатывающий маршрут по сегментам.
    Для каждого сегмента загружает только нужный участок карты.
    """

    def __init__(self, segment_length_m: float = 300.0, zoom: int = 18):
        self.segment_length_m = segment_length_m
        self.zoom = zoom
        self.tile_manager = TileManager()
        self.segments: List[RouteSegment] = []
        self.results: List[Dict] = []

    def load_route(self, route_gps: List[Tuple[float, float]]):
        """Загрузить маршрут и разбить на сегменты."""
        print(f"\n[SegmentNavigator] Разбивка маршрута на сегменты по {self.segment_length_m}м...")
        self.segments = split_route_into_segments(route_gps, self.segment_length_m)
        print(f"[SegmentNavigator] Маршрут: {len(self.segments)} сегментов, "
              f"{sum(s.distance_m for s in self.segments)/1000:.1f} км")

        for seg in self.segments:
            print(f"  Сегмент #{seg.index}: ({seg.start_lat:.6f}, {seg.start_lon:.6f}) → "
                  f"({seg.end_lat:.6f}, {seg.end_lon:.6f}) | "
                  f"{seg.distance_m:.0f}м | cum={seg.cumulative_m:.0f}м")

    def process_segment(self, segment: RouteSegment) -> Dict:
        """
        Обработать один сегмент маршрута.
        
        Args:
            segment: сегмент маршрута
        
        Returns:
            Словарь с результатами
        """
        print(f"\n{'='*60}")
        print(f"[Segment #{segment.index}] Обработка сегмента")
        print(f"  Центр: ({segment.center_lat:.6f}, {segment.center_lon:.6f})")
        print(f"  Дистанция: {segment.distance_m:.0f} м")
        print(f"  Zoom: {self.zoom}")

        # 1. Загружаем карту для сегмента (только нужный участок!)
        # Размер карты = 2× длина сегмента (запас по краям)
        map_size_m = segment.distance_m * 2.5
        print(f"  Загрузка карты {map_size_m:.0f}×{map_size_m:.0f} м...")

        map_arr = self.tile_manager.load_segment_map(
            segment.center_lat, segment.center_lon,
            map_size_m, self.zoom
        )

        if map_arr is None:
            print(f"  ⚠ Карта не загружена, пропускаем сегмент")
            return {
                'segment': segment.index,
                'status': 'failed',
                'error': 'map_not_loaded'
            }

        print(f"  Карта: {map_arr.shape[1]}×{map_arr.shape[0]} px "
              f"({map_arr.nbytes / 1e6:.1f} MB)")

        # 2. Определяем GPS-координаты начала и конца сегмента в пикселях карты
        map_center_lat = segment.center_lat
        map_center_lon = segment.center_lon
        map_size_px = map_arr.shape[1]
        resolution = 0.5  # м/пиксель

        start_px_x, start_px_y = gps_to_pixels(
            segment.start_lat, segment.start_lon,
            map_center_lat, map_center_lon, resolution, map_size_px
        )
        end_px_x, end_px_y = gps_to_pixels(
            segment.end_lat, segment.end_lon,
            map_center_lat, map_center_lon, resolution, map_size_px
        )

        # Корректируем относительно центра карты
        center_offset = map_size_px // 2
        start_rel_x = start_px_x - center_offset
        start_rel_y = -(start_px_y - center_offset)  # инвертируем Y
        end_rel_x = end_px_x - center_offset
        end_rel_y = -(end_px_y - center_offset)

        # 3. Вычисляем направление и дистанцию в пикселях
        dx_px = end_rel_x - start_rel_x
        dy_px = end_rel_y - start_rel_y
        dist_px = np.sqrt(dx_px**2 + dy_px**2)
        dist_m = dist_px * resolution

        # Heading в градусах (от севера, по часовой)
        heading_rad = np.arctan2(dx_px, -dy_px)  # инвертируем Y для compass heading
        heading_deg = np.degrees(heading_rad) % 360

        # 4. Генерируем промежуточные точки вдоль сегмента
        num_samples = max(10, int(dist_m / 50.0))  # каждые 50м
        sample_points = []
        for i in range(num_samples + 1):
            t = i / num_samples
            px = start_rel_x + t * dx_px
            py = start_rel_y + t * dy_px
            sample_points.append((px, py))

        # 5. Для каждой точки извлекаем тайл из карты и "обрабатываем"
        # (в реальном сценарии — сравнение с камерой дрона)
        tile_size = 224  # размер тайла для навигации
        half_tile = tile_size // 2
        segment_results = []

        for i, (px, py) in enumerate(sample_points):
            # Извлекаем тайл из карты
            map_x = center_offset + int(px)
            map_y = center_offset - int(py)

            x1 = max(0, map_x - half_tile)
            y1 = max(0, map_y - half_tile)
            x2 = min(map_arr.shape[1], x1 + tile_size)
            y2 = min(map_arr.shape[0], y1 + tile_size)

            tile = map_arr[y1:y2, x1:x2]

            # Заполняем недостающие части серым фоном
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded

            # Вычисляем GPS-координаты центра тайла
            tile_center_px_x = map_x
            tile_center_px_y = map_y
            tile_center_lat = map_center_lat + (map_y - tile_center_px_y) * resolution / 111320.0
            tile_center_lon = map_center_lon + (tile_x - map_x) * resolution / (111320.0 * np.cos(np.radians(map_center_lat)))

            segment_results.append({
                'sample_idx': i,
                'pixel_x': px,
                'pixel_y': py,
                'tile_shape': tile.shape,
                'tile_mean_rgb': tile.mean(axis=(0, 1)).tolist()
            })

        # 6. Итоговые данные сегмента
        result = {
            'segment': segment.index,
            'status': 'success',
            'start_gps': (segment.start_lat, segment.start_lon),
            'end_gps': (segment.end_lat, segment.end_lon),
            'center_gps': (segment.center_lat, segment.center_lon),
            'distance_m': dist_m,
            'heading_deg': heading_deg,
            'map_size_px': map_arr.shape,
            'map_size_mb': map_arr.nbytes / 1e6,
            'num_samples': len(segment_results),
            'samples': segment_results
        }

        self.results.append(result)

        print(f"  ✓ Обработано {len(segment_results)} точек")
        print(f"  Heading: {heading_deg:.1f}°")
        print(f"  Дистанция: {dist_m:.0f} м")

        # Освобождаем память карты
        del map_arr
        gc.collect()

        return result

    def process_all_segments(self) -> List[Dict]:
        """Обработать все сегменты маршрута."""
        print(f"\n{'='*70}")
        print("ОБРАБОТКА МАРШРУТА ПО СЕГМЕНТАМ")
        print(f"{'='*70}")

        for seg in self.segments:
            result = self.process_segment(seg)

            # Освобождаем кэш тайлов после каждого сегмента
            self.tile_manager.clear_cache()

        # Итоговая статистика
        self._print_summary()

        return self.results

    def _print_summary(self):
        """Вывести итоговую статистику."""
        print(f"\n{'='*70}")
        print("ИТОГОВАЯ СТАТИСТИКА МАРШРУТА")
        print(f"{'='*70}")

        total_dist = 0
        success = 0
        failed = 0

        for r in self.results:
            if r['status'] == 'success':
                total_dist += r['distance_m']
                success += 1
            else:
                failed += 1

        print(f"  Всего сегментов: {len(self.segments)}")
        print(f"  Успешно: {success}")
        print(f"  Ошибок: {failed}")
        print(f"  Общая дистанция: {total_dist/1000:.2f} км")
        print(f"  Размер карты на сегмент: ~{self.segment_length_m * 2.5 * 2.5 * 0.5 / 1e6:.1f} MB")
        print(f"  Пиковая RAM: ~{self.segment_length_m * 2.5 * 2.5 * 0.5 * 3 / 1e6 / 1000:.1f} GB")


# ======================== MAIN ========================

def main():
    parser = argparse.ArgumentParser(description='Сегментный навигатор маршрута')
    parser.add_argument('--route', nargs='+', required=True,
                        help='Точки маршрута: lat1,lon1 lat2,lon2 ...')
    parser.add_argument('--segment', type=float, default=300.0,
                        help='Длина сегмента в метрах (default: 300)')
    parser.add_argument('--zoom', type=int, default=18,
                        help='Zoom level (default: 18)')
    parser.add_argument('--output', type=str, default=None,
                        help='Файл для сохранения результатов (JSON)')

    args = parser.parse_args()

    # Парсим маршрут
    route_gps = []
    for point in args.route:
        lat, lon = point.split(',')
        route_gps.append((float(lat), float(lon)))

    if len(route_gps) < 2:
        print("Ошибка: маршрут должен содержать минимум 2 точки")
        sys.exit(1)

    # Создаём навигатор
    navigator = RouteSegmentNavigator(
        segment_length_m=args.segment,
        zoom=args.zoom
    )

    # Загружаем маршрут
    navigator.load_route(route_gps)

    # Обрабатываем все сегменты
    results = navigator.process_all_segments()

    # Сохраняем результаты
    if args.output:
        import json
        # Убираем samples из сохранения (слишком много данных)
        save_results = []
        for r in results:
            save_r = {k: v for k, v in r.items() if k != 'samples'}
            save_results.append(save_r)

        with open(args.output, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"\nРезультаты сохранены: {args.output}")


if __name__ == '__main__':
    main()
