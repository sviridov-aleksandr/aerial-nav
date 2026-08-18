"""
Route Strip Map Downloader — скачивание карты вдоль маршрута.

Логика:
  1. Вокруг каждой точки — квадрат 1км×1км
  2. Между точками — полоса 500м шириной вдоль линии
  3. Склеиваем: точка0, полоса0, точка1, полоса1, ..., точкаN

Оптимизация: сегменты склеиваются по частям, не загружая всё в RAM.
"""

import numpy as np
from PIL import Image, ImageDraw
import os
import math
import argparse
import requests
from io import BytesIO
import time
import sys

# Убираем лимит PIL на размер изображений
Image.MAX_IMAGE_PIXELS = None


class ProgressBar:
    """Консольный прогресс-бар."""

    def __init__(self, total, label=""):
        self.total = total
        self.current = 0
        self.label = label
        self.start_time = time.time()
        self.bar_width = 40
        self.last_print = 0
        self.tile_count = 0  # счётчик тайлов для печати

    def update(self, n=1):
        self.current += n
        self.tile_count += n
        now = time.time()

        # Печатаем каждые 20 тайлов или каждые 2 секунды
        if self.tile_count < 20 and now - self.last_print < 2:
            return
        self.last_print = now
        self.tile_count = 0

        elapsed = now - self.start_time
        progress = self.current / self.total if self.total > 0 else 0
        filled = int(self.bar_width * progress)
        remaining = (elapsed / self.current * (self.total - self.current)) if self.current > 0 else 0

        bar = '█' * filled + '░' * (self.bar_width - filled)
        speed = self.current / elapsed if elapsed > 0 else 0

        print(f"[{self.label}] [{bar}] {progress*100:.1f}% ({self.current}/{self.total}) "
              f"{speed:.1f}/s  ETA: {remaining:.0f}s")

    def finish(self):
        elapsed = time.time() - self.start_time
        print(f"[{self.label}] [DONE] {self.current}/{self.total} ({elapsed:.1f}s)")


class RouteStripMapDownloader:
    """Скачивание карты вдоль маршрута полосками."""

    def __init__(self, route_gps, resolution_m=0.5, source='esri',
                 point_patch_km=1.0, segment_width_m=500):
        self.route_gps = route_gps
        self.resolution_m = resolution_m
        self.source = source
        self.point_patch_km = point_patch_km
        self.point_patch_m = point_patch_km * 1000
        self.segment_width_m = segment_width_m

        self.zoom = self._get_zoom()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            'Referer': 'https://www.arcgis.com/',
        })

        total_dist = 0
        for i in range(len(route_gps) - 1):
            total_dist += self._haversine(
                route_gps[i][0], route_gps[i][1],
                route_gps[i+1][0], route_gps[i+1][1])

        print(f"[StripMap] Точек: {len(route_gps)}")
        print(f"[StripMap] Общая длина: {total_dist/1000:.1f} км")
        print(f"[StripMap] Квадрат у точки: {point_patch_km}×{point_patch_km} км")
        print(f"[StripMap] Полоса между точками: {segment_width_m}м ширина")
        print(f"[StripMap] Zoom: {self.zoom}, Resolution: {resolution_m} м/px")
        print(f"[StripMap] Источник: {source}")

    def _get_zoom(self):
        if self.resolution_m <= 0.3:
            return 20
        elif self.resolution_m <= 0.6:
            return 19
        else:
            return 18

    def _haversine(self, lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = (np.sin(dphi/2)**2 +
             np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2)
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    def _latlon_to_tile(self, lat, lon, zoom):
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat)
        y = int((1.0 - math.log(math.tan(lat_rad) +
                 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
        return x, y

    def _tile_to_latlon(self, x, y, zoom):
        n = 2.0 ** zoom
        lon = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        return math.degrees(lat_rad), lon

    def _get_tile_url(self, x, y, zoom):
        if self.source == 'esri':
            return (f"https://server.arcgisonline.com/ArcGIS/rest/"
                    f"services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}")
        elif self.source == 'google':
            return (f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={zoom}")
        elif self.source == 'yandex':
            return (f"https://core-renderer-tiles.maps.yandex.net/tiles?"
                    f"l=sat&x={x}&y={y}&z={zoom}")
        else:
            return f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"

    def _download_tile(self, x, y, zoom, cache_dir=None, rate_limit=0.2):
        """Скачивает один тайл с кэшированием."""
        if cache_dir:
            cache_path = os.path.join(cache_dir, f"z{zoom}_{x}_{y}.png")
            if os.path.exists(cache_path):
                return np.array(Image.open(cache_path).convert('RGB'))

        url = self._get_tile_url(x, y, zoom)
        for attempt in range(8):
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 200:
                    img = Image.open(BytesIO(resp.content)).convert('RGB')
                    arr = np.array(img)
                    if cache_dir:
                        os.makedirs(cache_dir, exist_ok=True)
                        img.save(cache_path)
                    return arr
                elif resp.status_code in (400, 403, 429):
                    wait = 5 * (attempt + 1)
                    print(f"    ⚠ HTTP {resp.status_code} ({x},{y}) "
                          f"ожидание {wait}с...")
                    time.sleep(wait)
                else:
                    print(f"    ⚠ HTTP {resp.status_code} ({x},{y})")
                    time.sleep(3)
            except Exception as e:
                print(f"    ⚠ Исключение ({x},{y}): {e}")
                time.sleep(5)
            time.sleep(rate_limit)
        return None

    def _download_point_patch(self, lat, lon, idx, cache_base):
        """Скачивает квадрат 1км×1км вокруг точки."""
        half = self.point_patch_m / 2
        cos_lat = np.cos(np.radians(lat))

        lat_offset = half / 111320.0
        lon_offset = half / (111320.0 * cos_lat)

        min_lat = lat - lat_offset
        max_lat = lat + lat_offset
        min_lon = lon - lon_offset
        max_lon = lon + lon_offset

        cache_dir = os.path.join(cache_base, f"point_{idx}")
        return self._download_bbox(
            min_lat, max_lat, min_lon, max_lon,
            f"Точка {idx+1}", cache_dir
        )

    def _download_segment_strip(self, lat1, lon1, lat2, lon2, idx, cache_base):
        """
        Скачивает полосу вдоль линии между двумя точками.
        Ширина полосы = segment_width_m.
        Возвращает путь к временному файлу сегмента.
        """
        dist = self._haversine(lat1, lon1, lat2, lon2)

        strip_dist = dist - self.point_patch_m
        if strip_dist <= 0:
            print(f"    ⚠ Сегмент {idx+1}: расстояние {dist/1000:.2f} км < "
                  f"{self.point_patch_km} км, пропускаем")
            return None

        # Разбиваем на чанки по 200м
        chunk_size_m = 200
        num_chunks = max(1, int(strip_dist / chunk_size_m))
        chunk_len = strip_dist / num_chunks

        # Генерируем точки вдоль полосы
        points_along = []
        for i in range(num_chunks + 1):
            t = (self.point_patch_m / 2 + i * chunk_len) / dist
            lat = lat1 + t * (lat2 - lat1)
            lon = lon1 + t * (lon2 - lon1)
            points_along.append((lat, lon))

        # Скачиваем тайлы для каждого чанка
        tiles = {}
        total_chunks = len(points_along) - 1
        bar = ProgressBar(total_chunks, f"Сегмент {idx+1}")

        for ci in range(total_chunks):
            lat_a, lon_a = points_along[ci]
            lat_b, lon_b = points_along[ci + 1]

            cos_lat = np.cos(np.radians((lat_a + lat_b) / 2))
            half_w = self.segment_width_m / 2

            lat_offset = (chunk_len / 2 + half_w) / 111320.0
            lon_offset = (chunk_len / 2 + half_w) / (111320.0 * cos_lat)

            c_lat = (lat_a + lat_b) / 2
            c_lon = (lon_a + lon_b) / 2

            min_lat = c_lat - lat_offset
            max_lat = c_lat + lat_offset
            min_lon = c_lon - lon_offset
            max_lon = c_lon + lon_offset

            tx_min, ty_min = self._latlon_to_tile(max_lat, min_lon, self.zoom)
            tx_max, ty_max = self._latlon_to_tile(min_lat, max_lon, self.zoom)

            ty_lo, ty_hi = min(ty_min, ty_max), max(ty_min, ty_max)
            tx_lo, tx_hi = min(tx_min, tx_max), max(tx_min, tx_max)

            cache_dir = os.path.join(cache_base, f"seg_{idx}_chunk{ci}")

            for ty in range(ty_lo, ty_hi + 1):
                for tx in range(tx_lo, tx_hi + 1):
                    if (tx, ty) not in tiles:
                        tile = self._download_tile(tx, ty, self.zoom, cache_dir, rate_limit=0.3)
                        if tile is not None:
                            tiles[(tx, ty)] = tile
                    time.sleep(0.1)

            bar.update(1)
            time.sleep(2)

        bar.finish()

        if not tiles:
            print(f"    ✗ Нет тайлов для сегмента {idx+1}")
            return None

        # Определяем bbox всей полосы
        cos_lat = np.cos(np.radians((lat1 + lat2) / 2))
        half_w = self.segment_width_m / 2

        lat_offset = (strip_dist / 2 + half_w) / 111320.0
        lon_offset = (strip_dist / 2 + half_w) / (111320.0 * cos_lat)

        c_lat = (lat1 + lat2) / 2
        c_lon = (lon1 + lon2) / 2

        min_lat = c_lat - lat_offset
        max_lat = c_lat + lat_offset
        min_lon = c_lon - lon_offset
        max_lon = c_lon + lon_offset

        height_m = (max_lat - min_lat) * 111320.0
        width_m = (max_lon - min_lon) * 111320.0 * cos_lat

        px_w = int(width_m / self.resolution_m)
        px_h = int(height_m / self.resolution_m)

        # Сохраняем сегмент во временный файл
        temp_path = os.path.join(cache_base, f"seg_{idx}_temp.png")
        img = Image.new('RGB', (px_w, px_h), (100, 100, 100))

        for (tx, ty), tile_arr in tiles.items():
            tile_img = Image.fromarray(tile_arr)
            tile_lat, tile_lon = self._tile_to_latlon(tx, ty, self.zoom)

            dy = (tile_lat - c_lat) / (max_lat - min_lat) * px_h
            dx = (tile_lon - c_lon) / (max_lon - min_lon) * px_w

            px = int(px_w / 2 + dx - 128)
            py = int(px_h / 2 + dy - 128)

            img.paste(tile_img, (px, py))

        img.save(temp_path)
        print(f"    ✓ Сегмент {idx+1}: {px_w}x{px_h} px, "
              f"тайлов: {len(tiles)}, сохранён: {temp_path}")
        return temp_path

    def _download_bbox(self, min_lat, max_lat, min_lon, max_lon,
                       label, cache_dir):
        """Скачивает тайлы для bbox и склеивает в изображение."""
        tx_min, ty_min = self._latlon_to_tile(max_lat, min_lon, self.zoom)
        tx_max, ty_max = self._latlon_to_tile(min_lat, max_lon, self.zoom)

        ty_lo, ty_hi = min(ty_min, ty_max), max(ty_min, ty_max)
        tx_lo, tx_hi = min(tx_min, tx_max), max(tx_min, tx_max)

        tiles = {}
        total_tiles = (tx_hi - tx_lo + 1) * (ty_hi - ty_lo + 1)
        downloaded = 0

        print(f"    [{label}] Тайлов: {total_tiles}")
        bar = ProgressBar(total_tiles, label)

        for ty in range(ty_lo, ty_hi + 1):
            for tx in range(tx_lo, tx_hi + 1):
                tile = self._download_tile(tx, ty, self.zoom, cache_dir)
                if tile is not None:
                    tiles[(tx, ty)] = tile
                    downloaded += 1
                    bar.update(1)
                time.sleep(0.12)

        bar.finish()

        if downloaded == 0:
            print(f"    ✗ Нет тайлов для {label}")
            return None

        cos_lat = np.cos(np.radians((min_lat + max_lat) / 2))
        height_m = (max_lat - min_lat) * 111320.0
        width_m = (max_lon - min_lon) * 111320.0 * cos_lat

        px_w = int(width_m / self.resolution_m)
        px_h = int(height_m / self.resolution_m)

        img = Image.new('RGB', (px_w, px_h), (100, 100, 100))

        for (tx, ty), tile_arr in tiles.items():
            tile_img = Image.fromarray(tile_arr)
            tile_lat, tile_lon = self._tile_to_latlon(tx, ty, self.zoom)

            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2

            dy = (tile_lat - center_lat) / (max_lat - min_lat) * px_h
            dx = (tile_lon - center_lon) / (max_lon - min_lon) * px_w

            px = int(px_w / 2 + dx - 128)
            py = int(px_h / 2 + dy - 128)

            img.paste(tile_img, (px, py))

        return img, px_w, px_h

    def download_all(self, output_path,
                     cache_base='/home/alex/aerial-nav/map_cache/strips'):
        """Скачивает все участки и склеивает в одну ленту."""
        print(f"\n{'='*70}")
        print(f"СКАЧИВАНИЕ КАРТЫ ВДОЛЬ МАРШРУТА")
        print(f"{'='*70}")

        # Порядок: точка0, сегмент0, точка1, сегмент1, ..., точкаN
        parts = []  # list of (type, img_or_path, width, height, idx)

        # Точки
        total_parts = len(self.route_gps) + len(self.route_gps) - 1
        bar = ProgressBar(total_parts, "Общий прогресс")

        for i, (lat, lon) in enumerate(self.route_gps):
            result = self._download_point_patch(lat, lon, i, cache_base)
            if result:
                img, w, h = result
                parts.append(('point', img, w, h, i))
            bar.update(1)

        # Сегменты (сохраняются во временные файлы)
        for i in range(len(self.route_gps) - 1):
            lat1, lon1 = self.route_gps[i]
            lat2, lon2 = self.route_gps[i + 1]
            temp_path = self._download_segment_strip(
                lat1, lon1, lat2, lon2, i, cache_base)
            if temp_path:
                img = Image.open(temp_path)
                w, h = img.size
                parts.append(('segment', temp_path, w, h, i))
            bar.update(1)

        if not parts:
            raise Exception("Не удалось скачать ни одного участка")

        # Склеиваем по частям, не загружая всё в RAM
        print(f"\n[StripMap] Склеивание {len(parts)} участков...")

        # Сначала определим максимальную высоту
        max_h = max(p[2] if isinstance(p[1], str) else p[1].size[1] for p in parts)

        # Открываем итоговый файл
        total_w = sum(p[2] for p in parts)
        full_strip = Image.new('RGB', (total_w, max_h), (100, 100, 100))

        x_offset = 0
        for kind, data, w, h, idx in parts:
            y_offset = (max_h - h) // 2

            if kind == 'point':
                img = data
            else:
                img = Image.open(data)

            full_strip.paste(img, (x_offset, y_offset))

            label = f"Точка {idx+1}" if kind == 'point' else f"Сегмент {idx+1}"
            print(f"  {label}: {w}x{h}")

            # Освобождаем память
            if kind == 'point':
                del img
            else:
                img.close()
                os.remove(data)  # Удаляем временный файл

            x_offset += w

        # Рисуем линию маршрута поверх
        print(f"\n[StripMap] Рисуем маршрут...")
        draw = ImageDraw.Draw(full_strip)

        point_xs = []
        x_off = 0
        for kind, data, w, h, idx in parts:
            if kind == 'point':
                point_xs.append(x_off + w // 2)
            x_off += w

        for i in range(len(point_xs) - 1):
            x1, y1 = point_xs[i], full_strip.size[1] // 2
            x2, y2 = point_xs[i+1], full_strip.size[1] // 2
            draw.line([(x1, y1), (x2, y2)], fill='red', width=4)
            for x in [x1, x2]:
                r = 6
                draw.ellipse([x-r, y1-r, x+r, y1+r],
                             fill='yellow', outline='red')

        # Сохраняем
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        full_strip.save(output_path)

        file_size = os.path.getsize(output_path) / 1e6
        print(f"\n{'='*70}")
        print(f"ГОТОВО!")
        print(f"{'='*70}")
        print(f"  Участков: {len(parts)}")
        print(f"  Размер ленты: {full_strip.size[0]}x{full_strip.size[1]} px")
        print(f"  Файл: {output_path}")
        print(f"  Размер файла: {file_size:.1f} MB")
        print(f"{'='*70}")

        return full_strip


def main():
    parser = argparse.ArgumentParser(
        description='Скачивание карты вдоль маршрута')
    parser.add_argument('--route', nargs='+', required=True,
                        help='Точки: lat1,lon1 lat2,lon2 ...')
    parser.add_argument('--resolution', type=float, default=0.5,
                        help='Разрешение м/пиксель (default: 0.5)')
    parser.add_argument('--source', choices=['esri', 'osm', 'google', 'yandex'], default='esri',
                        help='Источник: esri, osm, google, yandex (default: esri)')
    parser.add_argument('--point-size', type=float, default=1.0,
                        help='Квадрат у точки км (default: 1.0)')
    parser.add_argument('--seg-width', type=float, default=500,
                        help='Ширина полосы между точками м (default: 500)')
    parser.add_argument('--output', type=str, default='route_strip.png',
                        help='Путь сохранения (default: route_strip.png)')

    args = parser.parse_args()

    route_gps = []
    for point in args.route:
        lat, lon = point.split(',')
        route_gps.append((float(lat), float(lon)))

    downloader = RouteStripMapDownloader(
        route_gps=route_gps,
        resolution_m=args.resolution,
        source=args.source,
        point_patch_km=args.point_size,
        segment_width_m=args.seg_width
    )

    try:
        downloader.download_all(args.output)
    except Exception as e:
        print(f"\n✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
