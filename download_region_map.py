#!/usr/bin/env python3
"""
download_region_map.py — скачивание региональной карты (Google Satellite)
для curriculum learning сиамской сети.

Скачивает тайлы zoom 19 (0.5 м/px) для заданного bbox, кэширует их,
затем склеивает в GeoTIFF (как stitch_map.py).

Использование:
  python3 download_region_map.py \
    --min-lat 46.19 --max-lat 46.34 \
    --min-lon 33.18 --max-lon 33.44 \
    --zoom 19 \
    --cache .aerial_nav_tiles/region_google_z19 \
    --output map_cache/region_google.tif
"""

import os
import sys
import math
import time
import argparse
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin

Image.MAX_IMAGE_PIXELS = None


def latlon_to_tile(lat, lon, zoom):
    """Конвертация lat/lon в координаты тайла."""
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) +
             1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_to_latlon(x, y, zoom):
    """Конвертация координат тайла в lat/lon (верхний левый угол)."""
    n = 2.0 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


class RegionMapDownloader:
    """Скачивание региональной карты из Google Satellite."""

    def __init__(self, min_lat, max_lat, min_lon, max_lon, zoom=19,
                 cache_dir=None, threads=4, rate_limit=0.25):
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.zoom = zoom
        self.cache_dir = cache_dir
        self.threads = threads
        self.rate_limit = rate_limit

        # Диапазон тайлов
        self.tx_min, self.ty_max = latlon_to_tile(max_lat, min_lon, zoom)
        self.tx_max, self.ty_min = latlon_to_tile(min_lat, max_lon, zoom)
        self.tx_lo, self.tx_hi = min(self.tx_min, self.tx_max), max(self.tx_min, self.tx_max)
        self.ty_lo, self.ty_hi = min(self.ty_min, self.ty_max), max(self.ty_min, self.ty_max)

        self.total_tiles = (self.tx_hi - self.tx_lo + 1) * (self.ty_hi - self.ty_lo + 1)

        # Гео-размеры
        self.width_km = (self.tx_hi - self.tx_lo + 1) * 256 * 0.5 / 1000
        self.height_km = (self.ty_hi - self.ty_lo + 1) * 256 * 0.5 / 1000

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/120.0 Safari/537.36',
            'Referer': 'https://maps.google.com/',
        })

        self.stats = {'downloaded': 0, 'cached': 0, 'failed': 0, 'errors': 0}
        self.start_time = time.time()

    def tile_path(self, tx, ty):
        if self.cache_dir:
            return os.path.join(self.cache_dir, f"z{self.zoom}_{tx}_{ty}.png")
        return None

    def download_tile(self, tx, ty):
        """Скачивает один тайл с кэшированием и ретраями."""
        path = self.tile_path(tx, ty)
        if path and os.path.exists(path):
            self.stats['cached'] += 1
            return True

        url = (f"https://mt1.google.com/vt/lyrs=s&x={tx}&y={ty}&z={self.zoom}")
        for attempt in range(8):
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 200 and len(resp.content) > 500:
                    img = Image.open(BytesIO(resp.content)).convert('RGB')
                    if path:
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        img.save(path)
                    self.stats['downloaded'] += 1
                    return True
                elif resp.status_code in (400, 403, 429):
                    wait = 5 * (attempt + 1)
                    print(f"    ⚠ HTTP {resp.status_code} ({tx},{ty}) "
                          f"ожидание {wait}с...", flush=True)
                    time.sleep(wait)
                else:
                    print(f"    ⚠ HTTP {resp.status_code} ({tx},{ty})", flush=True)
                    time.sleep(3)
            except Exception as e:
                self.stats['errors'] += 1
                print(f"    ⚠ Ошибка ({tx},{ty}): {e}", flush=True)
                time.sleep(5)
            time.sleep(self.rate_limit)

        self.stats['failed'] += 1
        return False

    def print_progress(self):
        """Печатает прогресс."""
        elapsed = time.time() - self.start_time
        done = self.stats['downloaded'] + self.stats['cached'] + self.stats['failed']
        pct = done / self.total_tiles * 100 if self.total_tiles else 0
        speed = done / elapsed if elapsed > 0 else 0
        eta = (self.total_tiles - done) / speed if speed > 0 else 0
        print(f"  [{pct:.1f}%] {done}/{self.total_tiles} "
              f"({speed:.1f} т/с, ETA {eta/60:.0f} мин) "
              f"новых={self.stats['downloaded']} кэш={self.stats['cached']} "
              f"ошибок={self.stats['errors']}", flush=True)

    def download_all(self):
        """Скачивает все тайлы параллельно."""
        print(f"[Region] Регион: lat {self.min_lat:.4f}-{self.max_lat:.4f}, "
              f"lon {self.min_lon:.4f}-{self.max_lon:.4f}")
        print(f"[Region] Zoom: {self.zoom}, "
              f"размер: {self.width_km:.1f}×{self.height_km:.1f} км")
        print(f"[Region] Тайлов: {self.total_tiles} "
              f"(x {self.tx_lo}-{self.tx_hi}, y {self.ty_lo}-{self.ty_hi})")
        print(f"[Region] Потоков: {self.threads}, rate_limit: {self.rate_limit}с")
        print()

        tasks = [(tx, ty) for ty in range(self.ty_lo, self.ty_hi + 1)
                 for tx in range(self.tx_lo, self.tx_hi + 1)]

        # Обрабатываем в порядке столбцов/строк
        done = 0
        last_print = 0
        with ThreadPoolExecutor(max_workers=self.threads) as pool:
            futures = {pool.submit(self.download_tile, tx, ty): (tx, ty)
                       for tx, ty in tasks}
            for future in as_completed(futures):
                future.result()
                done += 1
                now = time.time()
                if now - last_print > 10:
                    self.print_progress()
                    last_print = now

        self.print_progress()
        print(f"\n[Region] Готово: скачано {self.stats['downloaded']}, "
              f"кэш {self.stats['cached']}, ошибок {self.stats['errors']}")

    def stitch(self, output_path):
        """Склеивает тайлы в GeoTIFF (если задан cache_dir)."""
        if not self.cache_dir:
            print("[Region] Кэш не задан, пропускаем склейку")
            return

        print(f"\n[Region] Склейка в GeoTIFF: {output_path}")

        # Размер карты в пикселях
        tile_size = 256
        px_w = (self.tx_hi - self.tx_lo + 1) * tile_size
        px_h = (self.ty_hi - self.ty_lo + 1) * tile_size

        # Geo-привязка
        top_lat, left_lon = tile_to_latlon(self.tx_lo, self.ty_lo, self.zoom)
        bot_lat, _ = tile_to_latlon(self.tx_lo, self.ty_hi + 1, self.zoom)
        _, right_lon = tile_to_latlon(self.tx_hi + 1, self.ty_lo, self.zoom)
        res_lon = (right_lon - left_lon) / px_w
        res_lat = (top_lat - bot_lat) / px_h

        print(f"[Region] Размер: {px_w}x{px_h} px")
        print(f"[Region] Geo: lat={top_lat:.6f}, lon={left_lon:.6f}, "
              f"res={res_lat:.8f}x{res_lon:.8f} град/пикс")

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=px_h, width=px_w,
            count=3, dtype='uint8',
            crs='EPSG:4326',
            transform=from_origin(left_lon, top_lat, res_lon, res_lat),
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress='deflate',
            bigtiff='YES',
        ) as dst:
            written = 0
            missing = 0
            for ty in range(self.ty_lo, self.ty_hi + 1):
                for tx in range(self.tx_lo, self.tx_hi + 1):
                    path = self.tile_path(tx, ty)
                    if not path or not os.path.exists(path):
                        missing += 1
                        continue
                    try:
                        tile_arr = np.array(Image.open(path).convert('RGB'))
                    except Exception:
                        missing += 1
                        continue

                    px_left = (tx - self.tx_lo) * tile_size
                    px_top = (ty - self.ty_lo) * tile_size
                    window = rasterio.windows.Window(
                        px_left, px_top, tile_size, tile_size)
                    for b in range(3):
                        dst.write(tile_arr[:, :, b], b + 1, window=window)
                    written += 1

                    if written % 2000 == 0:
                        print(f"[Region] Записано {written}/{self.total_tiles}...")

        print(f"[Region] ✓ Сохранено: {output_path}")
        print(f"[Region] Размер файла: {os.path.getsize(output_path) / 1e6:.1f} MB")
        print(f"[Region] Тайлов записано: {written}, пропущено: {missing}")


def main():
    parser = argparse.ArgumentParser(description='Скачивание региональной карты')
    parser.add_argument('--min-lat', type=float, required=True)
    parser.add_argument('--max-lat', type=float, required=True)
    parser.add_argument('--min-lon', type=float, required=True)
    parser.add_argument('--max-lon', type=float, required=True)
    parser.add_argument('--zoom', type=int, default=19)
    parser.add_argument('--cache', type=str, default='.aerial_nav_tiles/region_google_z19')
    parser.add_argument('--output', type=str, default='map_cache/region_google.tif')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--rate-limit', type=float, default=0.25,
                        help='Задержка между запросами в потоке (с)')
    parser.add_argument('--stitch-only', action='store_true',
                        help='Только склейка из кэша')
    args = parser.parse_args()

    downloader = RegionMapDownloader(
        min_lat=args.min_lat,
        max_lat=args.max_lat,
        min_lon=args.min_lon,
        max_lon=args.max_lon,
        zoom=args.zoom,
        cache_dir=args.cache,
        threads=args.threads,
        rate_limit=args.rate_limit,
    )

    if not args.stitch_only:
        downloader.download_all()
    downloader.stitch(args.output)


if __name__ == '__main__':
    main()
