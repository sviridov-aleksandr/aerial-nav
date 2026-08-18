#!/usr/bin/env python3
"""
stitch_map.py — склейка тайлов карты из кэша в GeoTIFF.

Записывает напрямую в TIFF через rasterio — не создаёт промежуточные
массивы в RAM. Каждый тайл пишется отдельным блоком.

Использование:
  python3 stitch_map.py \
    --tiles map_cache/strips \
    --output map_cache/antiuav_route_strip.tif \
    --resolution 0.5
"""

import os
import sys
import re
import argparse
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin


def tile_to_latlon(x, y, zoom):
    """Конвертирует координаты тайла в lat/lon."""
    n = 2.0 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y / n)))
    lat = np.degrees(lat_rad)
    return lat, lon


def parse_tile_name(filename):
    """Извлекает zoom, x, y из имени файла типа z19_309252_185788.png"""
    m = re.match(r'z(\d+)_(\d+)_(\d+)\.png', filename)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description='Склейка тайлов карты из кэша')
    parser.add_argument('--tiles', '-t', required=True,
                        help='Директория с тайлами (рекурсивно)')
    parser.add_argument('--output', '-o', required=True,
                        help='Выходной TIFF')
    parser.add_argument('--resolution', '-r', type=float, default=0.5,
                        help='Разрешение м/пиксель (default: 0.5)')
    args = parser.parse_args()

    print(f"[Stitch] Склейка тайлов из: {args.tiles}")

    # Собираем все тайлы
    tiles = []
    for root, dirs, files in os.walk(args.tiles):
        for fname in files:
            if fname.endswith('.png'):
                zoom, x, y = parse_tile_name(fname)
                if zoom is not None:
                    lat, lon = tile_to_latlon(x, y, zoom)
                    tiles.append({
                        'path': os.path.join(root, fname),
                        'zoom': zoom,
                        'x': x,
                        'y': y,
                        'lat': lat,
                        'lon': lon,
                    })

    print(f"[Stitch] Найдено тайлов: {len(tiles)}")
    if not tiles:
        print("[Stitch] ✗ Нет тайлов!")
        return

    # Определяем bounding box по индексам тайлов
    min_tx = min(t['x'] for t in tiles)
    max_tx = max(t['x'] for t in tiles)
    min_ty = min(t['y'] for t in tiles)
    max_ty = max(t['y'] for t in tiles)

    tile_size = 256  # размер тайла в пикселях
    px_w = (max_tx - min_tx + 1) * tile_size
    px_h = (max_ty - min_ty + 1) * tile_size

    # Geo-привязка: lat/lon верхнего левого угла минимального тайла
    top_lat, left_lon = tile_to_latlon(min_tx, min_ty, tiles[0]['zoom'])
    # Разрешение в градусах на пиксель
    bot_lat, _ = tile_to_latlon(min_tx, max_ty + 1, tiles[0]['zoom'])
    _, right_lon = tile_to_latlon(max_tx + 1, min_ty, tiles[0]['zoom'])
    res_lon = (right_lon - left_lon) / px_w
    res_lat = (top_lat - bot_lat) / px_h

    print(f"[Stitch] Сетка тайлов: x [{min_tx}-{max_tx}] ({max_tx-min_tx+1} кол.), "
          f"y [{min_ty}-{max_ty}] ({max_ty-min_ty+1} строк)")
    print(f"[Stitch] Размер: {px_w}x{px_h} px")
    print(f"[Stitch] Geo: lat={top_lat:.6f}, lon={left_lon:.6f}, "
          f"res={res_lat:.8f}x{res_lon:.8f} град/пикс")

    # Создаём TIFF через rasterio
    print(f"[Stitch] Создаём TIFF: {args.output}")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    with rasterio.open(
        args.output, 'w',
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
        # Записываем тайлы по прямому мапированию индексов

        for i, t in enumerate(tiles):
            if (i + 1) % 500 == 0 or i == 0:
                print(f"[Stitch] Тайл {i+1}/{len(tiles)}: "
                      f"({t['x']},{t['y']})")

            tile_img = Image.open(t['path']).convert('RGB')
            tile_arr = np.array(tile_img)
            tile_h, tile_w = tile_arr.shape[:2]

            # Прямое мапирование: каждый тайл = 256x256 пикселей
            px_left = (t['x'] - min_tx) * tile_size
            px_top = (t['y'] - min_ty) * tile_size
            px_right = px_left + tile_w
            px_bottom = px_top + tile_h

            # Проверяем границы
            if px_left >= px_w or px_top >= px_h:
                continue
            if px_right <= 0 or px_bottom <= 0:
                continue

            # Обрезаем по границам
            tx_start = 0
            ty_start = 0
            if px_left < 0:
                tx_start = -px_left
                px_left = 0
            if px_top < 0:
                ty_start = -px_top
                px_top = 0
            px_right = min(px_right, px_w)
            px_bottom = min(px_bottom, px_h)

            ty_end = ty_start + (px_bottom - px_top)
            tx_end = tx_start + (px_right - px_left)

            if ty_end > ty_start and tx_end > tx_start:
                window = rasterio.windows.Window(px_left, px_top,
                                                 px_right - px_left,
                                                 px_bottom - px_top)
                for b in range(3):
                    dst.write(tile_arr[ty_start:ty_end, tx_start:tx_end, b],
                              b + 1, window=window)

    print(f"[Stitch] ✓ Сохранено: {args.output}")
    print(f"[Stitch] Размер файла: {os.path.getsize(args.output) / 1e6:.1f} MB")


if __name__ == '__main__':
    main()