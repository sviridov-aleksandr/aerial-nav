#!/usr/bin/env python3
"""
convert_map_to_tiff.py — конвертация PNG-ленты маршрута в GeoTIFF.

Почему не gdal_translate:
  - gdal_translate на гигантских PNG (157K×44K) иногда пишет пустые тайлы
    (ошибка TIFFFillTile: got 0 bytes, expected 786432).
  - Здесь запись идёт по тайлам 512×512 через rasterio — надёжнее.

Использование:
  python3 convert_map_to_tiff.py \
    --input map_cache/antiuav_route_strip.png \
    --output map_cache/antiuav_route_strip.tif \
    --block 512 \
    --compress lzw
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageFile

# Разрешаем гигантские изображения
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

import rasterio
from rasterio.transform import from_origin


def main():
    parser = argparse.ArgumentParser(description='Конвертация PNG в GeoTIFF по тайлам')
    parser.add_argument('--input', '-i', required=True, help='Входной PNG')
    parser.add_argument('--output', '-o', required=True, help='Выходной TIFF')
    parser.add_argument('--block', type=int, default=512, help='Размер тайла (default: 512)')
    parser.add_argument('--compress', choices=['lzw', 'deflate', 'none'], default='lzw',
                        help='Сжатие (default: lzw)')
    args = parser.parse_args()

    print(f"[Convert] Открываем PNG: {args.input}")
    img = Image.open(args.input)
    w, h = img.size
    print(f"[Convert] PNG: {w}x{h}, mode={img.mode}")

    if img.mode != 'RGB':
        print(f"[Convert] Конвертация {img.mode} -> RGB")
        img = img.convert('RGB')

    # Читаем в массив (для 157K×44K это ~20 ГБ в RAM — опасно!)
    # Поэтому читаем по строкам тайлов, не загружая всё сразу.
    block = args.block

    compress_map = {'lzw': 'lzw', 'deflate': 'deflate', 'none': None}
    compress = compress_map[args.compress]

    print(f"[Convert] Создаём TIFF: {args.output} (block={block}, compress={compress})")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    with rasterio.open(
        args.output, 'w',
        driver='GTiff',
        height=h, width=w,
        count=3, dtype='uint8',
        crs=None,
        transform=from_origin(0, 0, 1, 1),
        tiled=True,
        blockxsize=block,
        blockysize=block,
        compress=compress,
        predictor=2 if compress else None,
        bigtiff='YES',
    ) as dst:
        # Читаем PNG по полосам (strip) — не загружаем весь файл в RAM
        strip_h = block  # читаем по 512 строк за раз
        total_strips = (h + strip_h - 1) // strip_h

        for si in range(total_strips):
            y0 = si * strip_h
            y1 = min(y0 + strip_h, h)

            # Читаем полосу из PNG
            strip = img.crop((0, y0, w, y1))
            arr = np.array(strip)  # (strip_h, w, 3)

            # Записываем по тайлам 512×512
            for x in range(0, w, block):
                x1 = min(x + block, w)
                tile = arr[:, x:x1, :]
                window = rasterio.windows.Window(x, y0, tile.shape[1], tile.shape[0])
                for b in range(3):
                    dst.write(b + 1, window=window, data=tile[:, :, b])

            if (si + 1) % 10 == 0 or si == total_strips - 1:
                print(f"[Convert] Полоса {si+1}/{total_strips} (y={y1}/{h})")

    print(f"[Convert] Готово: {args.output}")
    print(f"[Convert] Размер: {os.path.getsize(args.output) / 1e6:.1f} MB")


if __name__ == '__main__':
    main()