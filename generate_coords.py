"""
Generate Coords — генерация координат тайлов для он-лет датасета.

Создаёт файл positive_coords.npy с координатами всех тайлов карты.
Во время обучения тайлы будут читаться из TIFF по мере необходимости.

Использование:
  python3 generate_coords.py \
    --map /home/alex/aerial-nav/map_cache/antiuav_route_strip.tif \
    --output /home/alex/aerial-nav/training_data/route_dataset \
    --tile-size 512 \
    --stride 256
"""

import os
import sys
import argparse
import numpy as np
import rasterio


def main():
    parser = argparse.ArgumentParser(description='Генерация координат тайлов')
    parser.add_argument('--map', type=str, required=True,
                        help='Путь к карте (GeoTIFF)')
    parser.add_argument('--output', type=str, required=True,
                        help='Директория сохранения')
    parser.add_argument('--tile-size', type=int, default=512,
                        help='Размер тайла (default: 512)')
    parser.add_argument('--stride', type=int, default=256,
                        help='Шаг скольжения (default: 256)')
    parser.add_argument('--min-std', type=float, default=10.0,
                        help='Минимальный std тайла для включения (отброс однородных, default: 10.0)')

    args = parser.parse_args()

    # Открываем карту
    print(f"[Coords] Открываем карту: {args.map}")
    src = rasterio.open(args.map)
    width, height = src.width, src.height
    print(f"[Coords] Карта: {width}x{height} px")
    print(f"[Coords] Фильтр: min_std={args.min_std}")

    # Генерируем координаты тайлов, пропуская пустые и однородные
    coords = []
    skipped_empty = 0
    skipped_flat = 0
    y = 0
    while y + args.tile_size <= height:
        x = 0
        while x + args.tile_size <= width:
            window = rasterio.windows.Window(x, y, args.tile_size, args.tile_size)
            data = src.read(window=window)
            if data.max() == 0:
                skipped_empty += 1
            else:
                # Проверяем std только непустых пикселей (отбрасываем чёрный фон)
                mask = data > 0
                if mask.sum() < data.size * 0.3:
                    # Меньше 30% непустых — граница карты, пропускаем
                    skipped_flat += 1
                else:
                    non_zero = data[mask]
                    if non_zero.std() < args.min_std:
                        skipped_flat += 1
                    else:
                        coords.append((x, y))
            x += args.stride
        y += args.stride

    print(f"[Coords] Positive тайлов: {len(coords)}, "
          f"пропущено пустых: {skipped_empty}, однородных: {skipped_flat}")

    # Сохраняем координаты
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, 'positive_coords.npy')
    np.save(output_path, np.array(coords))
    print(f"[Coords] Сохранено: {output_path}")
    print(f"[Coords] Размер файла: {os.path.getsize(output_path) / 1024:.1f} KB")

    src.close()


if __name__ == '__main__':
    main()
