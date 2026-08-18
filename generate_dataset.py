"""
Dataset Generator — генерация датасета для обучения сиамской сети.

Использует rasterio для постраничного чтения большой карты без загрузки в память.

Логика:
  1. Открываем карту через rasterio (без загрузки в RAM)
  2. Разбиваем карту на тайлы 512×512 с перекрытием 50%
  3. Для каждого тайла генерируем "кадр с камеры" (аугментации)
  4. Сохраняем пары в формат numpy

Пример использования:
  python3 generate_dataset.py \\
    --map /home/alex/aerial-nav/map_cache/antiuav_route_strip.png \\
    --output /home/alex/aerial-nav/training_data/route_dataset \\
    --tile-size 512 \\
    --stride 256 \\
    --neg-multiplier 3
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import rasterio
from rasterio.windows import Window
import time
import random
from io import BytesIO

# Убираем лимит PIL на размер изображений
Image.MAX_IMAGE_PIXELS = None

# Добавляем проект в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class DatasetGenerator:
    """Генерация датасета для сиамской сети."""

    def __init__(self, map_path: str,
                 output_dir: str, tile_size: int = 512, stride: int = 256,
                 neg_multiplier: int = 3):
        self.map_path = map_path
        self.output_dir = output_dir
        self.tile_size = tile_size
        self.stride = stride
        self.neg_multiplier = neg_multiplier

        # Открываем карту через rasterio (без загрузки в RAM)
        print(f"[DatasetGen] Открываем карту: {map_path}")
        self.src = rasterio.open(map_path)
        self.width = self.src.width
        self.height = self.src.height
        print(f"[DatasetGen] Карта: {self.width}x{self.height} px")
        print(f"[DatasetGen] Каналов: {self.src.count}")

        # Создаём директорию вывода
        os.makedirs(output_dir, exist_ok=True)

    def augment_camera_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Применяет аугментации к кадру камеры.
        """
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

    def generate_dataset(self):
        """Генерирует датасет пар."""
        print(f"\n{'='*70}")
        print("ГЕНЕРАЦИЯ ДАТАСЕТА")
        print(f"{'='*70}")

        print(f"[DatasetGen] Размер карты: {self.width}x{self.height} px")
        print(f"[DatasetGen] Размер тайла: {self.tile_size}x{self.tile_size}")
        print(f"[DatasetGen] Шаг: {self.stride}")

        # Генерируем positive пары (скользим по карте)
        positive_pairs = []
        y = 0
        while y + self.tile_size <= self.height:
            x = 0
            while x + self.tile_size <= self.width:
                # Вырезаем тайл через rasterio (без загрузки всей карты)
                window = Window(x, y, self.tile_size, self.tile_size)
                tile = self.src.read(window=window)

                # tile shape: (C, H, W) -> (H, W, C)
                if tile.shape[0] == 3:
                    tile_array = np.transpose(tile, (1, 2, 0))  # RGB
                else:
                    # Если больше 3 каналов, берём первые 3
                    tile_array = np.transpose(tile[:3], (1, 2, 0))

                # Генерируем аугментированный "кадр камеры"
                camera_frame = self.augment_camera_frame(tile_array)
                camera_crop = Image.fromarray(camera_frame)

                # ID тайла
                tile_id = (x, y, x + self.tile_size, y + self.tile_size)

                positive_pairs.append((tile_id, camera_crop, 1.0))

                x += self.stride
            y += self.stride

        print(f"[DatasetGen] Positive пары: {len(positive_pairs)}")

        # Генерируем negative пары через FAISS
        print(f"[DatasetGen] Генерация negative пар через FAISS...")
        negative_pairs = []

        # Индексы для negative
        num_neg = len(positive_pairs) * self.neg_multiplier
        print(f"[DatasetGen] Нужно negative пар: {num_neg}")

        # Загружаем метаданные индекса
        meta_path = self.map_path.replace('.png', '_meta.npy').replace('antiuav_route_strip', 'route_map_index')
        if os.path.exists(meta_path):
            meta = np.load(meta_path, allow_pickle=True).item()
            tile_paths = meta['tile_paths'].tolist()
        else:
            # Если метаданные не найдены, используем случайные тайлы из карты
            print(f"[DatasetGen] ⚠ Метаданные не найдены, используем случайные тайлы")
            tile_paths = []

        # Генерируем negative пары
        for i, pos_pair in enumerate(positive_pairs):
            tile_id, camera_crop, label = pos_pair

            # Берём neg_multiplier негативов для каждого positive
            for j in range(self.neg_multiplier):
                # Случайное смещение для negative
                neg_x = random.randint(0, max(0, self.width - self.tile_size))
                neg_y = random.randint(0, max(0, self.height - self.tile_size))

                # Вырезаем negative тайл
                window = Window(neg_x, neg_y, self.tile_size, self.tile_size)
                neg_tile = self.src.read(window=window)

                if neg_tile.shape[0] == 3:
                    neg_array = np.transpose(neg_tile, (1, 2, 0))
                else:
                    neg_array = np.transpose(neg_tile[:3], (1, 2, 0))

                neg_crop = Image.fromarray(neg_array)
                negative_pairs.append((neg_x, neg_y, camera_crop, 0.0))

        print(f"[DatasetGen] Negative пары: {len(negative_pairs)}")

        # Объединяем
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)

        print(f"[DatasetGen] Всего пар: {len(all_pairs)}")

        # Сохраняем датасет
        print(f"\n[DatasetGen] Сохранение датасета: {self.output_dir}")

        # Сохраняем как numpy файлы
        map_dir = os.path.join(self.output_dir, 'map_tiles')
        camera_dir = os.path.join(self.output_dir, 'camera_tiles')
        os.makedirs(map_dir, exist_ok=True)
        os.makedirs(camera_dir, exist_ok=True)

        labels = []
        tile_ids = []

        for i, pair in enumerate(all_pairs):
            if i % 1000 == 0:
                print(f"[DatasetGen] Сохранено: {i}/{len(all_pairs)}")

            if len(pair) == 3:
                # Positive: (tile_id, camera_crop, label)
                tile_id, camera_crop, label = pair
                if isinstance(tile_id, tuple) and len(tile_id) == 4:
                    x, y, x2, y2 = tile_id
                    window = Window(x, y, self.tile_size, self.tile_size)
                    map_tile = self.src.read(window=window)
                    if map_tile.shape[0] == 3:
                        map_arr = np.transpose(map_tile, (1, 2, 0))
                    else:
                        map_arr = np.transpose(map_tile[:3], (1, 2, 0))
                else:
                    map_arr = np.array(Image.open(tile_id).convert('RGB'))
                camera_arr = np.array(camera_crop)
            else:
                # Negative: (neg_x, neg_y, camera_crop, label)
                neg_x, neg_y, camera_crop, label = pair
                window = rasterio.windows.from_origin(neg_x, neg_y, self.tile_size, self.tile_size)
                neg_tile = self.src.read(window=window)
                if neg_tile.shape[0] == 3:
                    map_arr = np.transpose(neg_tile, (1, 2, 0))
                else:
                    map_arr = np.transpose(neg_tile[:3], (1, 2, 0))
                camera_arr = np.array(camera_crop)

            labels.append(label)
            tile_ids.append(f"{i:06d}")

            # Сохраняем как numpy
            np.save(os.path.join(map_dir, f'{i:06d}.npy'), map_arr)
            np.save(os.path.join(camera_dir, f'{i:06d}.npy'), camera_arr)

        # Сохраняем метаданные
        np.save(os.path.join(self.output_dir, 'labels.npy'), np.array(labels))
        np.save(os.path.join(self.output_dir, 'tile_ids.npy'), np.array(tile_ids))

        print(f"\n{'='*70}")
        print("ДАТАСЕТ СОХРАНЁН")
        print(f"{'='*70}")
        print(f"  Позиции: {map_dir}")
        print(f"  Камера: {camera_dir}")
        print(f"  Метки: {os.path.join(self.output_dir, 'labels.npy')}")
        print(f"  Всего: {len(all_pairs)} пар")
        print(f"  Positive: {len(positive_pairs)}")
        print(f"  Negative: {len(negative_pairs)}")
        print(f"{'='*70}")

        # Закрываем rasterio
        self.src.close()

        return all_pairs


def main():
    parser = argparse.ArgumentParser(description='Генерация датасета для сиамской сети')
    parser.add_argument('--map', type=str, required=True,
                        help='Путь к карте маршрута (PNG)')
    parser.add_argument('--output', type=str, required=True,
                        help='Директория сохранения датасета')
    parser.add_argument('--tile-size', type=int, default=512,
                        help='Размер тайла (default: 512)')
    parser.add_argument('--stride', type=int, default=256,
                        help='Шаг скольжения (default: 256, 50 pct overlap)')
    parser.add_argument('--neg-multiplier', type=int, default=3,
                        help='Количество negative на 1 positive (default: 3)')

    args = parser.parse_args()

    generator = DatasetGenerator(
        map_path=args.map,
        output_dir=args.output,
        tile_size=args.tile_size,
        stride=args.stride,
        neg_multiplier=args.neg_multiplier
    )

    generator.generate_dataset()


if __name__ == '__main__':
    main()

