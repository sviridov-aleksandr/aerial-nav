#!/usr/bin/env python3
"""
multi_level_index.py — многоуровневый индекс карты для навигации на разных высотах.

Проблема: при высоте 1000-1200 м кадр камеры видит большую область (scale 2.5-3.0×),
а индекс состоит из тайлов 512×512. Модель не может сопоставить «уменьшённый вид
большой области» с «детальным видом маленького участка».

Решение: несколько уровней индекса, каждый под свой диапазон высот.
  Уровень 0: патч 512×512   (scale=1.0,  h≈400 м)  → resize в 512
  Уровень 1: патч 1024×1024 (scale=2.0,  h≈800 м)  → resize в 512
  Уровень 2: патч 1792×1792 (scale=3.5,  h≈1200 м) → resize в 512

При поиске: высота из барометра → выбор уровня → поиск только в нём.
Каждый уровень: читает патч нужного размера из GeoTIFF, ресайзит в 512×512, индексирует.

Использование:
  from multi_level_index import MultiLevelIndex
  index = MultiLevelIndex(model, src, coords, device)
  index.build()
  result = index.search(frame, altitude=1000)
"""

import os
import sys
import math
import time
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from siamese_network import AerialFeatureExtractor

RESOLUTION = 0.206  # м/px
TILE_SIZE = 512
CAM_W = 3840
CAM_FOV_H = 90.0
ALT_STEP = 100       # шаг высот в метрах
ALT_MIN = 0
ALT_MAX = 1200


def altitude_to_patch_size(altitude_m: float) -> int:
    """Размер патча карты для заданной высоты (px)."""
    if altitude_m <= 0:
        return 128
    footprint_w = 2 * altitude_m * math.tan(math.radians(CAM_FOV_H / 2))
    gsd_cam = footprint_w / CAM_W
    patch_m = gsd_cam * TILE_SIZE
    patch_px = int(round(patch_m / RESOLUTION))
    patch_px = max(128, patch_px)
    patch_px = ((patch_px + 63) // 64) * 64  # кратно 64 — чётный, без артефактов
    return patch_px


def _generate_levels():
    """Генерация 13 уровней индекса (0-1200 м, шаг 100 м)."""
    levels = []
    for alt in range(ALT_MIN, ALT_MAX + ALT_STEP, ALT_STEP):
        patch_size = altitude_to_patch_size(alt)
        alt_min = max(0, alt - ALT_STEP // 2)
        alt_max = alt + ALT_STEP // 2
        levels.append({
            'patch_size': patch_size,
            'altitude': alt,
            'alt_min': alt_min,
            'alt_max': alt_max,
        })
    levels[-1]['alt_max'] = 9999  # последний уровень — до бесконечности
    return levels


LEVELS = _generate_levels()


def altitude_to_level(altitude_m: float) -> int:
    """Выбор уровня индекса по высоте полёта (13 уровней, шаг 100 м)."""
    idx = round(altitude_m / ALT_STEP)
    return max(0, min(idx, len(LEVELS) - 1))


def normalize(arr):
    """Нормализация ImageNet."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    out = arr.astype(np.float32) / 255.0
    out = (out - mean) / std
    return out


class MultiLevelIndex:
    """
    Многоуровневый индекс карты.

    Для каждого уровня:
      1. Читает патч patch_size×patch_size из GeoTIFF (центр = координата тайла)
      2. Ресайзит в 512×512
      3. Вычисляет embedding через Siamese-модель
      4. Хранит эмбеддинги и координаты

    При поиске:
      1. Определяет уровень по высоте
      2. Вычисляет embedding кадра камеры
      3. Ищет ближайший тайл в выбранном уровне
    """

    def __init__(self, model, src, coords, device,
                 index_step=1, batch_size=32):
        """
        Args:
            model: AerialFeatureExtractor (загруженная, eval)
            src: rasterio dataset (открытый GeoTIFF)
            coords: np.array [(x, y), ...] — координаты тайлов (верхний левый угол)
            device: torch.device
            index_step: шаг прореживания индекса (1=все тайлы, 4=каждый 4-й)
            batch_size: размер батча для индексации
        """
        self.model = model
        self.src = src
        self.coords = coords[::index_step]
        self.device = device
        self.batch_size = batch_size

        # Индексы по уровням: embs[level] = tensor (N, 256), locs[level] = np.array (N, 2)
        self.embs = {i: None for i in range(len(LEVELS))}
        self.locs = {i: None for i in range(len(LEVELS))}

    def _read_patch(self, cx, cy, patch_size):
        """Читает патч patch_size×patch_size из карты, ресайзит в 512×512."""
        x1 = int(cx - patch_size / 2)
        y1 = int(cy - patch_size / 2)

        win = Window(x1, y1, patch_size, patch_size)
        try:
            data = self.src.read(window=win)
            arr = np.transpose(data, (1, 2, 0))
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
        except Exception:
            return np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)

        h, w = arr.shape[:2]
        if h < patch_size or w < patch_size:
            padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
            padded[:h, :w] = arr
            arr = padded

        img = Image.fromarray(arr)
        img = img.resize((TILE_SIZE, TILE_SIZE), Image.BILINEAR)
        return np.array(img)

    @torch.no_grad()
    def _embed_batch(self, tiles):
        """Вычисляет эмбеддинги батча тайлов."""
        arr = normalize(np.array(tiles))
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device)
        return self.model(tensor)

    def build(self):
        """Построение индексов всех уровней."""
        total = len(self.coords)
        print(f"[MultiLevel] Индексация {total} тайлов × {len(LEVELS)} уровней...")

        for lvl_idx, lvl in enumerate(LEVELS):
            patch_size = lvl['patch_size']
            print(f"\n[MultiLevel] Уровень {lvl_idx}: патч {patch_size}×{patch_size} "
                  f"(h={lvl['alt_min']}-{lvl['alt_max']} м)")

            all_embs = []
            all_locs = []
            batch_tiles = []
            batch_locs = []
            t0 = time.time()

            for i, (x, y) in enumerate(self.coords):
                cx = x + TILE_SIZE / 2
                cy = y + TILE_SIZE / 2
                tile = self._read_patch(cx, cy, patch_size)
                if tile.max() == 0:
                    continue

                batch_tiles.append(tile)
                batch_locs.append((x, y))

                if len(batch_tiles) >= self.batch_size:
                    embs = self._embed_batch(batch_tiles)
                    all_embs.append(embs)
                    all_locs.extend(batch_locs)
                    batch_tiles = []
                    batch_locs = []

                if (i + 1) % 1000 == 0:
                    print(f"  [{i+1}/{total}]")

            if batch_tiles:
                embs = self._embed_batch(batch_tiles)
                all_embs.append(embs)
                all_locs.extend(batch_locs)

            self.embs[lvl_idx] = torch.cat(all_embs, dim=0)
            self.locs[lvl_idx] = np.array(all_locs)
            elapsed = time.time() - t0
            print(f"  Уровень {lvl_idx}: {self.embs[lvl_idx].shape[0]} тайлов, "
                  f"{elapsed:.0f}s")

        total_embs = sum(self.embs[i].shape[0] for i in range(len(LEVELS)))
        print(f"\n[MultiLevel] Всего эмбеддингов: {total_embs}")

    @torch.no_grad()
    def search(self, frame, altitude_m):
        """
        Поиск позиции кадра в индексе.

        Args:
            frame: кадр камеры (512×512, RGB)
            altitude_m: высота полёта (м) — определяет уровень индекса

        Returns:
            dict: position (x, y), confidence, level, distance
        """
        lvl_idx = altitude_to_level(altitude_m)

        # Embedding кадра
        arr = normalize(np.array([frame]))
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device)
        query_emb = self.model(tensor)[0]

        # Поиск в выбранном уровне
        index_embs = self.embs[lvl_idx]
        d = torch.cdist(query_emb.unsqueeze(0), index_embs).squeeze(0)
        nearest = d.argmin().item()
        best_x, best_y = self.locs[lvl_idx][nearest]
        confidence = 1.0 - d[nearest].item()

        return {
            'position': (best_x, best_y),
            'confidence': confidence,
            'level': lvl_idx,
            'distance': d[nearest].item(),
            'nearest_idx': nearest,
        }

    def search_all_levels(self, frame):
        """
        Поиск по всем уровням (для тестов — какой уровень лучше).

        Returns:
            list of dict для каждого уровня
        """
        results = []
        arr = normalize(np.array([frame]))
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device)
        query_emb = self.model(tensor)[0]

        for lvl_idx in range(len(LEVELS)):
            index_embs = self.embs[lvl_idx]
            d = torch.cdist(query_emb.unsqueeze(0), index_embs).squeeze(0)
            nearest = d.argmin().item()
            best_x, best_y = self.locs[lvl_idx][nearest]
            confidence = 1.0 - d[nearest].item()

            results.append({
                'position': (best_x, best_y),
                'confidence': confidence,
                'level': lvl_idx,
                'distance': d[nearest].item(),
                'nearest_idx': nearest,
            })

        return results

    def get_level_info(self):
        """Информация об уровнях индекса."""
        info = []
        for i, lvl in enumerate(LEVELS):
            info.append({
                'level': i,
                'patch_size': lvl['patch_size'],
                'alt_range': f"{lvl['alt_min']}-{lvl['alt_max']} м",
                'num_tiles': self.embs[i].shape[0] if self.embs[i] is not None else 0,
            })
        return info


def camera_frame_from_map(src, center_px, altitude, tile_size=512):
    """Имитация кадра камеры с высоты altitude м."""
    footprint_w = 2 * altitude * math.tan(math.radians(CAM_FOV_H / 2))
    gsd_cam = footprint_w / CAM_W
    patch_m = gsd_cam * tile_size
    patch_px_map = int(patch_m / RESOLUTION)

    cx, cy = center_px
    x1 = int(cx - patch_px_map / 2)
    y1 = int(cy - patch_px_map / 2)

    win = Window(x1, y1, patch_px_map, patch_px_map)
    try:
        data = src.read(window=win)
        map_crop = np.transpose(data, (1, 2, 0))
        if map_crop.shape[2] == 4:
            map_crop = map_crop[:, :, :3]
    except Exception:
        return np.zeros((tile_size, tile_size, 3), dtype=np.uint8)

    h, w = map_crop.shape[:2]
    if h < patch_px_map or w < patch_px_map:
        padded = np.zeros((patch_px_map, patch_px_map, 3), dtype=np.uint8)
        padded[:h, :w] = map_crop
        map_crop = padded

    img = Image.fromarray(map_crop)
    img = img.resize((tile_size, tile_size), Image.BILINEAR)
    return np.array(img)


if __name__ == '__main__':
    """Тест многоуровневого индекса."""
    import argparse

    parser = argparse.ArgumentParser(description="Тест многоуровневого индекса")
    parser.add_argument('--model', default='region_model.pth')
    parser.add_argument('--index-step', type=int, default=4,
                        help='Шаг прореживания (4=1559 тайлов)')
    parser.add_argument('--num-test', type=int, default=20)
    args = parser.parse_args()

    MAP_PATH = os.path.join(PROJECT_DIR, 'map_cache/antiuav_route_strip.tif')
    COORDS_PATH = os.path.join(PROJECT_DIR, 'training_data/route_dataset/positive_coords.npy')
    MODEL_PATH = os.path.join(PROJECT_DIR, args.model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Test] Device: {device}")

    # Загрузка модели
    model = AerialFeatureExtractor(embedding_dim=256).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"[Test] Модель: epoch={ckpt.get('epoch', '?')}")
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Открытие карты
    src = rasterio.open(MAP_PATH)
    coords = np.load(COORDS_PATH)
    print(f"[Test] Карта: {src.width}x{src.height}, тайлов: {len(coords)}")

    # Построение индекса
    index = MultiLevelIndex(model, src, coords, device, index_step=args.index_step)
    index.build()

    print(f"\n[Test] Уровни индекса:")
    for info in index.get_level_info():
        print(f"  Уровень {info['level']}: патч {info['patch_size']}×{info['patch_size']}, "
              f"{info['alt_range']}, {info['num_tiles']} тайлов")

    # Тест: для каждой высоты — поиск по правильному уровню
    ALTITUDES = [100, 200, 300, 400, 450, 700, 800, 1000, 1200]
    rng = np.random.default_rng(42)
    test_indices = rng.choice(len(coords), size=min(args.num_test, len(coords)), replace=False)

    print(f"\n{'='*70}")
    print("ТЕСТ: Многоуровневый индекс (поиск по высоте)")
    print(f"{'='*70}")

    for alt in ALTITUDES:
        correct = 0
        errs = []
        confs = []
        lvl_used = altitude_to_level(alt)

        for ti in test_indices:
            x, y = coords[ti]
            center_px = (x + TILE_SIZE / 2, y + TILE_SIZE / 2)

            # Кадр камеры с высоты
            frame = camera_frame_from_map(src, center_px, alt, TILE_SIZE)
            if frame.max() == 0:
                continue

            # Поиск по уровню
            result = index.search(frame, alt)
            pred_x, pred_y = result['position']

            # Ошибка
            err_px = math.sqrt((pred_x - x)**2 + (pred_y - y)**2)
            err_m = err_px * RESOLUTION
            errs.append(err_m)
            confs.append(result['confidence'])

            if err_m < 30:
                correct += 1

        n = len(errs)
        recall = correct / n * 100 if n > 0 else 0
        med_err = float(np.median(errs)) if errs else 0
        avg_conf = float(np.mean(confs)) if confs else 0

        print(f"\n  h={alt:4d}м (уровень {lvl_used}):")
        print(f"    Recall (<30м): {correct}/{n} ({recall:.0f}%)")
        print(f"    Медиана err:   {med_err:.0f} м")
        print(f"    Средняя conf:  {avg_conf:.3f}")

    # Дополнительно: сравнение уровней для h=1000 м
    print(f"\n{'='*70}")
    print("СРАВНЕНИЕ УРОВНЕЙ для h=1000 м")
    print(f"{'='*70}")

    for ti in test_indices[:5]:
        x, y = coords[ti]
        center_px = (x + TILE_SIZE / 2, y + TILE_SIZE / 2)
        frame = camera_frame_from_map(src, center_px, 1000, TILE_SIZE)

        results = index.search_all_levels(frame)
        print(f"\n  Тайл ({x}, {y}):")
        for r in results:
            err_px = math.sqrt((r['position'][0] - x)**2 + (r['position'][1] - y)**2)
            err_m = err_px * RESOLUTION
            print(f"    Уровень {r['level']}: dist={r['distance']:.3f}, "
                  f"err={err_m:.0f} м, conf={r['confidence']:.3f}")

    src.close()
    print("\n[Test] Завершено")
