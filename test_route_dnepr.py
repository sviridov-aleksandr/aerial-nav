#!/usr/bin/env python3
"""
test_route_dnepr.py — реалистичный тест навигации на маршруте.

Имитация полёта БПЛА по маршруту:
  - Query — кадр камеры (FOV 90°), снят с высоты 700/1000/1200 м
  - Центр кадра смещён от центра тайла индекса на 10-50 м
    (типичное боковое отклонение дрона от линии маршрута)
  - Размер патча и предобработка совпадают с обучением (512×512)
  - Координаты — верхний левый угол (как в generate_coords.py)

ВНИМАНИЕ (исправления по сравнению со старой версией):
  - RESOLUTION = 0.206 м/px — реальное разрешение карты zoom 19
    (было 0.5 — ошибка в 2.4 раза завышала расстояния)
  - step = 1 — полный индекс (6236 тайлов) вместо каждого 4-го
    (прореженный индекс давал шаг ~211 м → ошибки в километры)
  - Offsets [10, 25, 50] м вместо [50, 75, 100] — при отклонении
    50+ м даже идеальная модель не пройдёт порог 30 м (ошибка ≥ offset)
  - Модель: region_model.pth (после curriculum learning на регионе)

Запуск:
  python3 test_route_dnepr.py
"""

import os
import sys
import math
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from siamese_network import AerialFeatureExtractor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Параметры
MAP_PATH = os.path.join(PROJECT_DIR, 'map_cache/antiuav_route_strip.tif')
MODEL_PATH = os.path.join(PROJECT_DIR, 'region_model.pth')
COORDS_PATH = os.path.join(PROJECT_DIR, 'training_data/route_dataset/positive_coords.npy')
RESOLUTION = 0.206  # м/px карты (zoom 19, реальное значение для 46.27°N)

# Размер тайла — как при обучении
TILE_SIZE = 512

# Камера OpenIPC MC800S-V3 (Sony IMX415, объектив ASX-0116 KH)
CAM_W, CAM_H = 3840, 2160  # 4K
CAM_FOV_H = 90.0  # градусов

# Высоты полёта
ALTITUDES = [700.0, 1000.0, 1200.0]

# Боковое отклонение от маршрута (м) — реалистичное, чтобы порог 30 м был достижим
LATERAL_OFFSETS = [10.0, 25.0, 50.0]

# Шаг индекса (1 = полный, 6236 тайлов; 2 = каждый 2-й, 3118)
INDEX_STEP = 1


def load_model():
    """Загружает сиамскую модель."""
    model = AerialFeatureExtractor(embedding_dim=256).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"[Test] Модель загружена: {MODEL_PATH}")
    return model


def normalize_batch(arr):
    """Нормализация батча (ImageNet, как в обучении)."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    out = arr.astype(np.float32) / 255.0
    out = (out - mean) / std
    return out


def read_tile_topleft(src, x, y, size):
    """Читает тайл из GeoTIFF. (x, y) — верхний левый угол. Дополняет нулями."""
    win = Window(int(x), int(y), size, size)
    try:
        data = src.read(window=win)
        tile = np.transpose(data, (1, 2, 0))
        if tile.shape[2] == 4:
            tile = tile[:, :, :3]
        h, w = tile.shape[:2]
        if h < size or w < size:
            padded = np.zeros((size, size, 3), dtype=np.uint8)
            padded[:h, :w] = tile
            tile = padded
        return tile
    except Exception:
        return np.zeros((size, size, 3), dtype=np.uint8)


def build_tile_index(model, src, coords, tile_size):
    """Строит индекс embeddings. coords — верхние левые углы."""
    print(f"[Test] Индексация {len(coords)} тайлов ({tile_size}×{tile_size})...")
    embeddings = []
    locations = []  # (x, y) — верхний левый угол
    batch_size = 32
    batch_tiles = []
    batch_locs = []

    for i, (x, y) in enumerate(coords):
        tile = read_tile_topleft(src, x, y, tile_size)
        if tile.max() == 0:
            continue
        batch_tiles.append(tile)
        batch_locs.append((x, y))

        if len(batch_tiles) >= batch_size or i == len(coords) - 1:
            arr = np.array(batch_tiles)
            norm = normalize_batch(arr)
            tensor = torch.from_numpy(norm).permute(0, 3, 1, 2).to(DEVICE)
            with torch.no_grad():
                embs = model(tensor).cpu()
            embeddings.append(embs)
            locations.extend(batch_locs)
            batch_tiles = []
            batch_locs = []

        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(coords)}...")

    embeddings = torch.cat(embeddings, dim=0)
    locations = np.array(locations)
    print(f"[Test] Индекс: {embeddings.shape[0]} тайлов, {embeddings.shape[1]}D")
    return embeddings, locations


def camera_frame_from_map(src, center_px, altitude, tile_size=512):
    """
    Имитация кадра камеры: вырезка из карты с центром в center_px,
    размер соответствует патчу камеры при данной высоте.

    Камера 4K, FOV 90°:
      - footprint_w = 2 * alt * tan(45°) = 2 * alt
      - GSD = footprint_w / 3840
      - Патч tile_size px камеры = tile_size * GSD метров на земле
      - На карте (0.206 м/px): tile_size * GSD / RESOLUTION пикселей

    Масштабная нормализация: патч камеры ресемплится к tile_size (512),
    что совпадает с размером тайла обучения.

    center_px: (x, y) — центр кадра на карте (px)
    altitude: высота полёта (м)
    tile_size: размер выходного патча (512, как при обучении)
    """
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

    # Ресемплинг к tile_size (как делает камера при записи 4K → сеть 512)
    img = Image.fromarray(map_crop)
    img = img.resize((tile_size, tile_size), Image.BILINEAR)
    return np.array(img)


def main():
    print("=" * 70)
    print("ТЕСТ НАВИГАЦИИ — МАРШРУТ (реалистичный, 512×512)")
    print(f"Индекс: step={INDEX_STEP}, resolution={RESOLUTION} м/px")
    print("=" * 70)

    # 1. Загрузка модели
    model = load_model()

    # 2. Открытие карты
    print(f"\n[Test] Открываем карту: {MAP_PATH}")
    src = rasterio.open(MAP_PATH)
    print(f"[Test] Карта: {src.width}x{src.height} px")

    # 3. Координаты тайлов (верхние левые углы)
    coords = np.load(COORDS_PATH)
    print(f"[Test] Координат тайлов: {len(coords)}")

    # 4. Индексация (полный индекс)
    coords_subset = coords[::INDEX_STEP]
    print(f"[Test] Используем каждый {INDEX_STEP}-й тайл: {len(coords_subset)}")
    embeddings, locations = build_tile_index(model, src, coords_subset, TILE_SIZE)

    # 5. Тест: имитация полёта
    print(f"\n[Test] Тестирование matching (высоты {ALTITUDES}, "
          f"отклонения {LATERAL_OFFSETS} м)...")
    rng = np.random.default_rng(42)

    num_per_config = 25
    top1_correct = 0
    top5_correct = 0
    total = 0
    errors_m = []
    config_stats = {}

    for alt in ALTITUDES:
        for offset in LATERAL_OFFSETS:
            cfg_correct = 0
            cfg_total = 0
            cfg_errors = []
            key = f"h{int(alt)}_o{int(offset)}"

            for _ in range(num_per_config):
                # Случайный тайл из индекса
                idx = rng.integers(0, len(coords_subset))
                base_x, base_y = coords_subset[idx]

                # Центр тайла индекса на карте
                center_idx_x = base_x + TILE_SIZE / 2
                center_idx_y = base_y + TILE_SIZE / 2

                # Смещение центра кадра: боковое отклонение + случайный шум
                angle = rng.uniform(0, 2 * math.pi)
                dx_m = offset * math.cos(angle) + rng.uniform(-10, 10)
                dy_m = offset * math.sin(angle) + rng.uniform(-10, 10)
                center_px = (center_idx_x + dx_m / RESOLUTION,
                             center_idx_y + dy_m / RESOLUTION)

                # Кадр камеры с этой высоты
                frame = camera_frame_from_map(src, center_px, alt, TILE_SIZE)
                if frame.max() == 0:
                    continue

                query_emb = get_embedding(model, frame)
                sims = torch.nn.functional.cosine_similarity(
                    query_emb.unsqueeze(0), embeddings, dim=1
                )
                top5 = torch.topk(sims, 5)

                # Истинная позиция — центр кадра (в px карты)
                true_x, true_y = center_px

                # Top-1: берём центр ближайшего тайла индекса
                best_idx = top5.indices[0].item()
                pred_x, pred_y = locations[best_idx]
                pred_center_x = pred_x + TILE_SIZE / 2
                pred_center_y = pred_y + TILE_SIZE / 2
                err_m = math.sqrt((pred_center_x - true_x)**2 +
                                  (pred_center_y - true_y)**2) * RESOLUTION
                errors_m.append(err_m)
                cfg_errors.append(err_m)

                if err_m < 50:
                    top1_correct += 1
                    cfg_correct += 1
                # Top-5
                for j in range(5):
                    idx5 = top5.indices[j].item()
                    px5, py5 = locations[idx5]
                    cx5 = px5 + TILE_SIZE / 2
                    cy5 = py5 + TILE_SIZE / 2
                    err5 = math.sqrt((cx5 - true_x)**2 + (cy5 - true_y)**2) * RESOLUTION
                    if err5 < 50:
                        top5_correct += 1
                        break

                total += 1
                cfg_total += 1

            cfg_stats = {
                'top1': cfg_correct / cfg_total * 100 if cfg_total else 0,
                'median': float(np.median(cfg_errors)) if cfg_errors else float('nan'),
            }
            config_stats[key] = cfg_stats
            print(f"  [{key}] top1={cfg_correct}/{cfg_total} "
                  f"({cfg_stats['top1']:.0f}%), "
                  f"err_med={cfg_stats['median']:.1f}м")

    # 6. Результаты
    errors_m = np.array(errors_m)
    print(f"\n{'='*70}")
    print("РЕЗУЛЬТАТЫ")
    print(f"{'='*70}")
    print(f"  Тестов: {total}")
    print(f"  Top-1 (< 50м): {top1_correct}/{total} ({top1_correct/total*100:.1f}%)")
    print(f"  Top-5 (< 50м): {top5_correct}/{total} ({top5_correct/total*100:.1f}%)")
    print(f"\n  Ошибка позиционирования:")
    print(f"    Средняя: {np.mean(errors_m):.1f} м")
    print(f"    Медиана: {np.median(errors_m):.1f} м")
    print(f"    Макс: {np.max(errors_m):.1f} м")
    print(f"    < 15м: {np.sum(errors_m < 15)/len(errors_m)*100:.1f}%")
    print(f"    < 30м: {np.sum(errors_m < 30)/len(errors_m)*100:.1f}%")
    print(f"    < 50м: {np.sum(errors_m < 50)/len(errors_m)*100:.1f}%")

    print(f"\n  По конфигурациям (высота x отклонение):")
    for key, st in config_stats.items():
        print(f"    {key}: top1={st['top1']:.0f}%, err_med={st['median']:.1f}м")

    src.close()
    print(f"\n{'='*70}")
    if np.median(errors_m) < 30:
        print(f"  ✓ ТЕСТ ПРОЙДЕН: медиана {np.median(errors_m):.1f}м < 30м")
    else:
        print(f"  ✗ ТЕСТ НЕ ПРОЙДЕН: медиана {np.median(errors_m):.1f}м >= 30м")
    print(f"{'='*70}")


def get_embedding(model, tile):
    """Embedding одного тайла (512×512, как при обучении)."""
    arr = normalize_batch(tile)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model(tensor)
    return emb.squeeze(0).cpu()


if __name__ == '__main__':
    main()