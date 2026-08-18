#!/usr/bin/env python3
"""
test_route_dnepr.py — тест навигации на маршруте в долине Днепра.

Использует antiuav_route_strip.tif (GeoTIFF) и добученную сиамскую модель.
Читает тайлы через rasterio (не загружает карту в RAM целиком).

Запуск:
  /home/alex/my_project_env/bin/python test_route_dnepr.py
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
MODEL_PATH = os.path.join(PROJECT_DIR, 'siamese_model_kalanchak_v2.pth')
COORDS_PATH = os.path.join(PROJECT_DIR, 'training_data/route_dataset/positive_coords.npy')
TILE_SIZE = 224
RESOLUTION = 0.5  # м/px

# Маршрут в долине Днепра (6 точек)
ROUTE_GPS = [
    (46.850, 31.950),  # 0 — старт
    (46.750, 32.000),  # 1
    (46.650, 32.100),  # 2
    (46.550, 32.200),  # 3
    (46.450, 32.300),  # 4
    (46.340, 32.360),  # 5 — финиш
]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


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


def normalize(tile):
    """Нормализация ImageNet."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = tile.astype(np.float32) / 255.0
    arr = (arr - mean) / std
    return arr


def get_embedding(model, tile):
    """Embedding одного тайла."""
    arr = normalize(tile)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model(tensor)
    return emb.squeeze(0).cpu()


def read_tile(src, cx, cy, size):
    """Читает тайл из GeoTIFF по центру. Дополняет нулями до size×size."""
    x1 = int(cx - size // 2)
    y1 = int(cy - size // 2)
    win = Window(x1, y1, size, size)
    try:
        data = src.read(window=win)
        tile = np.transpose(data, (1, 2, 0))
        # Дополняем если окно выходит за границы
        h, w = tile.shape[:2]
        if h < size or w < size:
            padded = np.zeros((size, size, 3), dtype=np.uint8)
            padded[:h, :w] = tile
            tile = padded
        return tile
    except Exception:
        return np.zeros((size, size, 3), dtype=np.uint8)


def build_tile_index(model, src, coords, tile_size):
    """Строит индекс embeddings для всех тайлов с данными."""
    print(f"[Test] Индексация {len(coords)} тайлов...")
    embeddings = []
    locations = []
    batch_size = 64
    batch_tiles = []
    batch_locs = []

    for i, (cx, cy) in enumerate(coords):
        tile = read_tile(src, cx, cy, tile_size)
        if tile.max() == 0:
            continue
        batch_tiles.append(tile)
        batch_locs.append((cx, cy))

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


def normalize_batch(arr):
    """Нормализация батча."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    out = arr.astype(np.float32) / 255.0
    out = (out - mean) / std
    return out


def main():
    print("=" * 70)
    print("ТЕСТ НАВИГАЦИИ — МАРШРУТ ДОЛИНЫ ДНЕПРА")
    print("=" * 70)

    # 1. Загрузка модели
    model = load_model()

    # 2. Открытие карты
    print(f"\n[Test] Открываем карту: {MAP_PATH}")
    src = rasterio.open(MAP_PATH)
    print(f"[Test] Карта: {src.width}x{src.height} px")

    # 3. Координаты тайлов
    coords = np.load(COORDS_PATH)
    print(f"[Test] Координат тайлов: {len(coords)}")

    # 4. Индексация (подмножество для скорости — каждый 4-й тайл)
    step = 4
    coords_subset = coords[::step]
    print(f"[Test] Используем каждый {step}-й тайл: {len(coords_subset)}")
    embeddings, locations = build_tile_index(model, src, coords_subset, TILE_SIZE)

    # 5. Тест: берём случайные тайлы как query и ищем ближайший
    print(f"\n[Test] Тестирование matching...")
    num_tests = min(100, len(coords_subset))
    test_indices = np.random.choice(len(coords_subset), num_tests, replace=False)

    top1_correct = 0
    top5_correct = 0
    errors_px = []

    for i, idx in enumerate(test_indices):
        cx, cy = coords_subset[idx]
        query_tile = read_tile(src, cx, cy, TILE_SIZE)
        if query_tile.max() == 0:
            continue

        query_emb = get_embedding(model, query_tile)
        # Cosine similarity со всеми тайлами
        sims = torch.nn.functional.cosine_similarity(
            query_emb.unsqueeze(0), embeddings, dim=1
        )
        top5 = torch.topk(sims, 5)

        # Истинная позиция
        true_x, true_y = cx, cy

        # Top-1
        best_idx = top5.indices[0].item()
        pred_x, pred_y = locations[best_idx]
        err_px = math.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
        err_m = err_px * RESOLUTION
        errors_px.append(err_m)

        if err_m < 50:
            top1_correct += 1
        # Top-5
        for j in range(5):
            idx5 = top5.indices[j].item()
            px5, py5 = locations[idx5]
            err5 = math.sqrt((px5 - true_x)**2 + (py5 - true_y)**2) * RESOLUTION
            if err5 < 50:
                top5_correct += 1
                break

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{num_tests}: top1={top1_correct}/{i+1}, "
                  f"err_med={np.median(errors_px):.1f}м")

    # 6. Результаты
    errors_px = np.array(errors_px)
    print(f"\n{'='*70}")
    print("РЕЗУЛЬТАТЫ")
    print(f"{'='*70}")
    print(f"  Тестов: {num_tests}")
    print(f"  Top-1 (< 50м): {top1_correct}/{num_tests} ({top1_correct/num_tests*100:.1f}%)")
    print(f"  Top-5 (< 50м): {top5_correct}/{num_tests} ({top5_correct/num_tests*100:.1f}%)")
    print(f"\n  Ошибка позиционирования:")
    print(f"    Средняя: {np.mean(errors_px):.1f} м")
    print(f"    Медиана: {np.median(errors_px):.1f} м")
    print(f"    Макс: {np.max(errors_px):.1f} м")
    print(f"    < 15м: {np.sum(errors_px < 15)/len(errors_px)*100:.1f}%")
    print(f"    < 30м: {np.sum(errors_px < 30)/len(errors_px)*100:.1f}%")
    print(f"    < 50м: {np.sum(errors_px < 50)/len(errors_px)*100:.1f}%")

    src.close()
    print(f"\n{'='*70}")
    if np.median(errors_px) < 30:
        print(f"  ✓ ТЕСТ ПРОЙДЕН: медиана {np.median(errors_px):.1f}м < 30м")
    else:
        print(f"  ✗ ТЕСТ НЕ ПРОЙДЕН: медиана {np.median(errors_px):.1f}м >= 30м")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
