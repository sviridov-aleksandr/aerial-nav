#!/usr/bin/env python3
"""
quick_sanity_test.py — быстрая проверка качества модели (без GPU, CPU).

Проверяет, насколько хорошо модель различает правильный тайл среди
случайных. Использует МАЛЫЙ индекс (400 тайлов) и 40 запросов —
достаточно для sanity check, не мешает обучению на GPU.

Метрики:
  - Top-1 recall: доля запросов, где ближайший тайл — правильный
  - Медианная ошибка позиционирования (м)
  - Среднее расстояние d(anchor, positive) vs d(anchor, nearest)

Запуск (пока идёт обучение на GPU):
  /home/alex/my_project_env/bin/python quick_sanity_test.py
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
from augmentations import apply_camera_conditions

# CPU — чтобы не мешать обучению на GPU
DEVICE = torch.device('cpu')

MAP_PATH = os.path.join(PROJECT_DIR, 'map_cache/antiuav_route_strip.tif')
MODEL_PATH = os.path.join(PROJECT_DIR, 'region_model.pth')
COORDS_PATH = os.path.join(PROJECT_DIR, 'training_data/route_dataset/positive_coords.npy')
RESOLUTION = 0.206  # м/px

TILE_SIZE = 512

# Малые размеры для быстрой проверки на CPU
INDEX_SIZE = 400   # тайлов в индексе
NUM_QUERIES = 40   # запросов


def load_model():
    model = AerialFeatureExtractor(embedding_dim=256).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"[Test] Модель: {MODEL_PATH}")
    return model


def normalize(arr):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    out = arr.astype(np.float32) / 255.0
    out = (out - mean) / std
    return out


def read_tile(src, x, y, size=512):
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


@torch.no_grad()
def emb_batch(model, tiles):
    """Embedding батча тайлов (H,W,3) uint8."""
    arr = normalize(np.array(tiles))
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).to(DEVICE)
    return model(tensor)


def main():
    print("=" * 70)
    print("БЫСТРЫЙ SANITY-ТЕСТ (CPU, малый индекс)")
    print("=" * 70)

    model = load_model()
    src = rasterio.open(MAP_PATH)
    coords = np.load(COORDS_PATH)
    print(f"[Test] Тайлов в датасете: {len(coords)}")

    rng = np.random.default_rng(42)
    idx_choice = rng.choice(len(coords), size=INDEX_SIZE, replace=False)
    idx_choice = np.sort(idx_choice)

    # 1. Индексация
    print(f"\n[Test] Индексация {INDEX_SIZE} тайлов на CPU...")
    t0 = time.time()
    index_embs = []
    index_locs = []
    batch_tiles = []
    batch_locs = []
    BS = 8
    for i, ci in enumerate(idx_choice):
        x, y = coords[ci]
        tile = read_tile(src, x, y)
        if tile.max() == 0:
            continue
        batch_tiles.append(tile)
        batch_locs.append((x, y))
        if len(batch_tiles) >= BS:
            embs = emb_batch(model, batch_tiles)
            index_embs.append(embs)
            index_locs.extend(batch_locs)
            batch_tiles, batch_locs = [], []
    if batch_tiles:
        embs = emb_batch(model, batch_tiles)
        index_embs.append(embs)
        index_locs.extend(batch_locs)
    index_embs = torch.cat(index_embs, dim=0)
    index_locs = np.array(index_locs)
    print(f"[Test] Индекс: {index_embs.shape[0]} тайлов за {time.time()-t0:.0f}s")

    # 2. Запросы: аугментированный тайл → ближайший в индексе
    print(f"\n[Test] {NUM_QUERIES} запросов (аугментированный кадр камеры)...")
    t0 = time.time()
    top1_correct = 0
    errs = []
    d_pos_list = []
    d_neg_list = []

    q_choice = rng.choice(len(idx_choice), size=NUM_QUERIES, replace=False)
    for qi, pos in enumerate(q_choice):
        ci = idx_choice[pos]
        x, y = coords[ci]
        # Чистый тайл (positive) и аугментированный (anchor)
        pos_tile = read_tile(src, x, y)
        img = Image.fromarray(pos_tile)
        anchor_tile = np.array(apply_camera_conditions(img))

        emb_a = emb_batch(model, [anchor_tile])[0]
        emb_p = emb_batch(model, [pos_tile])[0]

        # Расстояния до индекса
        d = torch.cdist(emb_a.unsqueeze(0), index_embs).squeeze(0)
        nearest = d.argmin().item()

        # Правильный ли тайл в индексе и где он
        true_in_index = np.where((index_locs[:, 0] == x) & (index_locs[:, 1] == y))[0]
        d_pos = torch.cdist(emb_p.unsqueeze(0), index_embs).squeeze(0)[pos] if pos < len(index_locs) else None

        # Ошибка: расстояние до предсказанного тайла
        pred_x, pred_y = index_locs[nearest]
        err = math.sqrt((pred_x - x) ** 2 + (pred_y - y) ** 2) * RESOLUTION
        errs.append(err)

        if len(true_in_index) > 0 and nearest == true_in_index[0]:
            top1_correct += 1

        d_pos_list.append(d[pos].item() if pos < len(index_embs) else float('nan'))
        # Среднее расстояние до остальных (negative)
        mask = np.ones(len(index_embs), dtype=bool)
        mask[pos] = False
        d_neg_list.append(d[mask].mean().item())

        if (qi + 1) % 10 == 0:
            print(f"  {qi+1}/{NUM_QUERIES}...")

    errs = np.array(errs)
    print(f"\n[Test] Запросы выполнены за {time.time()-t0:.0f}s")

    # 3. Результаты
    print(f"\n{'='*70}")
    print("РЕЗУЛЬТАТЫ SANITY-ТЕСТА")
    print(f"{'='*70}")
    print(f"  Запросов: {len(errs)}")
    print(f"  Top-1 recall (правильный тайл): "
          f"{top1_correct}/{len(errs)} ({top1_correct/len(errs)*100:.1f}%)")
    print(f"  (случайное угадывание: {100/INDEX_SIZE:.2f}%)")
    print(f"\n  Ошибка позиционирования:")
    print(f"    Медиана: {np.median(errs):.1f} м")
    print(f"    Средняя: {np.mean(errs):.1f} м")
    print(f"    < 50м: {np.sum(errs < 50)/len(errs)*100:.1f}%")
    print(f"    < 100м: {np.sum(errs < 100)/len(errs)*100:.1f}%")
    print(f"    < 200м: {np.sum(errs < 200)/len(errs)*100:.1f}%")
    print(f"\n  Расстояния (L2, нормализованные эмбеддинги):")
    print(f"    d(anchor, true):  {np.mean(d_pos_list):.3f}")
    print(f"    d(anchor, other): {np.mean(d_neg_list):.3f}")
    print(f"    Разрыв:           {np.mean(d_neg_list) - np.mean(d_pos_list):.3f}")

    print(f"\n  Вывод:")
    if top1_correct / len(errs) > 0.5 and np.median(errs) < 100:
        print(f"    ✓ Модель работает: top-1 {top1_correct/len(errs)*100:.0f}% >> случайного")
    else:
        print(f"    ✗ Модель плохо различает тайлы — проверьте обучение")
    print(f"{'='*70}")

    src.close()


if __name__ == '__main__':
    main()
