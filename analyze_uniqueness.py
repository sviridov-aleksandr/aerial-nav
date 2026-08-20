#!/usr/bin/env python3
"""
analyze_uniqueness.py — анализ однородности ландшафта маршрута.

Проверяет гипотезу: "тайлы полей одинаковые, модель не может их различить".

Метрики для каждого тайла:
  1. std пикселей — однородность (низкий std = чистое поле)
  2. min/max расстояние до соседних тайлов в embedding-пространстве
     (уникальность: если тайл близок ко всем соседям — он неразличим)

Тест:
  - Строит индекс из 400 тайлов
  - Разделяет на квартили по std
  - Проверяет recall (чистый тайл → поиск) по квартилям

Запуск:
  python3 analyze_uniqueness.py
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

DEVICE = torch.device('cpu')

MAP_PATH = os.path.join(PROJECT_DIR, 'map_cache/antiuav_route_strip.tif')
MODEL_PATH = os.path.join(PROJECT_DIR, 'region_model.pth')
COORDS_PATH = os.path.join(PROJECT_DIR, 'training_data/route_dataset/positive_coords.npy')
RESOLUTION = 0.206  # м/px

TILE_SIZE = 512
INDEX_SIZE = 400


def load_model():
    model = AerialFeatureExtractor(embedding_dim=256).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
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
    arr = normalize(np.array(tiles))
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).to(DEVICE)
    return model(tensor)


def main():
    print("=" * 70)
    print("АНАЛИЗ ОДНОРОДНОСТИ ЛАНДШАФТА МАРШРУТА")
    print("=" * 70)

    model = load_model()
    src = rasterio.open(MAP_PATH)
    coords = np.load(COORDS_PATH)
    print(f"[Analysis] Тайлов в датасете: {len(coords)}")

    rng = np.random.default_rng(42)
    idx_choice = np.sort(rng.choice(len(coords), size=INDEX_SIZE, replace=False))

    # --- Читаем тайлы, считаем std ---
    print(f"\n[Analysis] Чтение {INDEX_SIZE} тайлов и вычисление std...")
    tiles = []
    locs = []
    stds = []
    batch_tiles = []
    BS = 8

    for ci in idx_choice:
        x, y = coords[ci]
        tile = read_tile(src, x, y)
        if tile.max() == 0:
            continue
        tiles.append(tile)
        locs.append((x, y))
        stds.append(tile.std())
        if len(tiles) >= INDEX_SIZE:
            break

    # Индексация
    print(f"[Analysis] Индексация {len(tiles)} тайлов...")
    index_embs = emb_batch(model, tiles)
    locs = np.array(locs)
    stds = np.array(stds)

    # --- Уникальность: расстояние до ближайшего соседа ---
    print(f"[Analysis] Вычисление расстояний между тайлами...")
    D = torch.cdist(index_embs, index_embs)
    # Убираем диагональ
    mask = torch.eye(len(tiles), dtype=torch.bool)
    D_masked = D.clone()
    D_masked[mask] = float('inf')
    min_d = D_masked.min(dim=1).values.numpy()
    # Среднее расстояние до 10 ближайших
    knn_d = D_masked.topk(10, largest=False).values.mean(dim=1).numpy()

    print(f"\n{'='*70}")
    print("СТАТИСТИКА ТАЙЛОВ")
    print(f"{'='*70}")
    print(f"  std пикселей: min={stds.min():.1f}, "
          f"25%={np.percentile(stds, 25):.1f}, "
          f"med={np.median(stds):.1f}, "
          f"75%={np.percentile(stds, 75):.1f}, "
          f"max={stds.max():.1f}")
    print(f"  min d до соседа: min={min_d.min():.3f}, "
          f"med={np.median(min_d):.3f}, "
          f"max={min_d.max():.3f}")
    print(f"  knn(10) d: min={knn_d.min():.3f}, "
          f"med={np.median(knn_d):.3f}, "
          f"max={knn_d.max():.3f}")

    # --- Корреляция std и уникальности ---
    corr = np.corrcoef(stds, min_d)[0, 1]
    print(f"\n  Корреляция std ↔ min_d: {corr:.3f}")
    corr2 = np.corrcoef(stds, knn_d)[0, 1]
    print(f"  Корреляция std ↔ knn_d: {corr2:.3f}")

    # --- Тест recall по квартилям std ---
    print(f"\n{'='*70}")
    print("RECALL ПО КВАРТИЛЯМ std (однородности)")
    print(f"{'='*70}")
    quartiles = np.percentile(stds, [25, 50, 75])

    # Для каждого тайла проверяем, узнаёт ли модель его
    correct_all = 0
    for q_idx, (q_lo, q_hi) in enumerate([(0, quartiles[0]),
                                           (quartiles[0], quartiles[1]),
                                           (quartiles[1], quartiles[2]),
                                           (quartiles[2], 999)]):
        sel = (stds >= q_lo) & (stds < q_hi)
        if sel.sum() == 0:
            continue
        # Чистые тайлы — идеальный случай
        eq = index_embs[sel]
        d = torch.cdist(eq, index_embs)
        nearest = d.argmin(dim=1)
        # Правильный ответ — сам тайл (диагональ)
        # Внутри sel-группы позиции в общем индексе
        positions = np.where(sel)[0]
        correct = (nearest.numpy() == positions).sum()
        recall = correct / len(positions)
        correct_all += correct

        print(f"  std [{q_lo:.0f}-{q_hi:.0f}): "
              f"recall={correct}/{len(positions)} ({recall*100:.0f}%), "
              f"min_d_med={np.median(min_d[sel]):.3f}, "
              f"knn_med={np.median(knn_d[sel]):.3f}")

    print(f"\n  ВСЕГО: recall={correct_all}/{len(tiles)} "
          f"({correct_all/len(tiles)*100:.1f}%)")

    # --- Детальный анализ самых однородных и самых уникальных ---
    print(f"\n{'='*70}")
    print("ТОП-5 САМЫХ ОДНОРОДНЫХ (низкий std)")
    print(f"{'='*70}")
    top_uni = np.argsort(stds)[:5]
    for i in top_uni:
        print(f"  std={stds[i]:.1f}, min_d={min_d[i]:.3f}, "
              f"knn={knn_d[i]:.3f}, pos=({locs[i][0]},{locs[i][1]})")

    print(f"\n{'='*70}")
    print("ТОП-5 САМЫХ УНИКАЛЬНЫХ (высокий min_d)")
    print(f"{'='*70}")
    top_unique = np.argsort(min_d)[-5:]
    for i in top_unique:
        print(f"  std={stds[i]:.1f}, min_d={min_d[i]:.3f}, "
              f"knn={knn_d[i]:.3f}, pos=({locs[i][0]},{locs[i][1]})")

    src.close()
    print(f"\n{'='*70}")
    print("ВЫВОД")
    print(f"{'='*70}")
    if corr2 > 0.5:
        print("  Корреляция std↔уникальность ВЫСОКАЯ: однородные поля")
        print("  действительно неразличимы → нужен больший контекст")
    elif corr2 > 0.2:
        print("  Корреляция std↔уникальность СРЕДНЯЯ: поля частично различимы")
    else:
        print("  Корреляция std↔уникальность НИЗКАЯ: однородность не")
        print("  главная причина провала")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
