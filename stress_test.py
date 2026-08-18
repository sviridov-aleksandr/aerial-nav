#!/usr/bin/env python3
"""
stress_test.py — стресс-тест модели на всех уровнях аугментаций.

Проверяет recall и ошибку позиционирования для curriculum-уровней 0-3.
Запускается на CPU (не мешает обучению на GPU).

Два режима проверки:
  1. Аугментированный тайл → поиск в индексе (чистые аугментации)
  2. Кадр камеры с высоты 700/1000/1200 м → поиск в индексе (реалистичный)

Запуск:
  /home/alex/my_project_env/bin/python stress_test.py
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

DEVICE = torch.device('cpu')

MAP_PATH = os.path.join(PROJECT_DIR, 'map_cache/antiuav_route_strip.tif')
MODEL_PATH = os.path.join(PROJECT_DIR, 'region_model.pth')
COORDS_PATH = os.path.join(PROJECT_DIR, 'training_data/route_dataset/positive_coords.npy')
RESOLUTION = 0.206  # м/px

TILE_SIZE = 512
INDEX_SIZE = 400
NUM_QUERIES = 40

# Камера
CAM_W = 3840
CAM_FOV_H = 90.0
ALTITUDES = [700.0, 1000.0, 1200.0]


def load_model():
    model = AerialFeatureExtractor(embedding_dim=256).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        epoch = ckpt.get('epoch', '?')
        loss = ckpt.get('loss', '?')
        print(f"[Stress] Модель: epoch={epoch}, loss={loss}")
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


def main():
    print("=" * 70)
    print("СТРЕСС-ТЕСТ МОДЕЛИ (CPU, все уровни аугментаций)")
    print("=" * 70)

    model = load_model()
    src = rasterio.open(MAP_PATH)
    coords = np.load(COORDS_PATH)
    print(f"[Stress] Тайлов в датасете: {len(coords)}")

    rng = np.random.default_rng(42)
    idx_choice = np.sort(rng.choice(len(coords), size=INDEX_SIZE, replace=False))

    # --- Индексация ---
    print(f"\n[Stress] Индексация {INDEX_SIZE} тайлов на CPU...")
    t0 = time.time()
    index_tiles = []
    index_locs = []
    batch_tiles = []
    BS = 8
    for ci in idx_choice:
        x, y = coords[ci]
        tile = read_tile(src, x, y)
        if tile.max() == 0:
            continue
        batch_tiles.append(tile)
        index_locs.append((x, y))
        if len(batch_tiles) >= BS:
            embs = emb_batch(model, batch_tiles)
            if 'index_embs' not in dir():
                index_embs = embs
            else:
                index_embs = torch.cat([index_embs, embs], dim=0)
            batch_tiles = []
    if batch_tiles:
        embs = emb_batch(model, batch_tiles)
        index_embs = torch.cat([index_embs, embs], dim=0)
    index_locs = np.array(index_locs)
    print(f"[Stress] Индекс: {index_embs.shape[0]} тайлов за {time.time()-t0:.0f}s")

    # --- Тест 1: аугментированные тайлы по уровням 0-3 ---
    print(f"\n{'='*70}")
    print("ТЕСТ 1: Аугментированные тайлы (curriculum levels 0-3)")
    print(f"{'='*70}")

    q_choice = rng.choice(len(idx_choice), size=NUM_QUERIES, replace=False)

    for lvl in range(4):
        correct = 0
        errs = []
        d_pos_list = []
        d_neg_list = []
        t0 = time.time()

        for pos in q_choice:
            ci = idx_choice[pos]
            x, y = coords[ci]
            tile = read_tile(src, x, y)
            q = np.array(apply_camera_conditions(Image.fromarray(tile), level=lvl))

            eq = emb_batch(model, [q])[0]
            d = torch.cdist(eq.unsqueeze(0), index_embs).squeeze(0)
            nearest = d.argmin().item()

            # Правильная позиция в индексе
            true_pos = np.where((index_locs[:, 0] == x) & (index_locs[:, 1] == y))[0]
            if len(true_pos) > 0 and nearest == true_pos[0]:
                correct += 1

            px, py = index_locs[nearest]
            err = math.sqrt((px - x) ** 2 + (py - y) ** 2) * RESOLUTION
            errs.append(err)

            # Расстояния
            tp = true_pos[0] if len(true_pos) > 0 else 0
            d_pos_list.append(d[tp].item())
            mask = np.ones(len(index_embs), dtype=bool)
            mask[tp] = False
            d_neg_list.append(d[mask].mean().item())

        errs = np.array(errs)
        elapsed = time.time() - t0
        print(f"\n  Level {lvl}:")
        print(f"    Recall:     {correct}/{NUM_QUERIES} ({correct/NUM_QUERIES*100:.1f}%)")
        print(f"    Медиана:    {np.median(errs):.1f} м")
        print(f"    Средняя:    {np.mean(errs):.1f} м")
        print(f"    < 30м:      {np.sum(errs < 30)/len(errs)*100:.1f}%")
        print(f"    < 50м:      {np.sum(errs < 50)/len(errs)*100:.1f}%")
        print(f"    < 100м:     {np.sum(errs < 100)/len(errs)*100:.1f}%")
        print(f"    d(true):    {np.mean(d_pos_list):.3f}")
        print(f"    d(other):   {np.mean(d_neg_list):.3f}")
        print(f"    Разрыв:     {np.mean(d_neg_list) - np.mean(d_pos_list):.3f}")
        print(f"    Время:      {elapsed:.0f}s")

    # --- Тест 2: реалистичный кадр камеры (высоты 700/1000/1200 м) ---
    print(f"\n{'='*70}")
    print("ТЕСТ 2: Кадр камеры с высоты (реалистичный)")
    print(f"{'='*70}")

    for alt in ALTITUDES:
        correct = 0
        errs = []
        t0 = time.time()

        for pos in q_choice:
            ci = idx_choice[pos]
            x, y = coords[ci]
            center_px = (x + TILE_SIZE / 2, y + TILE_SIZE / 2)

            # Кадр камеры с высоты
            frame = camera_frame_from_map(src, center_px, alt, TILE_SIZE)
            if frame.max() == 0:
                continue

            eq = emb_batch(model, [frame])[0]
            d = torch.cdist(eq.unsqueeze(0), index_embs).squeeze(0)
            nearest = d.argmin().item()

            true_pos = np.where((index_locs[:, 0] == x) & (index_locs[:, 1] == y))[0]
            if len(true_pos) > 0 and nearest == true_pos[0]:
                correct += 1

            px, py = index_locs[nearest]
            err = math.sqrt((px - x) ** 2 + (py - y) ** 2) * RESOLUTION
            errs.append(err)

        errs = np.array(errs)
        elapsed = time.time() - t0
        print(f"\n  h={int(alt)}м:")
        print(f"    Recall:     {correct}/{len(errs)} ({correct/len(errs)*100:.1f}%)")
        print(f"    Медиана:    {np.median(errs):.1f} м")
        print(f"    < 30м:      {np.sum(errs < 30)/len(errs)*100:.1f}%")
        print(f"    < 50м:      {np.sum(errs < 50)/len(errs)*100:.1f}%")
        print(f"    < 100м:     {np.sum(errs < 100)/len(errs)*100:.1f}%")
        print(f"    Время:      {elapsed:.0f}s")

    # --- Итог ---
    print(f"\n{'='*70}")
    print("ИТОГ")
    print(f"{'='*70}")
    print(f"  Случайное угадывание: {100/INDEX_SIZE:.2f}%")
    print(f"  Модель: region_model.pth")
    print(f"  Индекс: {index_embs.shape[0]} тайлов")
    print(f"{'='*70}")

    src.close()


if __name__ == '__main__':
    main()
