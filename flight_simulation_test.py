#!/usr/bin/env python3
"""
flight_simulation_test.py — моделирование полёта дрона с многоуровневым индексом.

Три эксперимента:

ЭКСПЕРИМЕНТ 1 — ВЗЛЁТ (масштаб/высота):
  Для каждой высоты строит кадр камеры, ищет правильный уровень индекса,
  проверяет recall. Сравнивает одноуровневый и многоуровневый подходы.

ЭКСПЕРИМЕНТ 2 — ВРАЩЕНИЕ (рыскание):
  На оптимальной высоте вращает кадр на 0-180°, проверяет recall.

ЭКСПЕРИМЕНТ 3 — СЕЗОННОСТЬ:
  Применяет сезонные аугментации к кадру, проверяет recall.

Запуск (CPU, не мешает GPU):
  python3 flight_simulation_test.py
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
from augmentations import apply_seasonal

DEVICE = torch.device('cpu')

MAP_PATH = os.path.join(PROJECT_DIR, 'map_cache/antiuav_route_strip.tif')
MODEL_PATH = os.path.join(PROJECT_DIR, 'region_model.pth')
COORDS_PATH = os.path.join(PROJECT_DIR, 'training_data/route_dataset/positive_coords.npy')
RESOLUTION = 0.206  # м/px

TILE_SIZE = 512
INDEX_SIZE = 400
NUM_TEST_TILES = 20

# Камера
CAM_W = 3840
CAM_FOV_H = 90.0

# Уровни индекса (синхронизированы с multi_level_index.py)
LEVELS = [
    {'patch_size': 512,   'alt_min': 0,   'alt_max': 550},
    {'patch_size': 1024,  'alt_min': 550, 'alt_max': 950},
    {'patch_size': 1792,  'alt_min': 950, 'alt_max': 1500},
]

# Высоты для эксперимента 1
ALTITUDES = [100, 200, 300, 400, 450, 700, 800, 1000, 1200]

# Углы для эксперимента 2
ANGLES = [0, 5, 10, 15, 20, 30, 45, 90, 180]


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


def altitude_to_level(altitude_m):
    """Выбор уровня индекса по высоте."""
    for i, lvl in enumerate(LEVELS):
        if lvl['alt_min'] <= altitude_m < lvl['alt_max']:
            return i
    return len(LEVELS) - 1


def read_patch(src, cx, cy, patch_size, tile_size=512):
    """Читает патч patch_size×patch_size с центром (cx, cy), ресайз в tile_size."""
    x1 = int(cx - patch_size / 2)
    y1 = int(cy - patch_size / 2)
    win = Window(x1, y1, patch_size, patch_size)
    try:
        data = src.read(window=win)
        arr = np.transpose(data, (1, 2, 0))
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
    except Exception:
        return np.zeros((tile_size, tile_size, 3), dtype=np.uint8)

    h, w = arr.shape[:2]
    if h < patch_size or w < patch_size:
        padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        padded[:h, :w] = arr
        arr = padded

    img = Image.fromarray(arr)
    img = img.resize((tile_size, tile_size), Image.BILINEAR)
    return np.array(img)


@torch.no_grad()
def emb_batch(model, tiles):
    arr = normalize(np.array(tiles))
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).to(DEVICE)
    return model(tensor)


def camera_frame_from_map(src, center_px, altitude, tile_size=512):
    """
    Имитация кадра камеры с высоты altitude м.
    Возвращает (frame, scale) — кадр 512×512 и масштаб относительно тайла.
    """
    footprint_w = 2 * altitude * math.tan(math.radians(CAM_FOV_H / 2))
    gsd_cam = footprint_w / CAM_W
    patch_m = gsd_cam * tile_size
    patch_px_map = int(patch_m / RESOLUTION)
    scale = patch_px_map / tile_size

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
        return np.zeros((tile_size, tile_size, 3), dtype=np.uint8), scale

    h, w = map_crop.shape[:2]
    if h < patch_px_map or w < patch_px_map:
        padded = np.zeros((patch_px_map, patch_px_map, 3), dtype=np.uint8)
        padded[:h, :w] = map_crop
        map_crop = padded

    img = Image.fromarray(map_crop)
    img = img.resize((tile_size, tile_size), Image.BILINEAR)
    return np.array(img), scale


def rotate_frame(frame, angle):
    """Поворот кадра. Без чёрных краёв: reflect-padding → поворот → вырезка центра."""
    from PIL import ImageOps
    img = Image.fromarray(frame)
    w, h = img.size
    diag = int(math.ceil(math.sqrt(w**2 + h**2)))
    pad_w = (diag - w) // 2
    pad_h = (diag - h) // 2
    img = ImageOps.expand(img, border=(pad_w, pad_h), fill=0)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
    cx, cy = img.size[0] // 2, img.size[1] // 2
    img = img.crop((cx - w // 2, cy - h // 2, cx + w - w // 2, cy + h - h // 2))
    return np.array(img)


def build_multi_level_index(model, src, coords_subset):
    """
    Строит многоуровневый индекс: для каждого уровня читает патч нужного размера,
    ресайзит в 512, вычисляет embedding.
    """
    index_data = {}

    for lvl_idx, lvl in enumerate(LEVELS):
        patch_size = lvl['patch_size']
        print(f"  Уровень {lvl_idx}: патч {patch_size}×{patch_size} "
              f"(h={lvl['alt_min']}-{lvl['alt_max']} м)")

        all_embs = []
        all_locs = []
        batch_tiles = []
        batch_locs = []
        BS = 8

        for i, (x, y) in enumerate(coords_subset):
            cx = x + TILE_SIZE / 2
            cy = y + TILE_SIZE / 2
            tile = read_patch(src, cx, cy, patch_size)
            if tile.max() == 0:
                continue
            batch_tiles.append(tile)
            batch_locs.append((x, y))

            if len(batch_tiles) >= BS:
                embs = emb_batch(model, batch_tiles)
                all_embs.append(embs)
                all_locs.extend(batch_locs)
                batch_tiles = []
                batch_locs = []

        if batch_tiles:
            embs = emb_batch(model, batch_tiles)
            all_embs.append(embs)
            all_locs.extend(batch_locs)

        embs_cat = torch.cat(all_embs, dim=0)
        locs_arr = np.array(all_locs)
        index_data[lvl_idx] = (embs_cat, locs_arr)
        print(f"    → {embs_cat.shape[0]} тайлов")

    return index_data


def main():
    print("=" * 70)
    print("МОДЕЛИРОВАНИЕ ПОЛЁТА: многоуровневый индекс")
    print("=" * 70)

    model = load_model()
    src = rasterio.open(MAP_PATH)
    coords = np.load(COORDS_PATH)
    print(f"[Sim] Тайлов в датасете: {len(coords)}")

    rng = np.random.default_rng(42)
    idx_choice = np.sort(rng.choice(len(coords), size=INDEX_SIZE, replace=False))
    coords_subset = coords[idx_choice]

    # --- Построение многоуровневого индекса ---
    print(f"\n[Sim] Индексация {INDEX_SIZE} тайлов × {len(LEVELS)} уровней...")
    index_data = build_multi_level_index(model, src, coords_subset)

    # Тестовые тайлы (используем уровень 0 — он есть для всех)
    test_indices = rng.choice(len(coords_subset), size=min(NUM_TEST_TILES, len(coords_subset)), replace=False)

    # ================================================================
    # ЭКСПЕРИМЕНТ 1: ВЗЛЁТ (масштаб) — многоуровневый индекс
    # ================================================================
    print(f"\n{'='*70}")
    print("ЭКСПЕРИМЕНТ 1: ВЗЛЁТ (многоуровневый индекс)")
    print(f"  Высоты: {ALTITUDES} м")
    print(f"  Тестовых тайлов: {len(test_indices)}")
    print(f"{'='*70}")

    results_alt = {}

    for alt in ALTITUDES:
        lvl_idx = altitude_to_level(alt)
        index_embs, locs = index_data[lvl_idx]

        correct = 0
        d_true_list = []
        d_other_list = []
        errs = []

        for ti in test_indices:
            if ti >= len(locs):
                continue
            x, y = locs[ti]
            center_px = (x + TILE_SIZE / 2, y + TILE_SIZE / 2)

            frame, scale = camera_frame_from_map(src, center_px, alt, TILE_SIZE)
            if frame.max() == 0:
                continue

            eq = emb_batch(model, [frame])[0]
            d = torch.cdist(eq.unsqueeze(0), index_embs).squeeze(0)
            nearest = d.argmin().item()

            if nearest == ti:
                correct += 1

            d_true_list.append(d[ti].item())
            mask = np.ones(len(index_embs), dtype=bool)
            mask[ti] = False
            d_other_list.append(d[mask].min().item())

            px, py = locs[nearest]
            err = math.sqrt((px - x) ** 2 + (py - y) ** 2) * RESOLUTION
            errs.append(err)

        n = len(d_true_list)
        recall = correct / n * 100 if n > 0 else 0
        med_err = float(np.median(errs)) if errs else float('nan')
        results_alt[alt] = {
            'recall': recall,
            'd_true': np.mean(d_true_list) if d_true_list else 0,
            'd_other': np.mean(d_other_list) if d_other_list else 0,
            'scale': alt / 400,
            'med_err': med_err,
            'level': lvl_idx,
        }

        print(f"\n  h={alt:4d}м (уровень {lvl_idx}, масштаб {alt/400:.2f}×):")
        print(f"    Recall:      {correct}/{n} ({recall:.0f}%)")
        print(f"    d(true):     {results_alt[alt]['d_true']:.3f}")
        print(f"    d(other):    {results_alt[alt]['d_other']:.3f}")
        print(f"    Разрыв:      {results_alt[alt]['d_other'] - results_alt[alt]['d_true']:.3f}")
        print(f"    Медиана err: {med_err:.0f} м")

    best_alt = max(results_alt, key=lambda a: results_alt[a]['recall'])
    print(f"\n  >>> Оптимальная высота: {best_alt}м "
          f"(recall={results_alt[best_alt]['recall']:.0f}%)")

    # ================================================================
    # ЭКСПЕРИМЕНТ 2: ВРАЩЕНИЕ (рыскание)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"ЭКСПЕРИМЕНТ 2: ВРАЩЕНИЕ (рыскание)")
    print(f"  Высота: {best_alt}м (оптимальная из эксперимента 1)")
    print(f"  Углы: {ANGLES}°")
    print(f"{'='*70}")

    best_lvl = altitude_to_level(best_alt)
    index_embs, locs = index_data[best_lvl]

    for angle in ANGLES:
        correct = 0
        d_true_list = []
        d_other_list = []
        errs = []

        for ti in test_indices:
            if ti >= len(locs):
                continue
            x, y = locs[ti]
            center_px = (x + TILE_SIZE / 2, y + TILE_SIZE / 2)

            frame, _ = camera_frame_from_map(src, center_px, best_alt, TILE_SIZE)
            if frame.max() == 0:
                continue

            frame_rot = rotate_frame(frame, angle)
            eq = emb_batch(model, [frame_rot])[0]
            d = torch.cdist(eq.unsqueeze(0), index_embs).squeeze(0)
            nearest = d.argmin().item()

            if nearest == ti:
                correct += 1

            d_true_list.append(d[ti].item())
            mask = np.ones(len(index_embs), dtype=bool)
            mask[ti] = False
            d_other_list.append(d[mask].min().item())

            px, py = locs[nearest]
            err = math.sqrt((px - x) ** 2 + (py - y) ** 2) * RESOLUTION
            errs.append(err)

        n = len(d_true_list)
        recall = correct / n * 100 if n > 0 else 0
        med_err = float(np.median(errs)) if errs else float('nan')

        print(f"\n  угол={angle:3d}°:")
        print(f"    Recall:      {correct}/{n} ({recall:.0f}%)")
        print(f"    d(true):     {np.mean(d_true_list):.3f}")
        print(f"    d(other):    {np.mean(d_other_list):.3f}")
        print(f"    Разрыв:      {np.mean(d_other_list) - np.mean(d_true_list):.3f}")
        print(f"    Медиана err: {med_err:.0f} м")

    # ================================================================
    # ЭКСПЕРИМЕНТ 3: СЕЗОННОСТЬ
    # ================================================================
    print(f"\n{'='*70}")
    print(f"ЭКСПЕРИМЕНТ 3: СЕЗОННОСТЬ")
    print(f"  Высота: {best_alt}м")
    print(f"{'='*70}")

    seasons = ['winter', 'autumn', 'summer', 'rain']
    for season in seasons:
        correct = 0
        d_true_list = []
        d_other_list = []
        errs = []

        for ti in test_indices:
            if ti >= len(locs):
                continue
            x, y = locs[ti]
            center_px = (x + TILE_SIZE / 2, y + TILE_SIZE / 2)

            frame, _ = camera_frame_from_map(src, center_px, best_alt, TILE_SIZE)
            if frame.max() == 0:
                continue

            # Применяем сезонную аугментацию
            img = Image.fromarray(frame)
            img = apply_seasonal(img, intensity=0.7)
            frame_seasonal = np.array(img)

            eq = emb_batch(model, [frame_seasonal])[0]
            d = torch.cdist(eq.unsqueeze(0), index_embs).squeeze(0)
            nearest = d.argmin().item()

            if nearest == ti:
                correct += 1

            d_true_list.append(d[ti].item())
            mask = np.ones(len(index_embs), dtype=bool)
            mask[ti] = False
            d_other_list.append(d[mask].min().item())

            px, py = locs[nearest]
            err = math.sqrt((px - x) ** 2 + (py - y) ** 2) * RESOLUTION
            errs.append(err)

        n = len(d_true_list)
        recall = correct / n * 100 if n > 0 else 0
        med_err = float(np.median(errs)) if errs else float('nan')

        print(f"\n  {season:8s}:")
        print(f"    Recall:      {correct}/{n} ({recall:.0f}%)")
        print(f"    d(true):     {np.mean(d_true_list):.3f}")
        print(f"    d(other):    {np.mean(d_other_list):.3f}")
        print(f"    Разрыв:      {np.mean(d_other_list) - np.mean(d_true_list):.3f}")
        print(f"    Медиана err: {med_err:.0f} м")

    # ================================================================
    # ИТОГ
    # ================================================================
    print(f"\n{'='*70}")
    print("ИТОГ ЭКСПЕРИМЕНТОВ")
    print(f"{'='*70}")
    print(f"\n  Масштаб (высота → recall, уровень):")
    for alt in ALTITUDES:
        r = results_alt[alt]
        bar = '█' * int(r['recall'] / 5)
        print(f"    h={alt:4d}м ({r['scale']:.2f}×, L{r['level']}): {r['recall']:5.0f}%  {bar}")
    print(f"\n  Оптимальная высота: {best_alt}м")
    print(f"{'='*70}")

    src.close()


if __name__ == '__main__':
    main()