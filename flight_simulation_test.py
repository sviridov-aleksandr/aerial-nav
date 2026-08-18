#!/usr/bin/env python3
"""
flight_simulation_test.py — моделирование полёта дрона.

Два эксперимента, изолирующих факторы:

ЭКСПЕРИМЕНТ 1 — ВЗЛЁТ (масштаб):
  Берём N тайлов из индекса. Для каждого имитируем кадр камеры
  с высот 100, 150, 200, 250, 300, 350, 400, 450, 700 м.
  Кадр = вырезка из карты (центр тайла), ресемплинг в 512×512.
  Без аугментаций — чистый масштаб.
  Цель: найти высоту, на которой модель начинает терять совпадение.

ЭКСПЕРИМЕНТ 2 — ВРАЩЕНИЕ (рыскание):
  Берём те же N тайлов. Для каждого имитируем кадр с оптимальной
  высоты (из эксперимента 1), вращаем на 0, 5, 10, 15, 20, 30, 45, 90, 180°.
  Цель: найти угол, на котором модель ломается.

Запуск (CPU, не мешает GPU):
  /home/alex/my_project_env/bin/python flight_simulation_test.py
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
NUM_TEST_TILES = 20  # тайлов для тестирования

# Камера
CAM_W = 3840
CAM_FOV_H = 90.0

# Высоты для эксперимента 1
ALTITUDES = [100, 150, 200, 250, 300, 350, 400, 450, 700]

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
    """
    Имитация кадра камеры с высоты altitude м.
    Возвращает (frame, scale) — кадр 512×512 и масштаб относительно тайла.
    """
    footprint_w = 2 * altitude * math.tan(math.radians(CAM_FOV_H / 2))
    gsd_cam = footprint_w / CAM_W
    patch_m = gsd_cam * tile_size
    patch_px_map = int(patch_m / RESOLUTION)
    # Масштаб: во сколько раз кадр камеры больше/меньше тайла обучения
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
    """Поворот кадра на angle градусов. Без чёрных краёв: reflect-padding → поворот → вырезка центра."""
    import math
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


def main():
    print("=" * 70)
    print("МОДЕЛИРОВАНИЕ ПОЛЁТА: масштаб + вращение")
    print("=" * 70)

    model = load_model()
    src = rasterio.open(MAP_PATH)
    coords = np.load(COORDS_PATH)
    print(f"[Sim] Тайлов в датасете: {len(coords)}")

    rng = np.random.default_rng(42)
    idx_choice = np.sort(rng.choice(len(coords), size=INDEX_SIZE, replace=False))

    # --- Индексация (чистые тайлы) ---
    print(f"\n[Sim] Индексация {INDEX_SIZE} тайлов...")
    tiles = []
    locs = []
    batch_tiles = []
    BS = 8
    index_embs = None
    for ci in idx_choice:
        x, y = coords[ci]
        tile = read_tile(src, x, y)
        if tile.max() == 0:
            continue
        batch_tiles.append(tile)
        locs.append((x, y))
        if len(batch_tiles) >= BS:
            embs = emb_batch(model, batch_tiles)
            index_embs = embs if index_embs is None else torch.cat([index_embs, embs], dim=0)
            batch_tiles = []
    if batch_tiles:
        embs = emb_batch(model, batch_tiles)
        index_embs = torch.cat([index_embs, embs], dim=0) if index_embs is not None else embs
    locs = np.array(locs)
    print(f"[Sim] Индекс: {index_embs.shape[0]} тайлов")

    # Выбираем тестовые тайлы из индекса
    test_indices = rng.choice(len(locs), size=min(NUM_TEST_TILES, len(locs)), replace=False)

    # ================================================================
    # ЭКСПЕРИМЕНТ 1: ВЗЛЁТ (масштаб)
    # ================================================================
    print(f"\n{'='*70}")
    print("ЭКСПЕРИМЕНТ 1: ВЗЛЁТ (масштаб)")
    print(f"  Высоты: {ALTITUDES} м")
    print(f"  Тестовых тайлов: {len(test_indices)}")
    print(f"{'='*70}")

    # Для каждой высоты: recall, d(true), d(nearest other), масштаб
    results_alt = {}

    for alt in ALTITUDES:
        correct = 0
        d_true_list = []
        d_other_list = []
        scales = []
        errs = []

        for ti in test_indices:
            x, y = locs[ti]
            center_px = (x + TILE_SIZE / 2, y + TILE_SIZE / 2)

            frame, scale = camera_frame_from_map(src, center_px, alt, TILE_SIZE)
            if frame.max() == 0:
                continue
            scales.append(scale)

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
        avg_scale = np.mean(scales) if scales else 0
        results_alt[alt] = {
            'recall': recall,
            'd_true': np.mean(d_true_list),
            'd_other_min': np.mean(d_other_list),
            'scale': avg_scale,
            'med_err': med_err,
        }

        print(f"\n  h={alt:4d}м (масштаб {avg_scale:.2f}×):")
        print(f"    Recall:      {correct}/{n} ({recall:.0f}%)")
        print(f"    d(true):     {np.mean(d_true_list):.3f}")
        print(f"    d(other):    {np.mean(d_other_list):.3f}")
        print(f"    Разрыв:      {np.mean(d_other_list) - np.mean(d_true_list):.3f}")
        print(f"    Медиана err: {med_err:.0f} м")

    # Находим оптимальную высоту (макс recall)
    best_alt = max(results_alt, key=lambda a: results_alt[a]['recall'])
    print(f"\n  >>> Оптимальная высота: {best_alt}м "
          f"(recall={results_alt[best_alt]['recall']:.0f}%, "
          f"масштаб={results_alt[best_alt]['scale']:.2f}×)")

    # ================================================================
    # ЭКСПЕРИМЕНТ 2: ВРАЩЕНИЕ (рыскание)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"ЭКСПЕРИМЕНТ 2: ВРАЩЕНИЕ (рыскание)")
    print(f"  Высота: {best_alt}м (оптимальная из эксперимента 1)")
    print(f"  Углы: {ANGLES}°")
    print(f"  Тестовых тайлов: {len(test_indices)}")
    print(f"{'='*70}")

    for angle in ANGLES:
        correct = 0
        d_true_list = []
        d_other_list = []
        errs = []

        for ti in test_indices:
            x, y = locs[ti]
            center_px = (x + TILE_SIZE / 2, y + TILE_SIZE / 2)

            frame, _ = camera_frame_from_map(src, center_px, best_alt, TILE_SIZE)
            if frame.max() == 0:
                continue

            # Поворот кадра
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
    # ИТОГ
    # ================================================================
    print(f"\n{'='*70}")
    print("ИТОГ ЭКСПЕРИМЕНТОВ")
    print(f"{'='*70}")
    print(f"\n  Масштаб (высота → recall):")
    for alt in ALTITUDES:
        r = results_alt[alt]
        bar = '█' * int(r['recall'] / 5)
        print(f"    h={alt:4d}м ({r['scale']:.2f}×): {r['recall']:5.0f}%  {bar}")
    print(f"\n  Оптимальная высота: {best_alt}м")
    print(f"{'='*70}")

    src.close()


if __name__ == '__main__':
    main()
