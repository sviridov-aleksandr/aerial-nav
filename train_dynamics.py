#!/usr/bin/env python3
"""
train_dynamics.py — дообучение сиамской сети на динамике полёта (этап 4).

Цель: научить модель возвращать близкие эмбеддинги для кадров одного
участка, снятых в разное время (последовательность кадров).

Мотивация:
  - При 5 Гц и V_max=120 км/ч (33.3 м/с) смещение между кадрами = 6.7 м
  - При 1 Гц смещение = 33 м (13-32% тайла)
  - Модель должна давать близкие эмбеддинги для перекрывающихся кадров
  - Это основа для одометрии и фильтра Калмана

Триплеты:
  - anchor:   кадр в позиции P (патч из карты с масштабом + аугментации)
  - positive: кадр в позиции P+Δ (смещение 0-33 м, тот же масштаб)
  - negative: другой участок карты (чистый тайл)

Смещение Δ:
  - Диапазон 0-33 м (покрывает 1-5 Гц при V_max=120 км/ч)
  - Направление случайное (равномерное по окружности)
  - В пикселях: 0-160 px при 0.206 м/px

Масштаб пары одинаковый (один полёт, одна высота) — модель учит именно
смещение, а не изменение высоты.

Использование:
  python3 train_dynamics.py
"""

import os
import sys
import math
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.windows import Window
from PIL import Image
import random
from tqdm import tqdm
import time

Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from siamese_network import AerialFeatureExtractor, TripletLoss
from augmentations import apply_camera_conditions, CURRICULUM_LEVELS

multiprocessing.set_start_method('fork', force=True)


class DynamicsDataset(Dataset):
    """
    Датасет для обучения на динамике полёта.

    Генерирует триплеты (anchor, positive, negative):
      - anchor:   кадр в позиции P (масштаб + аугментации)
      - positive: кадр в позиции P+Δ (смещение 0-max_shift_px, тот же масштаб)
      - negative: чистый тайл другого участка
    """

    def __init__(self, map_path: str, coords_path: str,
                 tile_size: int = 512, hard_neg_prob: float = 0.5,
                 max_shift_px: int = 160, scale_range=(0.25, 2.0),
                 aug_level: int = 2):
        """
        Args:
            map_path: путь к GeoTIFF карте
            coords_path: путь к файлу с координатами тайлов (.npy)
            tile_size: размер тайла (512)
            hard_neg_prob: вероятность выбора negative из соседних тайлов
            max_shift_px: максимальное смещение кадра (px карты)
                          (160 px = 33 м при 0.206 м/px)
            scale_range: диапазон масштаба (высота полёта)
            aug_level: curriculum-этап аугментаций (0-2)
        """
        self.map_path = map_path
        self.tile_size = tile_size
        self.hard_neg_prob = hard_neg_prob
        self.max_shift_px = max_shift_px
        self.scale_range = scale_range
        self.aug_level = aug_level

        print(f"[DynamicsDataset] Открываем карту: {map_path}")
        with rasterio.open(map_path) as src:
            self.width = src.width
            self.height = src.height
        self._src = None
        print(f"[DynamicsDataset] Карта: {self.width}x{self.height} px")

        print(f"[DynamicsDataset] Загружаем координаты: {coords_path}")
        self.coords = np.load(coords_path)
        self.num_tiles = len(self.coords)
        print(f"[DynamicsDataset] Тайлов: {self.num_tiles}")
        print(f"[DynamicsDataset] Смещение: 0-{max_shift_px} px "
              f"({max_shift_px*0.206:.1f} м)")

    def __len__(self):
        return self.num_tiles

    def _ensure_src(self):
        if self._src is None:
            self._src = rasterio.open(self.map_path)

    def _read_scaled_patch(self, cx, cy, scale):
        """Читает патч scale×tile_size из карты с центром (cx, cy), ресемплинг в tile_size."""
        ts = self.tile_size
        patch_size = int(ts * scale)
        x1 = int(cx - patch_size / 2)
        y1 = int(cy - patch_size / 2)

        win = Window(x1, y1, patch_size, patch_size)
        try:
            data = self._src.read(window=win)
            arr = np.transpose(data, (1, 2, 0))
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
        except Exception:
            return np.zeros((ts, ts, 3), dtype=np.uint8)

        h, w = arr.shape[:2]
        if h < patch_size or w < patch_size:
            padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
            padded[:h, :w] = arr
            arr = padded

        img = Image.fromarray(arr)
        img = img.resize((ts, ts), Image.BILINEAR)
        return np.array(img)

    def _read_tile(self, x, y):
        """Читает тайл tile_size×tile_size из карты (верхний левый угол)."""
        win = Window(int(x), int(y), self.tile_size, self.tile_size)
        data = self._src.read(window=win)
        arr = np.transpose(data, (1, 2, 0))
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        h, w = arr.shape[:2]
        if h < self.tile_size or w < self.tile_size:
            padded = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
            padded[:h, :w] = arr
            arr = padded
        return arr

    def _normalize(self, arr):
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        out = arr.astype(np.float32) / 255.0
        out = (out - mean) / std
        return torch.from_numpy(out.transpose(2, 0, 1))

    def __getitem__(self, idx):
        self._ensure_src()

        # Позиция P — центр тайла
        px, py = self.coords[idx]
        cx, cy = px + self.tile_size / 2, py + self.tile_size / 2

        # Общий масштаб для пары (один полёт, одна высота)
        scale = random.uniform(*self.scale_range)

        # Anchor: кадр в позиции P
        anchor_tile = self._read_scaled_patch(cx, cy, scale)
        anchor_img = Image.fromarray(anchor_tile)
        anchor_tile = np.array(apply_camera_conditions(anchor_img, level=self.aug_level, skip_scale=True))

        # Positive: кадр в позиции P+Δ (смещение)
        # Случайное направление и длина смещения
        shift = random.uniform(0, self.max_shift_px)
        angle = random.uniform(0, 2 * math.pi)
        dx = shift * math.cos(angle)
        dy = shift * math.sin(angle)
        p_cx = cx + dx
        p_cy = cy + dy

        # Проверка границ: если выходит за пределы — уменьшаем до допустимого
        half = int(self.tile_size * scale / 2)
        margin = 32  # запас от края карты
        if p_cx - half < margin or p_cx + half > self.width - margin or \
           p_cy - half < margin or p_cy + half > self.height - margin:
            # Сдвигаем к центру: сокращаем смещение
            max_dx = min(self.width - margin - half - cx, cx - margin - half)
            max_dy = min(self.height - margin - half - cy, cy - margin - half)
            scale_dx = min(1.0, max_dx / max(1, abs(dx))) if abs(dx) > 0 else 1.0
            scale_dy = min(1.0, max_dy / max(1, abs(dy))) if abs(dy) > 0 else 1.0
            k = min(scale_dx, scale_dy, 1.0)
            p_cx = cx + dx * k
            p_cy = cy + dy * k

        pos_tile = self._read_scaled_patch(p_cx, p_cy, scale)
        pos_img = Image.fromarray(pos_tile)
        pos_tile = np.array(apply_camera_conditions(pos_img, level=self.aug_level, skip_scale=True))

        # Negative: другой тайл (чистый)
        if random.random() < self.hard_neg_prob:
            offset = random.choice([1, 2, 3, -1, -2, -3])
            neg_idx = (idx + offset) % self.num_tiles
        else:
            neg_idx = random.randint(0, self.num_tiles - 1)
            while neg_idx == idx:
                neg_idx = random.randint(0, self.num_tiles - 1)

        nx, ny = self.coords[neg_idx]
        neg_tile = self._read_tile(nx, ny)

        return {
            'anchor': self._normalize(anchor_tile),
            'positive': self._normalize(pos_tile),
            'negative': self._normalize(neg_tile),
        }

    def close(self):
        if self._src:
            self._src.close()


def main():
    # Параметры
    MAP_PATH = '/home/alex/aerial-nav/map_cache/region_google.tif'
    COORDS_PATH = '/home/alex/aerial-nav/training_data/region_dataset/positive_coords.npy'
    INIT_MODEL = 'region_model.pth'
    OUTPUT_PATH = 'dynamics_model.pth'
    EPOCHS = 5
    BATCH_SIZE = 32
    LR = 1e-4
    NUM_WORKERS = 6
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Смещение: 0-33 м при 0.206 м/px (покрывает 1-5 Гц при V_max=120 км/ч)
    MAX_SHIFT_PX = 160  # 33 м
    SCALE_RANGE = (0.25, 2.0)

    print(f"[Train] Device: {DEVICE}")
    print(f"[Train] Batch size: {BATCH_SIZE}")
    print(f"[Train] Workers: {NUM_WORKERS}")
    print(f"[Train] Loss: TripletLoss (margin=1.0)")
    print(f"[Train] LR: {LR} (fine-tune)")
    print(f"[Train] Смещение кадра: 0-{MAX_SHIFT_PX} px "
          f"({MAX_SHIFT_PX*0.206:.1f} м), масштаб {SCALE_RANGE}")

    # Модель
    print(f"[Train] Создание модели (ResNet-18, ImageNet pretrained)")
    model = AerialFeatureExtractor(embedding_dim=256).to(DEVICE)

    # Загружаем веса после curriculum (region_model.pth)
    if os.path.exists(INIT_MODEL):
        print(f"[Train] Загрузка весов: {INIT_MODEL}")
        ckpt = torch.load(INIT_MODEL, map_location=DEVICE, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"[Train] Веса загружены (epoch={ckpt.get('epoch', '?')}, "
                  f"loss={ckpt.get('loss', '?')})")
        else:
            model.load_state_dict(ckpt)
            print(f"[Train] Веса загружены")
    else:
        print(f"[Train] ВНИМАНИЕ: {INIT_MODEL} не найден, обучение с нуля!")

    # Loss и оптимизатор
    criterion = TripletLoss(margin=1.0).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Датасет
    print(f"[Train] Создание датасета динамики...")
    dataset = DynamicsDataset(
        map_path=MAP_PATH,
        coords_path=COORDS_PATH,
        tile_size=512,
        hard_neg_prob=0.5,
        max_shift_px=MAX_SHIFT_PX,
        scale_range=SCALE_RANGE,
        aug_level=2,  # полный набор аугментаций
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    print(f"[Train] Датасет: {len(dataset)} триплетов")
    print(f"[Train] Шагов на эпоху: {len(loader)}")

    # Обучение
    print(f"\n{'='*70}")
    print("ОБУЧЕНИЕ ДИНАМИКЕ (этап 4: скорость/ускорение)")
    print(f"{'='*70}")

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{EPOCHS}')

        for batch in pbar:
            anchor = batch['anchor'].to(DEVICE)
            positive = batch['positive'].to(DEVICE)
            negative = batch['negative'].to(DEVICE)

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss = criterion(emb_a, emb_p, emb_n)

            with torch.no_grad():
                d_pos = torch.nn.functional.pairwise_distance(emb_a, emb_p)
                d_neg = torch.nn.functional.pairwise_distance(emb_a, emb_n)
                correct += (d_pos < d_neg).sum().item()
                total += anchor.size(0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.3f}'
            })

        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        scheduler.step()
        elapsed = time.time() - start_time

        print(f"\n[Epoch {epoch+1}/{EPOCHS}] "
              f"Loss: {avg_loss:.4f} | "
              f"Triplet Acc: {accuracy:.3f} | "
              f"Time: {elapsed:.1f}s")

        # Сохраняем лучшую модель
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'loss': best_loss,
                'embedding_dim': 256,
                'loss_type': 'triplet',
                'margin': 1.0,
                'init_model': INIT_MODEL,
                'task': 'dynamics',
                'max_shift_px': MAX_SHIFT_PX,
            }, OUTPUT_PATH)
            print(f"  ✓ Лучшая модель сохранена: {OUTPUT_PATH}")

    print(f"\n{'='*70}")
    print("ОБУЧЕНИЕ ДИНАМИКЕ ЗАВЕРШЕНО")
    print(f"{'='*70}")
    print(f"  Лучший loss: {best_loss:.4f}")
    print(f"  Финальная модель: {OUTPUT_PATH}")
    print(f"{'='*70}")

    dataset.close()


if __name__ == '__main__':
    main()
