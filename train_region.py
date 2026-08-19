#!/usr/bin/env python3
"""
train_region.py — обучение сиамской сети на региональной карте.

Curriculum learning, этап 1: обучение на регионе (39×30 км, 70K тайлов).
  - TripletLoss (anchor=кадр камеры, positive=карта, negative=другой тайл)
  - Расширенные аугментации (масштаб, перспектива, туман, облака, шум)
  - ResNet-18 с ImageNet pretrained weights (torchvision)

Использование:
  python3 train_region.py
"""

import os
import sys
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from siamese_network import AerialFeatureExtractor, TripletLoss
from siamese_triplet_dataset import TripletDataset

multiprocessing.set_start_method('fork', force=True)


def main():
    # Параметры
    MAP_PATH = '/home/alex/aerial-nav/map_cache/region_google.tif'
    COORDS_PATH = '/home/alex/aerial-nav/training_data/region_dataset/positive_coords.npy'
    OUTPUT_PATH = 'region_model.pth'
    EPOCHS = 15
    BATCH_SIZE = 32
    LR = 1e-3
    NUM_WORKERS = 6
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"[Train] Device: {DEVICE}")
    print(f"[Train] Batch size: {BATCH_SIZE}")
    print(f"[Train] Workers: {NUM_WORKERS}")
    print(f"[Train] Loss: TripletLoss (margin=1.0)")

    # Модель: torchvision ResNet-18 с ImageNet pretrained weights
    print(f"[Train] Создание модели (ResNet-18, ImageNet pretrained)")
    model = AerialFeatureExtractor(embedding_dim=256).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Параметров: {n_params/1e6:.1f}M")

    # Loss и оптимизатор
    criterion = TripletLoss(margin=1.0).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Датасет
    print(f"[Train] Создание датасета...")
    dataset = TripletDataset(
        map_path=MAP_PATH,
        coords_path=COORDS_PATH,
        tile_size=512,
        hard_neg_prob=0.5,
        aug_level=0,  # стартуем с лёгких аугментаций (curriculum)
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
    print(f"[Train] Curriculum: 3 этапа × 5 эпох")
    print(f"  Этап 1 (эпохи 1-5):  масштаб (scale 0.25-3.5, высоты 100-1200 м)")
    print(f"  Этап 2 (эпохи 6-10): масштаб + поворот (±30°) + лёгкий наклон камеры")
    print(f"  Этап 3 (эпохи 11-15): полный набор (наклон + погода + шум)")

    def curriculum_level(epoch: int) -> int:
        """Этап curriculum по эпохе.
        15 эпох: 0-4 → этап 0 (масштаб), 5-9 → этап 1 (+поворот), 10-14 → этап 2 (полный).
        """
        if epoch < 5:
            return 0
        elif epoch < 10:
            return 1
        else:
            return 2

    # Обучение
    print(f"\n{'='*70}")
    print("ОБУЧЕНИЕ СИАМСКОЙ СЕТИ НА РЕГИОНЕ")
    print(f"{'='*70}")

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()

        # Curriculum: меняем уровень аугментаций по эпохам
        aug_lvl = curriculum_level(epoch)
        dataset.aug_level = aug_lvl
        print(f"\n[Epoch {epoch+1}] Curriculum level: {aug_lvl}")

        total_loss = 0
        # Метрика: доля триплетов, где d(a,p) < d(a,n)
        correct = 0
        total = 0

        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{EPOCHS}')

        for batch in pbar:
            anchor = batch['anchor'].to(DEVICE)
            positive = batch['positive'].to(DEVICE)
            negative = batch['negative'].to(DEVICE)

            # Forward
            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            # Loss
            loss = criterion(emb_a, emb_p, emb_n)

            # Accuracy: доля правильных триплетов
            with torch.no_grad():
                d_pos = torch.nn.functional.pairwise_distance(emb_a, emb_p)
                d_neg = torch.nn.functional.pairwise_distance(emb_a, emb_n)
                correct += (d_pos < d_neg).sum().item()
                total += anchor.size(0)

            # Backward
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
            }, OUTPUT_PATH)
            print(f"  ✓ Лучшая модель сохранена: {OUTPUT_PATH}")

    print(f"\n{'='*70}")
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"{'='*70}")
    print(f"  Лучший loss: {best_loss:.4f}")
    print(f"  Финальная модель: {OUTPUT_PATH}")
    print(f"{'='*70}")

    dataset.close()


if __name__ == '__main__':
    main()