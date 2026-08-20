#!/usr/bin/env python3
"""
train_route.py — fine-tune сиамской сети на маршруте.

Curriculum learning, этап 2: добучение на маршруте (6.2K тайлов).
  - Стартовая модель: region_model.pth (обучена на регионе, 70K тайлов)
  - TripletLoss (anchor=кадр камеры, positive=карта, negative=другой тайл)
  - Малый LR (1e-4) — тонкая настройка, не разрушаем признаки региона
  - Меньше эпох (8) — маршрут маленький, переобучение быстрое

Использование:
  python3 train_route.py
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

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    # Параметры (относительные пути)
    MAP_PATH = os.path.join(PROJECT_DIR, 'map_cache/antiuav_route_strip.tif')
    COORDS_PATH = os.path.join(PROJECT_DIR, 'training_data/route_dataset/positive_coords.npy')
    INIT_MODEL = os.path.join(PROJECT_DIR, 'region_model.pth')
    OUTPUT_PATH = os.path.join(PROJECT_DIR, 'route_model.pth')
    EPOCHS = 8
    BATCH_SIZE = 32
    LR = 1e-4
    NUM_WORKERS = 6
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"[Train] Device: {DEVICE}")
    print(f"[Train] Batch size: {BATCH_SIZE}")
    print(f"[Train] Workers: {NUM_WORKERS}")
    print(f"[Train] Loss: TripletLoss (margin=1.0)")
    print(f"[Train] LR: {LR} (fine-tune)")

    # Модель: torchvision ResNet-18 (ImageNet pretrained)
    print(f"[Train] Создание модели (ResNet-18, ImageNet pretrained)")
    model = AerialFeatureExtractor(embedding_dim=256).to(DEVICE)

    # Загружаем веса региона (этап 1 curriculum learning)
    if os.path.exists(INIT_MODEL):
        print(f"[Train] Загрузка весов региона: {INIT_MODEL}")
        ckpt = torch.load(INIT_MODEL, map_location=DEVICE, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print(f"[Train] Веса региона загружены")
    else:
        print(f"[Train] ВНИМАНИЕ: {INIT_MODEL} не найден, обучение с нуля!")

    # Loss и оптимизатор
    criterion = TripletLoss(margin=1.0).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Датасет (маршрут)
    print(f"[Train] Создание датасета...")
    dataset = TripletDataset(
        map_path=MAP_PATH,
        coords_path=COORDS_PATH,
        tile_size=512,
        hard_neg_prob=0.5,
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
    print("FINE-TUNE НА МАРШРУТЕ (этап 2 curriculum learning)")
    print(f"{'='*70}")

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
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
                'init_model': INIT_MODEL,
            }, OUTPUT_PATH)
            print(f"  ✓ Лучшая модель сохранена: {OUTPUT_PATH}")

    print(f"\n{'='*70}")
    print("FINE-TUNE ЗАВЕРШЁН")
    print(f"{'='*70}")
    print(f"  Лучший loss: {best_loss:.4f}")
    print(f"  Финальная модель: {OUTPUT_PATH}")
    print(f"{'='*70}")

    dataset.close()


if __name__ == '__main__':
    main()
