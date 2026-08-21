#!/usr/bin/env python3
"""
train_region.py — обучение сиамской сети на региональной карте.

Обучение по правилу многоуровневых индексов:
  - Каждый триплет привязан к уровню индекса (высоте полёта)
  - positive: патч уровня (512/1024/1792 → resize 512) — эталон индекса
  - anchor: тот же участок + аугментации (поворот, перспектива, сезоность, шум)
  - negative: патч другого участка того же уровня
  - Масштаб заложен в размере патча уровня, не в аугментации

Curriculum: 3 этапа × 5 эпох
  Этап 0 (1-5):   фотометрия + шум
  Этап 1 (6-10):  + поворот ±30° + перспектива + лёгкая сезоность + motion blur
  Этап 2 (11-15): полный набор (погода + сезоность + шум + динамика)

Использование:
  python3 train_region.py
"""

import os
import sys
import argparse
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from siamese_network import AerialFeatureExtractor, TripletLoss
from siamese_triplet_dataset import TripletDataset, LEVELS

multiprocessing.set_start_method('fork', force=True)


def main():
    parser = argparse.ArgumentParser(description="Обучение сиамской сети (многоуровневый индекс)")
    parser.add_argument('--batch', type=int, default=32, help='Размер батча')
    parser.add_argument('--epochs', type=int, default=15, help='Число эпох')
    parser.add_argument('--workers', type=int, default=6, help='Число воркеров')
    parser.add_argument('--lr', type=float, default=1e-3, help='Скорость обучения')
    parser.add_argument('--output', type=str, default='region_model.pth',
                        help='Имя выходной модели')
    parser.add_argument('--resume', type=str, default=None,
                        help='Путь к чекпоинту для продолжения обучения')
    args = parser.parse_args()

    # Параметры (относительные пути — переносимость проекта)
    MAP_PATH = os.path.join(PROJECT_DIR, 'map_cache/region_google.tif')
    COORDS_PATH = os.path.join(PROJECT_DIR, 'training_data/region_dataset/positive_coords.npy')
    OUTPUT_PATH = os.path.join(PROJECT_DIR, args.output)
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch
    LR = args.lr
    NUM_WORKERS = args.workers
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

    # Resume: загрузка чекпоинта
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"[Train] Продолжение обучения: {args.resume}")
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            start_epoch = ckpt.get('epoch', 0)
            print(f"[Train] Загружена эпоха {start_epoch}, loss={ckpt.get('loss', '?')}")
        else:
            model.load_state_dict(ckpt)
            print(f"[Train] Загружены веса (без метаданных эпохи)")
    elif args.resume:
        print(f"[Train] ВНИМАНИЕ: {args.resume} не найден, обучение с нуля")

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
    print(f"  Этап 0 (эпохи 1-5):   фотометрия + шум")
    print(f"  Этап 1 (эпохи 6-10):  + поворот ±30° + перспектива + сезоность + motion blur")
    print(f"  Этап 2 (эпохи 11-15): полный набор (погода + сезоность + динамика)")
    print(f"[Train] Многоуровневый индекс: {len(LEVELS)} уровней (0-1200 м, шаг 100 м)")

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

    for epoch in range(start_epoch, EPOCHS):
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