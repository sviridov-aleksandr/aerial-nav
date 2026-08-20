"""
Train Route — финальная версия с оптимизацией для GPU.

Использует multiprocessing.set_start_method('fork') для работы с rasterio.

Использование:
  python3 train_route_final.py
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

# Добавляем проект в путь
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from siamese_network import AerialFeatureExtractor, ContrastiveLoss
from siamese_dataset import SiameseDataset

# ВАЖНО: fork для работы с rasterio в multiprocessing
multiprocessing.set_start_method('fork', force=True)


def main():
    # Параметры (относительные пути)
    MAP_PATH = os.path.join(PROJECT_DIR, 'map_cache/antiuav_route_strip.tif')
    COORDS_PATH = os.path.join(PROJECT_DIR, 'training_data/route_dataset/positive_coords.npy')
    MODEL_PATH = os.path.join(PROJECT_DIR, 'siamese_model_kalanchak_v2.pth')
    OUTPUT_PATH = os.path.join(PROJECT_DIR, 'route_model.pth')
    EPOCHS = 10
    BATCH_SIZE = 16
    LR = 1e-4
    NEG_MULTIPLIER = 3
    NUM_WORKERS = 4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"[Train] Device: {DEVICE}")
    print(f"[Train] Batch size: {BATCH_SIZE}")
    print(f"[Train] Workers: {NUM_WORKERS}")

    # Загрузка модели
    print(f"[Train] Загрузка модели: {MODEL_PATH}")
    model = AerialFeatureExtractor(embedding_dim=256).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    print(f"[Train] Модель загружена")

    # Loss и оптимизатор
    criterion = ContrastiveLoss(margin=1.0).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Датасет
    print(f"[Train] Создание датасета...")
    dataset = SiameseDataset(
        map_path=MAP_PATH,
        coords_path=COORDS_PATH,
        tile_size=512,
        neg_multiplier=NEG_MULTIPLIER,
        augment=True
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    print(f"[Train] Датасет: {len(dataset)} пар")

    # Обучение
    print(f"\n{'='*70}")
    print("ДОУЧЕНИЕ СИАМСКОЙ СЕТИ НА МАРШРУТЕ")
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
            map_batch = batch['map'].to(DEVICE)
            camera_batch = batch['camera'].to(DEVICE)
            label_batch = batch['label'].to(DEVICE)

            # Forward
            map_emb = model(map_batch)
            camera_emb = model(camera_batch)

            # Loss
            loss = criterion(map_emb, camera_emb, label_batch)

            # Accuracy
            with torch.no_grad():
                sim = torch.cosine_similarity(map_emb, camera_emb, dim=1)
                pred = (sim > 0.5).float()
                correct += (pred == label_batch).sum().item()
                total += label_batch.size(0)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.3f}'})

        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        scheduler.step()
        elapsed = time.time() - start_time

        print(f"\n[Epoch {epoch+1}/{EPOCHS}] "
              f"Loss: {avg_loss:.4f} | "
              f"Accuracy: {accuracy:.3f} | "
              f"Time: {elapsed:.1f}s")

        # Сохраняем лучшую модель
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'loss': best_loss,
                'embedding_dim': 256
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
