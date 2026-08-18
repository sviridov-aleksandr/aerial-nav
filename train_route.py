"""
Train Route — дообучение сиамской сети на реальном датасете маршрута.

Использует:
- SiameseDataset (он-лет чтение из GeoTIFF)
- AerialFeatureExtractor (из siamese_network.py)
- ContrastiveLoss

Использование:
  python3 train_route.py \
    --map /home/alex/aerial-nav/map_cache/antiuav_route_strip.tif \
    --coords /home/alex/aerial-nav/training_data/route_dataset/positive_coords.npy \
    --model siamese_model_kalanchak_v2.pth \
    --output route_model.pth \
    --epochs 10 \
    --batch-size 32 \
    --lr 1e-4 \
    --neg-multiplier 3 \
    --num-workers 4
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Добавляем проект в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from siamese_network import AerialFeatureExtractor, ContrastiveLoss
from siamese_dataset import SiameseDataset


class Trainer:
    """Тренер сиамской сети."""

    def __init__(self, model_path: str, output_path: str,
                 map_path: str, coords_path: str,
                 epochs: int = 10, batch_size: int = 32,
                 lr: float = 1e-4, neg_multiplier: int = 3,
                 num_workers: int = 4, device: str = 'cuda'):
        self.model_path = model_path
        self.output_path = output_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print(f"[Trainer] Device: {self.device}")

        # Загружаем модель
        print(f"[Trainer] Загрузка модели: {model_path}")
        self.model = AerialFeatureExtractor(embedding_dim=256).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        self.model.load_state_dict(state_dict)
        print(f"[Trainer] Модель загружена")

        # Loss и оптимизатор
        self.criterion = ContrastiveLoss(margin=1.0).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        # Датасет
        print(f"[Trainer] Создание датасета...")
        self.dataset = SiameseDataset(
            map_path=map_path,
            coords_path=coords_path,
            tile_size=512,
            neg_multiplier=neg_multiplier,
            augment=True
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        print(f"[Trainer] Датасет: {len(self.dataset)} пар")

    def train_epoch(self, epoch: int):
        """Один epoch обучения."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.loader, desc=f'Epoch {epoch+1}/{self.epochs}')

        for batch in pbar:
            map_batch = batch['map'].to(self.device)
            camera_batch = batch['camera'].to(self.device)
            label_batch = batch['label'].to(self.device)

            # Forward
            map_emb = self.model(map_batch)
            camera_emb = self.model(camera_batch)

            # Loss
            loss = self.criterion(map_emb, camera_emb, label_batch)

            # Accuracy (cosine similarity > 0.5 = match)
            with torch.no_grad():
                sim = torch.cosine_similarity(map_emb, camera_emb, dim=1)
                pred = (sim > 0.5).float()
                correct += (pred == label_batch).sum().item()
                total += label_batch.size(0)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.3f}'})

        avg_loss = total_loss / len(self.loader)
        accuracy = correct / total
        self.scheduler.step()

        return avg_loss, accuracy

    def train(self):
        """Полное обучение."""
        print(f"\n{'='*70}")
        print("ДОУЧЕНИЕ СИАМСКОЙ СЕТИ НА МАРШРУТЕ")
        print(f"{'='*70}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  LR: {self.lr}")
        print(f"  Dataset: {len(self.dataset)} пар")
        print(f"{'='*70}\n")

        best_loss = float('inf')

        for epoch in range(self.epochs):
            start_time = time.time()
            avg_loss, accuracy = self.train_epoch(epoch)
            elapsed = time.time() - start_time

            print(f"\n[Epoch {epoch+1}/{self.epochs}] "
                  f"Loss: {avg_loss:.4f} | "
                  f"Accuracy: {accuracy:.3f} | "
                  f"Time: {elapsed:.1f}s")

            # Сохраняем лучшую модель
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch + 1,
                    'loss': best_loss,
                    'embedding_dim': 256
                }, self.output_path)
                print(f"  ✓ Лучшая модель сохранена: {self.output_path}")

        print(f"\n{'='*70}")
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print(f"{'='*70}")
        print(f"  Лучший loss: {best_loss:.4f}")
        print(f"  Финальная модель: {self.output_path}")
        print(f"{'='*70}")

        self.dataset.close()


def main():
    parser = argparse.ArgumentParser(description='Дообучение сиамской сети на маршруте')
    parser.add_argument('--map', type=str, required=True,
                        help='Путь к карте (GeoTIFF)')
    parser.add_argument('--coords', type=str, required=True,
                        help='Путь к координатам тайлов (.npy)')
    parser.add_argument('--model', type=str, default='siamese_model_kalanchak_v2.pth',
                        help='Путь к начальной модели')
    parser.add_argument('--output', type=str, default='route_model.pth',
                        help='Путь сохранения модели')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Количество эпох (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--neg-multiplier', type=int, default=3,
                        help='Negative multiplier (default: 3)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Количество workers (default: 4)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda или cpu (default: cuda)')

    args = parser.parse_args()

    trainer = Trainer(
        model_path=args.model,
        output_path=args.output,
        map_path=args.map,
        coords_path=args.coords,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        neg_multiplier=args.neg_multiplier,
        num_workers=args.num_workers,
        device=args.device
    )

    trainer.train()


if __name__ == '__main__':
    main()