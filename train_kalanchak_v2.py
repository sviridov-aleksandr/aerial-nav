"""
Дообучение модели на Каланчаке с улучшенными параметрами.
Цель: решить проблему переобучения на локальную область.

Улучшения:
  1. Большие оффсеты для positive (50-150px)
  2. Hard negatives из близлежащих областей (200-800px)
  3. Сильные аугментации
  4. Cosine embedding loss + triplet loss
  5. Cosine annealing LR
  6. Resume из checkpoint (best или last)
"""

import sys
import os
import argparse
sys.path.insert(0, '/home/alex/aerial-nav')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Логирование в файл
LOG_PATH = '/home/alex/aerial-nav/train_v2.log'
log_file = open(LOG_PATH, 'a')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[GPU] {torch.cuda.get_device_name(0)}")
print(f"[GPU] Память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def log(msg):
    """Логирование в файл и stdout."""
    print(msg)
    log_file.write(str(msg) + '\n')
    log_file.flush()



class ImprovedDataset(Dataset):
    """
    Улучшенный датасет с hard negatives и большими оффсетами.
    """

    def __init__(self, map_paths: list, num_samples: int = 50000,
                 tile_size: int = 224, map_resolution: float = 0.5):
        self.maps = []
        self.tile_size = tile_size
        self.map_resolution = map_resolution
        self.num_samples = num_samples

        for path in map_paths:
            img = np.array(Image.open(path))
            img = img.astype(np.uint8)
            self.maps.append(img)
            name = os.path.basename(path)[:20]
            print(f"  [Map] {name}: {img.shape}, {img.nbytes / 1e6:.0f} MB")

        # Предварительно вычисляем допустимые центры
        self.valid_centers = []
        for map_img in self.maps:
            h, w = map_img.shape[:2]
            margin = tile_size // 2 + 100
            cx_list = list(range(margin, w - margin, 50))
            cy_list = list(range(margin, h - margin, 50))
            self.valid_centers.append((cx_list, cy_list, h, w))

    def __len__(self):
        return self.num_samples

    def _get_map_tile(self, map_img, cx, cy, size):
        h, w = map_img.shape[:2]
        x1 = cx - size // 2
        y1 = cy - size // 2
        x2 = x1 + size
        y2 = y1 + size

        # Clamp к границам карты
        sx1, sy1 = max(0, x1), max(0, y1)
        sx2, sy2 = min(w, x2), min(h, y2)

        tile = np.zeros((size, size, 3), dtype=np.uint8)
        # Копируем только валидную область
        dx1, dy1 = sx1 - x1, sy1 - y1
        dx2, dy2 = dx1 + (sx2 - sx1), dy1 + (sy2 - sy1)
        tile[dy1:dy2, dx1:dx2] = map_img[sy1:sy2, sx1:sx2]
        return tile

    def _augment(self, tile, strength='medium'):
        """Аугментации с настраиваемой силой."""
        tile = tile.astype(np.float32) / 255.0

        if strength == 'strong':
            # Сильные аугментации
            brightness = np.random.uniform(0.5, 1.5)
            contrast = np.random.uniform(0.5, 1.5)
            noise_std = 0.03
            blur_prob = 0.7
        elif strength == 'medium':
            brightness = np.random.uniform(0.7, 1.3)
            contrast = np.random.uniform(0.8, 1.2)
            noise_std = 0.015
            blur_prob = 0.5
        else:
            brightness = np.random.uniform(0.9, 1.1)
            contrast = np.random.uniform(0.95, 1.05)
            noise_std = 0.005
            blur_prob = 0.2

        # Яркость
        tile = np.clip(tile * brightness, 0, 1)

        # Контраст
        mean = tile.mean(axis=(0, 1), keepdims=True)
        tile = np.clip((tile - mean) * contrast + mean, 0, 1)

        # Цветовой сдвиг
        if np.random.random() > 0.4:
            hue_shift = np.random.uniform(-0.1, 0.1)
            hsv = cv2.cvtColor(tile * 255, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift * 180) % 180
            tile = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

        # Шум
        noise = np.random.normal(0, noise_std, tile.shape)
        tile = np.clip(tile + noise, 0, 1)

        # Размытие
        if np.random.random() < blur_prob:
            k = np.random.choice([3, 5, 7])
            tile = cv2.GaussianBlur(tile, (k, k), 0.5)

        return (tile * 255).astype(np.uint8)

    def __getitem__(self, idx):
        map_idx = np.random.randint(0, len(self.maps))
        cx_list, cy_list, h, w = self.valid_centers[map_idx]
        map_img = self.maps[map_idx]

        # Anchor: случайный центр
        cx = np.random.choice(cx_list)
        cy = np.random.choice(cy_list)

        # Anchor: чистый тайл
        anchor = self._get_map_tile(map_img, cx, cy, self.tile_size)

        # Positive: большой оффсет (50-150px)
        offset_range = np.random.randint(50, 151)
        angle = np.random.uniform(0, 2 * np.pi)
        offset_x = int(offset_range * np.cos(angle))
        offset_y = int(offset_range * np.sin(angle))
        px = max(self.tile_size // 2, min(w - self.tile_size // 2, cx + offset_x))
        py = max(self.tile_size // 2, min(h - self.tile_size // 2, cy + offset_y))
        positive = self._get_map_tile(map_img, px, py, self.tile_size)

        # Случайная сила аугментации
        aug_strength = np.random.choice(['strong', 'medium', 'weak'], p=[0.5, 0.3, 0.2])
        positive = self._augment(positive, strength=aug_strength)

        # Hard negative: из близлежащей области (200-800px)
        neg_dist = np.random.randint(200, 801)
        neg_angle = np.random.uniform(0, 2 * np.pi)
        neg_cx = cx + int(neg_dist * np.cos(neg_angle))
        neg_cy = cy + int(neg_dist * np.sin(neg_angle))

        # Если вышли за границы — берём ближайшую допустимую
        if neg_cx < self.tile_size // 2:
            neg_cx = self.tile_size // 2
        elif neg_cx > w - self.tile_size // 2:
            neg_cx = w - self.tile_size // 2
        if neg_cy < self.tile_size // 2:
            neg_cy = self.tile_size // 2
        elif neg_cy > h - self.tile_size // 2:
            neg_cy = h - self.tile_size // 2

        negative = self._get_map_tile(map_img, neg_cx, neg_cy, self.tile_size)
        negative = self._augment(negative, strength=aug_strength)

        a = torch.from_numpy(anchor).permute(2, 0, 1).float()
        p = torch.from_numpy(positive).permute(2, 0, 1).float()
        n = torch.from_numpy(negative).permute(2, 0, 1).float()
        return a, p, n


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, a, p, n):
        pd = torch.sum((a - p) ** 2, dim=1)
        nd = torch.sum((a - n) ** 2, dim=1)
        return torch.clamp(pd - nd + self.margin, min=0.0).mean()


class CosineEmbeddingLoss(nn.Module):
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
        self.criterion = nn.CosineEmbeddingLoss(margin=margin)

    def forward(self, e1, e2, label):
        return self.criterion(e1, e2, label)


def train(resume_from_best=False, resume_from_last=False, start_epoch_override=None):
    from siamese_network import AerialFeatureExtractor

    print("=" * 60)
    print("LOADING MAPS")
    print("=" * 60)

    map_dir = '/home/alex/aerial-nav/map_cache/highres'
    map_paths = [
        os.path.join(map_dir, 'highres_46.2650_33.3732_z18.png'),
    ]
    existing = [p for p in map_paths if os.path.exists(p)]
    log(f"Found {len(existing)} maps")

    print("\n" + "=" * 60)
    print("CREATING IMPROVED DATASET")
    print("=" * 60)

    dataset = ImprovedDataset(
        map_paths=existing,
        num_samples=50000,
        tile_size=224,
        map_resolution=0.5,
    )
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=4,
        pin_memory=True, prefetch_factor=2
    )
    print(f"[Data] Samples: {len(dataset)}, Batch: 16")

    print("\n" + "=" * 60)
    print("LOADING PRETRAINED MODEL")
    print("=" * 60)

    pretrained_path = '/home/alex/aerial-nav/siamese_model_kalanchak.pth'
    checkpoint = torch.load(pretrained_path, map_location=DEVICE, weights_only=False)
    
    model = AerialFeatureExtractor(embedding_dim=256).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded from epoch {checkpoint['epoch']}, loss={checkpoint['loss']:.6f}")
    print(f"  Fine-tuning with improved parameters")

    criterion_triplet = TripletLoss(margin=0.3).to(DEVICE)
    criterion_cosine = CosineEmbeddingLoss(margin=0.5).to(DEVICE)
    
    # Меньший LR для дообучения
    optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    epochs = 40
    best_loss = float('inf')
    output_path = '/home/alex/aerial-nav/siamese_model_kalanchak_v2.pth'
    start_epoch = 0

    # === RESUME ===
    resume_ckpt_path = '/home/alex/aerial-nav/siamese_model_kalanchak_v2.pth'
    
    if resume_from_best:
        # Resume из лучшей модели (сохраняется в siamese_model_kalanchak_v2_best.pth)
        best_path = '/home/alex/aerial-nav/siamese_model_kalanchak_v2_best.pth'
        if os.path.exists(best_path):
            log(f"\n[Resume BEST] Found checkpoint: {best_path}")
            resume_ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
            start_epoch = resume_ckpt['epoch'] + 1
            best_loss = resume_ckpt['loss']
            model.load_state_dict(resume_ckpt['model_state_dict'])
            if 'optimizer_state_dict' in resume_ckpt:
                optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
            log(f"  Resuming from epoch {start_epoch}, best_loss={best_loss:.6f}")
            log(f"  Total epochs to train: {epochs - start_epoch}")
        else:
            log(f"\n[Resume BEST] No best checkpoint found at {best_path}, starting fresh")
    elif resume_from_last:
        # Resume из последнего checkpoint (последняя сохранённая модель)
        if os.path.exists(resume_ckpt_path):
            log(f"\n[Resume LAST] Found checkpoint: {resume_ckpt_path}")
            resume_ckpt = torch.load(resume_ckpt_path, map_location=DEVICE, weights_only=False)
            start_epoch = resume_ckpt['epoch'] + 1
            best_loss = resume_ckpt['loss']
            model.load_state_dict(resume_ckpt['model_state_dict'])
            if 'optimizer_state_dict' in resume_ckpt:
                optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
            log(f"  Resuming from epoch {start_epoch}, best_loss={best_loss:.6f}")
            log(f"  Total epochs to train: {epochs - start_epoch}")
        else:
            log(f"\n[Resume LAST] No checkpoint found, starting fresh")
    else:
        # Проверка: если checkpoint существует, предложить resume
        if os.path.exists(resume_ckpt_path):
            log(f"\n[Info] Existing checkpoint found: {resume_ckpt_path}")
            resume_ckpt = torch.load(resume_ckpt_path, map_location=DEVICE, weights_only=False)
            log(f"  Last saved: epoch {resume_ckpt['epoch']}, loss={resume_ckpt['loss']:.6f}")
            log(f"  To resume, use: --resume-from-last or --resume-from-best")
            log(f"  Starting fresh training from epoch 0")

    if start_epoch_override is not None:
        start_epoch = start_epoch_override
        log(f"\n[Override] Starting from epoch {start_epoch}")

    # Восстановление scheduler при resume
    if start_epoch > 0:
        # Scheduler уже был создан, нужно дойти до нужного шага
        # CosineAnnealingLR: step() вызывается каждый epoch
        for _ in range(start_epoch):
            scheduler.step()
        log(f"  Scheduler restored to epoch {start_epoch}, LR={optimizer.param_groups[0]['lr']:.8f}")

    print(f"\n[Train] Epochs: {epochs}, Batch: 64, LR: 2e-5\n")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        n = 0

        for a, p, neg in dataloader:
            a = a.to(DEVICE, non_blocking=True)
            p = p.to(DEVICE, non_blocking=True)
            neg = neg.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    ae = model(a)
                    pe = model(p)
                    ne = model(neg)
                    
                    t_loss = criterion_triplet(ae, pe, ne)
                    pos_label = torch.ones(ae.size(0), device=DEVICE)
                    neg_label = -torch.ones(ae.size(0), device=DEVICE)
                    c_loss_pos = criterion_cosine(ae, pe, pos_label)
                    c_loss_neg = criterion_cosine(ae, ne, neg_label)
                    
                    loss = t_loss + c_loss_pos + c_loss_neg
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                ae = model(a)
                pe = model(p)
                ne = model(neg)
                t_loss = criterion_triplet(ae, pe, ne)
                pos_label = torch.ones(ae.size(0), device=DEVICE)
                neg_label = -torch.ones(ae.size(0), device=DEVICE)
                c_loss_pos = criterion_cosine(ae, pe, pos_label)
                c_loss_neg = criterion_cosine(ae, ne, neg_label)
                loss = t_loss + c_loss_pos + c_loss_neg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            n += 1

        avg = total_loss / n
        scheduler.step()
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        log(f"Epoch {epoch + 1:3d}/{epochs} | "
            f"Loss: {avg:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if avg < best_loss:
            best_loss = avg
            # Сохраняем лучшую модель
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
                'optimizer_state_dict': optimizer.state_dict(),
                'embedding_dim': 256,
            }, '/home/alex/aerial-nav/siamese_model_kalanchak_v2_best.pth')
            log(f"  Saved BEST (loss={best_loss:.6f})")
            
            # Также обновляем последний checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
                'optimizer_state_dict': optimizer.state_dict(),
                'embedding_dim': 256,
            }, output_path)
            log(f"  Saved LAST (loss={best_loss:.6f})")

    print(f"\nDONE. Best loss: {best_loss:.6f}")
    print(f"Model: {output_path}")
    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Siamese network for Kalanchak')
    parser.add_argument('--resume-from-best', action='store_true',
                        help='Resume from best checkpoint')
    parser.add_argument('--resume-from-last', action='store_true',
                        help='Resume from last checkpoint')
    parser.add_argument('--start-epoch', type=int, default=None,
                        help='Override start epoch')
    args = parser.parse_args()
    
    train(
        resume_from_best=args.resume_from_best,
        resume_from_last=args.resume_from_last,
        start_epoch_override=args.start_epoch
    )

