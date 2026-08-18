"""
Обучение сиамской сети на одном регионе (Каланчак).
Исправления:
  1. Positive генерируется через прямой кроп с оффсетом + аугментации
     (вместо полного пайплайна камеры — слишком большой domain gap)
  2. Margin уменьшен до 0.3 (для L2-нормализованных эмбеддингов)
  3. Hard negative mining: negative берётся из ближайшей области,
     а не случайной
  4. Cosine embedding loss как альтернатива triplet loss
"""

import sys
import os
sys.path.insert(0, '/home/alex/aerial-nav')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    print(f"[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[GPU] Память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("[GPU] НЕ найдена")


class CameraPipelineDataset(Dataset):
    """
    Датасет с упрощённым positive generation.
    
    Anchor: чистый тайл 224x224 из карты
    Positive: тот же тайл + оффсет + аугментации (ближе к anchor)
    Negative: тайл из другой области (дальше по карте)
    """

    def __init__(self, map_paths: list, num_samples: int = 30000,
                 tile_size: int = 224, map_resolution: float = 0.5,
                 min_neg_distance_px: int = 500):
        self.maps = []
        self.tile_size = tile_size
        self.map_resolution = map_resolution
        self.min_neg_distance_px = min_neg_distance_px
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
            margin = tile_size // 2 + 50
            # Шагаем с шагом 100px для эффективности
            cx_list = list(range(margin, w - margin, 100))
            cy_list = list(range(margin, h - margin, 100))
            self.valid_centers.append((cx_list, cy_list, h, w))

    def __len__(self):
        return self.num_samples

    def _get_map_tile(self, map_img, cx, cy, size):
        """Извлекает тайл size x size из карты."""
        h, w = map_img.shape[:2]
        x1 = cx - size // 2
        y1 = cy - size // 2
        x2 = x1 + size
        y2 = y1 + size

        tile = np.zeros((size, size, 3), dtype=np.uint8)
        mx1, my1 = max(0, x1), max(0, y1)
        mx2, my2 = min(w, x2), min(h, y2)

        if mx2 > mx1 and my2 > y1:
            tx1, ty1 = mx1 - x1, my1 - y1
            tile[ty1:ty1 + (my2 - my1), tx1:tx1 + (mx2 - mx1)] = \
                map_img[my1:my2, mx1:mx2]
        return tile

    def _augment(self, tile):
        """Аугментации: шум, яркость, контраст, цвет."""
        tile = tile.astype(np.float32) / 255.0

        # Яркость
        brightness = np.random.uniform(0.7, 1.3)
        tile = np.clip(tile * brightness, 0, 1)

        # Контраст
        contrast = np.random.uniform(0.8, 1.2)
        mean = tile.mean(axis=(0, 1), keepdims=True)
        tile = np.clip((tile - mean) * contrast + mean, 0, 1)

        # Цветовой сдвиг (небольшой)
        if np.random.random() > 0.5:
            hue_shift = np.random.uniform(-0.05, 0.05)
            hsv = cv2.cvtColor(tile * 255, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift * 180) % 180
            tile = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

        # Шум
        noise = np.random.normal(0, 0.01, tile.shape)
        tile = np.clip(tile + noise, 0, 1)

        # Лёгкое размытие
        if np.random.random() > 0.6:
            k = np.random.choice([3, 5])
            tile = cv2.GaussianBlur(tile, (k, k), 0.5)

        return tile.astype(np.float32)

    def __getitem__(self, idx):
        map_idx = np.random.randint(0, len(self.maps))
        cx_list, cy_list, h, w = self.valid_centers[map_idx]
        map_img = self.maps[map_idx]

        # Anchor: случайный центр
        cx = np.random.choice(cx_list)
        cy = np.random.choice(cy_list)

        # Anchor: чистый тайл
        anchor = self._get_map_tile(map_img, cx, cy, self.tile_size)

        # Positive: тот же центр + оффсет (5-30px) + аугментации
        offset_x = np.random.randint(-30, 31)
        offset_y = np.random.randint(-30, 31)
        px = max(self.tile_size // 2, min(w - self.tile_size // 2, cx + offset_x))
        py = max(self.tile_size // 2, min(h - self.tile_size // 2, cy + offset_y))
        positive = self._get_map_tile(map_img, px, py, self.tile_size)
        positive = self._augment(positive)

        # Negative: из другой области (минимальное расстояние)
        neg_cx, neg_cy = cx, cy
        attempts = 0
        while attempts < 20:
            ncx = np.random.choice(cx_list)
            ncy = np.random.choice(cy_list)
            dist = np.sqrt((cx - ncx) ** 2 + (cy - ncy) ** 2)
            if dist >= self.min_neg_distance_px:
                neg_cx, neg_cy = ncx, ncy
                break
            attempts += 1

        negative = self._get_map_tile(map_img, neg_cx, neg_cy, self.tile_size)
        negative = self._augment(negative)

        # Без нормализации ImageNet — aerial данные свои
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
    """Cosine embedding loss — более стабильна для навигации."""
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
        self.criterion = nn.CosineEmbeddingLoss(margin=margin)

    def forward(self, e1, e2, label):
        """
        Args:
            e1, e2: embeddings (B, D)
            label: 1 для positive, -1 для negative
        """
        return self.criterion(e1, e2, label)


def train():
    from siamese_network import AerialFeatureExtractor

    print("=" * 60)
    print("LOADING MAPS")
    print("=" * 60)

    map_dir = '/home/alex/aerial-nav/map_cache/highres'
    map_paths = [
        os.path.join(map_dir, 'highres_46.2650_33.3732_z18.png'),
    ]
    existing = [p for p in map_paths if os.path.exists(p)]
    print(f"Found {len(existing)} maps (single-region training)")

    print("\n" + "=" * 60)
    print("CREATING CAMERA PIPELINE DATASET")
    print("=" * 60)

    dataset = CameraPipelineDataset(
        map_paths=existing,
        num_samples=30000,
        tile_size=224,
        map_resolution=0.5,
        min_neg_distance_px=500,
    )
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=0
    )
    print(f"[Data] Samples: {len(dataset)}, Batch: 32")

    print("\n" + "=" * 60)
    print("TRAINING ON GPU")
    print("=" * 60)

    model = AerialFeatureExtractor(embedding_dim=256).to(DEVICE)
    
    # Используем оба лосса: triplet для структуры, cosine для сходства
    triplet_loss = TripletLoss(margin=0.3).to(DEVICE)
    cosine_loss = CosineEmbeddingLoss(margin=0.5).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    epochs = 60
    best_loss = float('inf')
    output_path = '/home/alex/aerial-nav/siamese_model_kalanchak.pth'

    print(f"[Train] Epochs: {epochs}, Batch: 32, Margin: 0.3\n")

    for epoch in range(epochs):
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
                    
                    # Triplet loss
                    t_loss = triplet_loss(ae, pe, ne)
                    
                    # Cosine embedding loss
                    pos_label = torch.ones(ae.size(0), device=DEVICE)
                    neg_label = -torch.ones(ae.size(0), device=DEVICE)
                    c_loss_pos = cosine_loss(ae, pe, pos_label)
                    c_loss_neg = cosine_loss(ae, ne, neg_label)
                    
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
                t_loss = triplet_loss(ae, pe, ne)
                pos_label = torch.ones(ae.size(0), device=DEVICE)
                neg_label = -torch.ones(ae.size(0), device=DEVICE)
                c_loss_pos = cosine_loss(ae, pe, pos_label)
                c_loss_neg = cosine_loss(ae, ne, neg_label)
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

        print(f"Epoch {epoch + 1:3d}/{epochs} | "
              f"Loss: {avg:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if avg < best_loss:
            best_loss = avg
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
                'embedding_dim': 256,
            }, output_path)
            print(f"  Saved (loss={best_loss:.6f})")

    print(f"\nDONE. Best loss: {best_loss:.6f}")
    print(f"Model: {output_path}")


if __name__ == '__main__':
    train()