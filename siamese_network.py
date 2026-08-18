"""
siamese_network.py — сиамская сеть для GPS-denied навигации дрона.

Извлекает инвариантные признаки из кадра камеры и карты.

Backbone: torchvision ResNet-18 (ImageNet pretrained).
  - Правильные residual connections из коробки
  - Гарантированная загрузка ImageNet weights
  - BatchNorm работает корректно при batch >= 32

ВНИМАНИЕ: предыдущая версия использовала кастомный backbone с GroupNorm,
но shortcut (residual) НИКОГДА не вызывался — сеть была обычной глубокой CNN
без остаточных связей, из-за чего обучение не сходилось (Loss ≈ 0.5, Acc ≈ 0.5).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class AerialFeatureExtractor(nn.Module):
    """
    Сиамская сеть для извлечения признаков aerial изображений.
    Backbone: torchvision ResNet-18 (ImageNet pretrained).
    """

    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super().__init__()

        import torchvision
        if pretrained:
            self.backbone = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = torchvision.models.resnet18(weights=None)

        # Убираем последний классификатор, оставляем avgpool → 512 признаков
        self.backbone.fc = nn.Identity()

        # Projection head (LayerNorm — стабилен в train/eval)
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )

        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Извлекает признаки из изображения.

        Args:
            x: изображение (B, 3, H, W)

        Returns:
            нормализованный embedding (B, embedding_dim)
        """
        features = self.backbone(x)           # (B, 512)
        embedding = self.projection(features)  # (B, embedding_dim)

        # L2 нормализация
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss для обучения сиамской сети.
    Минимизирует расстояние между парами "карта-кадр"
    и максимизирует расстояние между негативными парами.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor,
                label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding1: embedding первого изображения
            embedding2: embedding второго изображения
            label: 1 если пара positive, 0 если negative

        Returns:
            loss value
        """
        # Евклидово расстояние
        distance = F.pairwise_distance(embedding1, embedding2)

        # Contrastive loss
        loss_positive = label * (distance ** 2)
        loss_negative = (1 - label) * torch.clamp(self.margin - distance, min=0.0) ** 2

        loss = loss_positive + loss_negative
        return loss.mean()


class TripletLoss(nn.Module):
    """
    Triplet loss для обучения сиамской сети.
    Минимизирует расстояние anchor-positive и максимизирует anchor-negative.

    anchor: кадр камеры (с аугментациями)
    positive: тайл карты (тот же участок)
    negative: другой тайл карты (hard negative)

    L = max(0, d(a,p) - d(a,n) + margin)

    Для L2-нормализованных эмбеддингов максимальное расстояние = 2.0,
    поэтому margin = 1.0 — разумный выбор (старый 0.5 был слишком мал,
    модель "удовлетворялась" случайными эмбеддингами).
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor: embedding кадра камеры (B, D)
            positive: embedding тайла карты (B, D)
            negative: embedding другого тайла (B, D)

        Returns:
            loss value
        """
        d_pos = F.pairwise_distance(anchor, positive)
        d_neg = F.pairwise_distance(anchor, negative)
        loss = F.relu(d_pos - d_neg + self.margin)
        return loss.mean()


class SiameseNavigator:
    """
    Навигатор на основе сиамской сети.
    """

    def __init__(self, embedding_dim: int = 128, use_cuda: bool = False):
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available()
                                   else 'cpu')
        self.model = AerialFeatureExtractor(embedding_dim=embedding_dim).to(self.device)
        self.embedding_dim = embedding_dim

        # Кэш карты для nearest neighbor
        self.map_embeddings = {}
        self.map_locations = {}

    def extract_feature(self, image: np.ndarray) -> torch.Tensor:
        """
        Извлекает признак из изображения.

        Args:
            image: изображение (H, W, 3) RGB

        Returns:
            embedding (embedding_dim,)
        """
        self.model.eval()

        # Предобработка
        img = image.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406]) / np.array([0.229, 0.224, 0.225]))

        # Конвертация в тензор
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor)

        return embedding.squeeze(0).cpu().numpy()

    def register_map_tile(self, tile_id: str, image: np.ndarray,
                          lat: float, lon: float):
        """
        Регистрирует тайл карты.

        Args:
            tile_id: идентификатор тайла
            image: изображение тайла (H, W, 3)
            lat: широта центра
            lon: долгота центра
        """
        embedding = self.extract_feature(image)
        self.map_embeddings[tile_id] = embedding
        self.map_locations[tile_id] = (lat, lon)

    def find_best_match(self, camera_frame: np.ndarray) -> Tuple[str, float]:
        """
        Находит лучший совпадающий тайл карты.

        Args:
            camera_frame: кадр с камеры дрона

        Returns:
            (tile_id, similarity_score)
        """
        if not self.map_embeddings:
            return None, 0.0

        camera_embedding = self.extract_feature(camera_frame)

        best_tile = None
        best_score = -1

        for tile_id, map_embedding in self.map_embeddings.items():
            # Косинусное сходство
            score = np.dot(camera_embedding, map_embedding)

            if score > best_score:
                best_score = score
                best_tile = tile_id

        return best_tile, best_score
