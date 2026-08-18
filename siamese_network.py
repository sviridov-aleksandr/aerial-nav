"""
Сиамская нейронная сеть для GPS-denied навигации дрона.
Извлекает инвариантные признаки из кадра камеры и карты.

Использует GroupNorm/LayerNorm вместо BatchNorm:
- BatchNorm даёт collapse в eval() режиме при обучении с малым batch
  и mixed precision (running statistics расходятся с batch statistics)
- GroupNorm/LayerNorm не зависят от batch size и работают одинаково
  в train() и eval() режимах
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class AerialFeatureExtractor(nn.Module):
    """
    Сиамская сеть для извлечения признаков aerial изображений.
    Базируется на ResNet-18 с модификациями для aerial-данных.
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        
        # ResNet-18 backbone (GroupNorm вместо BatchNorm)
        self.backbone = self._build_resnet18()
        
        # Адаптивный пулинг
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection head (LayerNorm вместо BatchNorm)
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )
        
        self.embedding_dim = embedding_dim

    def _build_resnet18(self) -> nn.Module:
        """Создаёт модифицированный ResNet-18 для aerial-данных."""
        layers = []
        
        # Первый слой — принимает 3 канала (RGB)
        layers.append(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False))
        layers.append(nn.GroupNorm(8, 64))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        # ResNet blocks
        layer1 = self._resnet_block(64, 64, 2, first=True)
        layer2 = self._resnet_block(64, 128, 2, stride=2)
        layer3 = self._resnet_block(128, 256, 2, stride=2)
        layer4 = self._resnet_block(256, 512, 2, stride=2)
        
        layers.extend(layer1)
        layers.extend(layer2)
        layers.extend(layer3)
        layers.extend(layer4)
        
        return nn.Sequential(*layers)

    def _resnet_block(self, in_channels: int, out_channels: int, 
                      num_layers: int, stride: int = 1, first: bool = False) -> list:
        """Создаёт блок ResNet (BasicBlock) с GroupNorm."""
        layers = []
        
        for i in range(num_layers):
            c_in = in_channels if i == 0 else out_channels
            c_out = out_channels
            s = stride if i == 0 else 1
            
            # Conv 1
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, stride=s, padding=1, bias=False))
            layers.append(nn.GroupNorm(8, c_out))
            layers.append(nn.ReLU(inplace=True))
            
            # Conv 2
            layers.append(nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False))
            layers.append(nn.GroupNorm(8, c_out))
            
            # Shortcut (если размеры не совпадают)
            if s != 1 or c_in != c_out:
                layers[-1].add_module('shortcut', nn.Sequential(
                    nn.Conv2d(c_in, c_out, kernel_size=1, stride=s, bias=False),
                    nn.GroupNorm(8, c_out)
                ))
            else:
                layers[-1].add_module('shortcut', nn.Identity())
            
            layers.append(nn.ReLU(inplace=True))
        
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Извлекает признаки из изображения.
        
        Args:
            x: изображение (B, 3, H, W)
        
        Returns:
            нормализованный embedding (B, embedding_dim)
        """
        features = self.backbone(x)
        features = self.avg_pool(features)
        features = features.view(features.size(0), -1)
        embedding = self.projection(features)
        
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