"""
Local Matcher — локальный поиск matchей с использованием Siamese embeddings.

Идея: вместо сравнения query со всеми 25000 тайлами карты,
ищем только в окне ±radius_m вокруг предсказанной позиции.

Ускорение: ×1000 (20 тайлов вместо 25000)
"""

import torch
import numpy as np
from PIL import Image
import cv2
import os

Image.MAX_IMAGE_PIXELS = None

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LocalMatcher:
    """Локальный matcher на основе Siamese embeddings."""

    def __init__(self, model_path, map_path, tile_size=224, resolution=0.5):
        """
        Args:
            model_path: путь к checkpoint Siamese модели
            map_path: путь к карте (highres)
            tile_size: размер тайла в пикселях
            resolution: м/пиксель
        """
        from siamese_network import AerialFeatureExtractor

        self.tile_size = tile_size
        self.resolution = resolution
        self.tile_size_m = tile_size * resolution  # 112 м

        # Загружаем модель
        self.model = AerialFeatureExtractor(embedding_dim=256).to(DEVICE)
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        print(f"[LocalMatcher] Model loaded from {model_path}")

        # Загружаем карту
        self.map_path = map_path
        self.map_img = np.array(Image.open(map_path).convert('RGB'))
        h, w = self.map_img.shape[:2]
        print(f"[LocalMatcher] Map: {w}x{h} px, {w*resolution/1000:.1f}x{h*resolution/1000:.1f} km")

        # Извлекаем тайлы и embeddings
        self._extract_tiles_and_embeddings()

    def _extract_tiles_and_embeddings(self):
        """Извлекаем все тайлы карты и их embeddings."""
        h, w = self.map_img.shape[:2]
        stride = self.tile_size // 2
        margin = self.tile_size // 2 + 50

        self.tiles = []
        self.locations = []  # (cx, cy) в пикселях

        for cy in range(margin, h - margin, stride):
            for cx in range(margin, w - margin, stride):
                tile = self._get_tile(cx, cy, self.tile_size)
                self.tiles.append(tile)
                self.locations.append((cx, cy))

        self.tiles = np.array(self.tiles)
        N = len(self.tiles)
        print(f"[LocalMatcher] Extracting {N} tiles...")

        # Извлекаем embeddings батчами
        self.embeddings = []
        batch_size = 256
        for i in range(0, N, batch_size):
            batch = self.tiles[i:i+batch_size]
            tensor = torch.from_numpy(batch.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(DEVICE)
            with torch.no_grad():
                embs = self.model(tensor)
            self.embeddings.append(embs.cpu())

        self.embeddings = torch.cat(self.embeddings, dim=0)
        print(f"[LocalMatcher] Embeddings: {self.embeddings.shape}")

        # Spatial index для быстрого поиска
        self.cx_array = np.array([loc[0] for loc in self.locations])
        self.cy_array = np.array([loc[1] for loc in self.locations])

    def _get_tile(self, cx, cy, size):
        """Извлечь тайл из карты."""
        h, w = self.map_img.shape[:2]
        x1 = cx - size // 2
        y1 = cy - size // 2
        x2 = x1 + size
        y2 = y1 + size

        tile = np.zeros((size, size, 3), dtype=np.uint8)
        mx1, my1 = max(0, x1), max(0, y1)
        mx2, my2 = min(w, x2), min(h, y2)

        if mx2 > mx1 and my2 > my1:
            tx1, ty1 = mx1 - x1, my1 - y1
            tile[ty1:ty1 + (my2 - my1), tx1:tx1 + (mx2 - mx1)] = \
                self.map_img[my1:my2, mx1:mx2]
        return tile

    def _extract_embedding(self, tile):
        """Извлечь embedding из тайла."""
        tile = tile.astype(np.float32) / 255.0
        tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = self.model(tensor)
        return emb.squeeze(0).cpu().numpy()

    def find_tiles_in_window(self, center_cx, center_cy, radius_m):
        """
        Найти индексы тайлов в окне вокруг центра.

        Args:
            center_cx, center_cy: центр в пикселях карты
            radius_m: радиус в метрах

        Returns:
            indices: индексы тайлов в окне
        """
        radius_px = radius_m / self.resolution
        mask = (
            (self.cx_array >= center_cx - radius_px) &
            (self.cx_array <= center_cx + radius_px) &
            (self.cy_array >= center_cy - radius_px) &
            (self.cy_array <= center_cy + radius_px)
        )
        return np.where(mask)[0]

    def match_local(self, query_tile, center_cx, center_cy, radius_m=50):
        """
        Локальный поиск match в окне вокруг предсказанной позиции.

        Args:
            query_tile: тайл текущего кадра (H, W, 3)
            center_cx, center_cy: центр поиска в пикселях карты
            radius_m: радиус поиска в метрах

        Returns:
            best_location: (cx, cy) в пикселях карты
            confidence: cosine similarity [0, 1]
            all_candidates: (N, 2) — все кандидаты для анализа
            all_scores: (N,) — scores кандидатов
        """
        # 1. Извлечь embedding query
        query_emb = self._extract_embedding(query_tile)

        # 2. Найти тайлы в окне
        candidate_indices = self.find_tiles_in_window(center_cx, center_cy, radius_m)
        N = len(candidate_indices)

        if N == 0:
            # Fallback: ближайший тайл к центру
            dists = np.sqrt((self.cx_array - center_cx)**2 + (self.cy_array - center_cy)**2)
            fallback_idx = np.argmin(dists)
            return self.locations[fallback_idx], 0.0, \
                   np.array([self.locations[fallback_idx]]), np.array([0.0])

        # 3. Сравнить embedding query с candidate embeddings
        candidate_embs = self.embeddings[candidate_indices].to(DEVICE)
        query_emb_t = torch.from_numpy(query_emb).to(DEVICE)
        similarities = (candidate_embs @ query_emb_t).cpu().numpy()

        # 4. Вернуть лучший match
        best_idx = np.argmax(similarities)
        best_local_idx = candidate_indices[best_idx]
        best_location = self.locations[best_local_idx]
        best_score = float(similarities[best_idx])

        # Все кандидаты для анализа
        all_candidates = np.array([self.locations[i] for i in candidate_indices])
        all_scores = similarities

        return best_location, best_score, all_candidates, all_scores

    def match_local_batch(self, query_tiles, centers, radius_m=50):
        """
        Пакетный локальный поиск.

        Args:
            query_tiles: (B, H, W, 3)
            centers: (B, 2) — [(cx, cy), ...]
            radius_m: радиус поиска

        Returns:
            locations: (B, 2)
            scores: (B,)
        """
        B = query_tiles.shape[0]
        locations = []
        scores = []

        for i in range(B):
            loc, score, _, _ = self.match_local(
                query_tiles[i], centers[i][0], centers[i][1], radius_m
            )
            locations.append(loc)
            scores.append(score)

        return np.array(locations), np.array(scores)


def test_local_matcher():
    """Тест локального matcher."""
    print("=" * 60)
    print("LOCAL MATCHER TEST")
    print("=" * 60)

    matcher = LocalMatcher(
        model_path='/home/alex/aerial-nav/siamese_model_kalanchak_v2.pth',
        map_path='/home/alex/aerial-nav/map_cache/highres/highres_46.2650_33.3732_z18.png',
        tile_size=224,
        resolution=0.5
    )

    # Тест: для каждого fine-тайла ищем match в окне ±100м
    print("\n" + "=" * 60)
    print("TEST: Local matching with ±100m window")
    print("=" * 60)

    step = 200
    margin = matcher.tile_size // 2 + 50
    test_tiles = []
    test_locations = []

    for cy in range(margin, matcher.map_img.shape[0] - margin, step):
        for cx in range(margin, matcher.map_img.shape[1] - margin, step):
            tile = matcher._get_tile(cx, cy, matcher.tile_size)
            test_tiles.append(tile)
            test_locations.append((cx, cy))

    test_tiles = np.array(test_tiles)
    N = len(test_tiles)
    print(f"  Testing {N} tiles with local matching...")

    # Результаты
    errors_m = []
    correct = 0
    total_candidates = []

    for i in range(N):
        cx, cy = test_locations[i]
        tile = test_tiles[i]

        # Local match
        pred_loc, score, candidates, scores = matcher.match_local(
            tile, cx, cy, radius_m=100
        )
        pred_cx, pred_cy = pred_loc

        # Ошибка
        error_px = np.sqrt((cx - pred_cx) ** 2 + (cy - pred_cy) ** 2)
        error_m = error_px * matcher.resolution
        errors_m.append(error_m)

        if error_px < matcher.tile_size // 2:
            correct += 1

        total_candidates.append(len(candidates))

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{N}] Accuracy: {correct}/{i+1} = {correct/(i+1)*100:.1f}%")

    errors_m = np.array(errors_m)
    mean_err = np.mean(errors_m)
    median_err = np.median(errors_m)
    max_err = np.max(errors_m)
    avg_candidates = np.mean(total_candidates)

    print(f"\n{'='*60}")
    print("LOCAL MATCHER RESULTS")
    print(f"{'='*60}")
    print(f"  Total tiles: {N}")
    print(f"  Avg candidates per query: {avg_candidates:.0f}")
    print(f"  Mean error: {mean_err:.1f} м")
    print(f"  Median error: {median_err:.1f} м")
    print(f"  Max error: {max_err:.1f} м")
    print(f"  < 15m: {np.sum(errors_m < 15)/N*100:.1f}%")
    print(f"  < 30m: {np.sum(errors_m < 30)/N*100:.1f}%")
    print(f"  < 56m: {np.sum(errors_m < 56)/N*100:.1f}%")
    print(f"  Match accuracy: {correct}/{N} = {correct/N*100:.1f}%")
    print(f"  Target < 15m: {'✓' if median_err < 15 else '✗'}")

    # Сравнение с полным поиском
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Full Search':<15} {'Local (±50m)':<15}")
    print(f"  {'Mean error':<20} {3753.8:<15.1f} {mean_err:<15.1f}")
    print(f"  {'Median error':<20} {3727.1:<15.1f} {median_err:<15.1f}")
    print(f"  {'< 15m':<20} {4.4:<15.1f} {np.sum(errors_m < 15)/N*100:<15.1f}")
    print(f"  {'Candidates':<20} {24964:<15.0f} {avg_candidates:<15.0f}")
    print(f"  {'Speedup':<20} {'1x':<15} {'{:.0f}x'.format(24964/avg_candidates):<15}")


if __name__ == '__main__':
    test_local_matcher()
