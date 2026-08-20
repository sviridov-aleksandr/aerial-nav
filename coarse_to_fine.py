"""
Coarse-to-Fine навигация.

Архитектура:
  Уровень 1 (Coarse): 2 м/px, tile=448 → грубый поиск зоны 500×500м
  Уровень 2 (Fine):   0.5 м/px, tile=224 → точное позиционирование в зоне

Pipeline:
  1. Извлечь embedding из текущего кадра (fine)
  2. Извлечь embedding из coarse карты → top-K кандидатов
  3. Для каждого кандидата: извлечь fine-тайл → сравнить
  4. Выбрать лучший match
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2

Image.MAX_IMAGE_PIXELS = None

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CoarseToFineNavigator:
    """Двухуровневый навигатор."""

    def __init__(self, fine_model_path):
        from siamese_network import AerialFeatureExtractor

        # Fine модель (0.5 м/px, tile=224)
        self.fine_model = AerialFeatureExtractor(embedding_dim=256).to(DEVICE)
        ckpt = torch.load(fine_model_path, map_location=DEVICE, weights_only=False)
        self.fine_model.load_state_dict(ckpt['model_state_dict'])
        self.fine_model.eval()
        self.fine_tile_size = 224
        self.fine_resolution = 0.5

        # Coarse tile size (4x fine)
        self.coarse_tile_size = self.fine_tile_size * 2  # 448
        self.coarse_resolution = self.fine_resolution * 2  # 1.0 м/px (используем fine модель)

        # Coarse карта
        self.coarse_map = None
        self.fine_map = None
        self.coarse_tiles = None
        self.fine_tiles = None
        self.coarse_locations = []
        self.fine_locations = []
        self.coarse_embeddings = None
        self.fine_embeddings = None

        # Spatial index для быстрого поиска
        self.fine_cx_array = None
        self.fine_cy_array = None

    def _build_spatial_index(self):
        """Строим spatial index для быстрого поиска тайлов."""
        self.fine_cx_array = np.array([loc[0] for loc in self.fine_locations])
        self.fine_cy_array = np.array([loc[1] for loc in self.fine_locations])

    def load_maps(self, coarse_map_path, fine_map_path):
        """Загрузить карты."""
        print("Loading coarse map...")
        self.coarse_map = np.array(Image.open(coarse_map_path).convert('RGB'))
        print(f"  Coarse: {self.coarse_map.shape}")

        print("Loading fine map...")
        self.fine_map = np.array(Image.open(fine_map_path).convert('RGB'))
        print(f"  Fine: {self.fine_map.shape}")

        # Pre-extract coarse tiles
        print("Pre-extracting coarse tiles...")
        self._extract_coarse_tiles()

        # Pre-extract fine tiles
        print("Pre-extracting fine tiles...")
        self._extract_fine_tiles()

    def _extract_coarse_tiles(self):
        """Извлечь тайлы coarse карты."""
        h, w = self.coarse_map.shape[:2]
        tile_size = self.coarse_tile_size
        stride = tile_size // 2
        margin = tile_size // 2 + 50

        self.coarse_tiles = []
        self.coarse_locations = []

        for cy in range(margin, h - margin, stride):
            for cx in range(margin, w - margin, stride):
                tile = self._get_tile(self.coarse_map, cx, cy, tile_size)
                self.coarse_tiles.append(tile)
                self.coarse_locations.append((cx, cy))

        self.coarse_tiles = np.array(self.coarse_tiles)
        N = len(self.coarse_tiles)
        print(f"  Coarse tiles: {N}")

        # Extract embeddings (resize to 224 for fine model)
        all_embs = []
        batch_size = 128
        for i in range(0, N, batch_size):
            batch = self.coarse_tiles[i:i+batch_size]
            # Resize to 224x224 for fine model
            resized = []
            for tile in batch:
                resized.append(cv2.resize(tile, (self.fine_tile_size, self.fine_tile_size),
                                          interpolation=cv2.INTER_AREA))
            resized = np.array(resized)
            tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(DEVICE)
            with torch.no_grad():
                embs = self.fine_model(tensor)
            all_embs.append(embs.cpu())

        self.coarse_embeddings = torch.cat(all_embs, dim=0)
        print(f"  Coarse embeddings: {self.coarse_embeddings.shape}")

    def _extract_fine_tiles(self):
        """Извлечь тайлы fine карты."""
        h, w = self.fine_map.shape[:2]
        tile_size = self.fine_tile_size
        stride = tile_size // 2
        margin = tile_size // 2 + 50

        self.fine_tiles = []
        self.fine_locations = []

        for cy in range(margin, h - margin, stride):
            for cx in range(margin, w - margin, stride):
                tile = self._get_tile(self.fine_map, cx, cy, tile_size)
                self.fine_tiles.append(tile)
                self.fine_locations.append((cx, cy))

        self.fine_tiles = np.array(self.fine_tiles)
        N = len(self.fine_tiles)
        print(f"  Fine tiles: {N}")

        # Extract embeddings
        all_embs = []
        batch_size = 256
        for i in range(0, N, batch_size):
            batch = self.fine_tiles[i:i+batch_size]
            tensor = torch.from_numpy(batch.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(DEVICE)
            with torch.no_grad():
                embs = self.fine_model(tensor)
            all_embs.append(embs.cpu())

        self.fine_embeddings = torch.cat(all_embs, dim=0)
        print(f"  Fine embeddings: {self.fine_embeddings.shape}")

        # Build spatial index
        self._build_spatial_index()

    def _get_tile(self, map_img, cx, cy, size):
        """Извлечь тайл из карты."""
        h, w = map_img.shape[:2]
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
                map_img[my1:my2, mx1:mx2]
        return tile

    def _extract_embedding(self, tile, model):
        """Извлечь embedding из тайла."""
        tile = tile.astype(np.float32) / 255.0
        tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model(tensor)
        return emb.squeeze(0).cpu().numpy()

    def _cosine_similarity(self, a, b):
        """Cosine similarity."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    def navigate(self, query_tile, query_location=None):
        """
        Двухуровневая навигация.

        Args:
            query_tile: тайл текущего кадра (H, W, 3)
            query_location: (cx, cy) в fine-координатах (опционально)

        Returns:
            best_location: (cx, cy) в fine-координатах карты
            confidence: score matching
        """
        # === Уровень 1: Coarse поиск ===
        # Извлекаем coarse embedding из query (ресайм до coarse размера, потом до 224)
        coarse_query = cv2.resize(query_tile, (self.coarse_tile_size, self.coarse_tile_size),
                                  interpolation=cv2.INTER_AREA)
        coarse_query_224 = cv2.resize(coarse_query, (self.fine_tile_size, self.fine_tile_size),
                                      interpolation=cv2.INTER_AREA)
        coarse_query_emb = self._extract_embedding(coarse_query_224, self.fine_model)

        # Cosine similarity с coarse картой
        coarse_query_emb_t = torch.from_numpy(coarse_query_emb).unsqueeze(0).to(DEVICE)
        coarse_sim = self.coarse_embeddings.to(DEVICE) @ coarse_query_emb_t.squeeze(0)
        coarse_sim = coarse_sim.cpu().numpy()

        # Top-K coarse кандидатов
        K_COARSE = 10
        top_k_indices = np.argsort(coarse_sim)[-K_COARSE:][::-1]
        top_k_scores = coarse_sim[top_k_indices]

        print(f"\n[Coarse] Top-{K_COARSE} candidates:")
        for idx, score in zip(top_k_indices, top_k_scores):
            cx, cy = self.coarse_locations[idx]
            print(f"  ({cx:5d}, {cy:5d}) score={score:.4f}")

        # === Уровень 2: Fine поиск в зоне coarse кандидатов ===
        # Определяем зону вокруг каждого coarse кандидата
        # coarse_cx, coarse_cy — в coarse-координатах (4480x4480)
        # fine_cx, fine_cy — в fine-координатах (17920x17920)
        # Масштаб: 4x
        ZONE_RADIUS_FINE = 15  # тайлов в каждую сторону

        best_fine_score = -float('inf')
        best_location = None
        best_coarse_idx = None

        for coarse_idx, coarse_score in zip(top_k_indices, top_k_scores):
            coarse_cx, coarse_cy = self.coarse_locations[coarse_idx]

            # Переводим coarse в fine-координаты (4x масштаб)
            fine_cx = coarse_cx * 4
            fine_cy = coarse_cy * 4

            # Ищем fine-тайлы в зоне
            zone_half = ZONE_RADIUS_FINE * self.fine_tile_size
            zone_start_x = max(0, fine_cx - zone_half)
            zone_end_x = min(self.fine_map.shape[1], fine_cx + zone_half)
            zone_start_y = max(0, fine_cy - zone_half)
            zone_end_y = min(self.fine_map.shape[0], fine_cy + zone_half)

            # Находим индексы fine-тайлов в зоне (используем spatial index)
            x_mask = (self.fine_cx_array >= zone_start_x) & (self.fine_cx_array <= zone_end_x)
            y_mask = (self.fine_cy_array >= zone_start_y) & (self.fine_cy_array <= zone_end_y)
            zone_fine_indices = np.where(x_mask & y_mask)[0].tolist()

            if not zone_fine_indices:
                continue

            # Извлекаем fine-embedding из query
            fine_query_emb = self._extract_embedding(query_tile, self.fine_model)

            # Сравниваем с fine-тайлами в зоне
            zone_embs = self.fine_embeddings[zone_fine_indices].to(DEVICE)  # (M, 256)
            fine_query_emb_t = torch.from_numpy(fine_query_emb).to(DEVICE)  # (256,)
            fine_sim = (zone_embs @ fine_query_emb_t).cpu().numpy()  # (M,)

            # Best в зоне
            best_local_idx = np.argmax(fine_sim)
            best_local_score = fine_sim[best_local_idx]
            best_fine_idx = zone_fine_indices[best_local_idx]
            best_fine_cx, best_fine_cy = self.fine_locations[best_fine_idx]

            combined_score = coarse_score * 0.3 + best_local_score * 0.7

            if best_local_score > best_fine_score:
                best_fine_score = best_local_score
                best_location = (best_fine_cx, best_fine_cy)
                best_coarse_idx = coarse_idx

        if best_location is None:
            # Fallback: лучший coarse match
            best_cx, best_cy = self.coarse_locations[top_k_indices[0]]
            best_location = (best_cx * 4, best_cy * 4)

        return best_location, best_fine_score

    def navigate_batch(self, query_tiles, query_locations=None):
        """
        Навигация для батча тайлов.

        Args:
            query_tiles: (B, H, W, 3)
            query_locations: optional (B, 2)

        Returns:
            locations: (B, 2) — best match locations
            scores: (B,) — confidence scores
        """
        B = query_tiles.shape[0]
        locations = []
        scores = []

        for i in range(B):
            loc, score = self.navigate(query_tiles[i], query_locations[i] if query_locations else None)
            locations.append(loc)
            scores.append(score)

        return np.array(locations), np.array(scores)


def test_coarse_to_fine():
    """Тест coarse-to-fine навигации."""
    print("=" * 60)
    print("COARSE-TO-FINE NAVIGATION TEST")
    print("=" * 60)

    navigator = CoarseToFineNavigator(
        fine_model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'siamese_model_kalanchak_v2.pth')
    )

    # Загружаем карты
    navigator.load_maps(
        coarse_map_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map_cache/coarse_46.2650_33.3732_z16.png'),
        fine_map_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map_cache/highres/highres_46.2650_33.3732_z18.png')
    )

    # Тест: для каждого fine-тайла находим match
    print("\n" + "=" * 60)
    print("TEST: Fine tile matching with coarse-to-fine")
    print("=" * 60)

    # Берём подмножество тайлов для теста
    step = 200
    margin = navigator.fine_tile_size // 2 + 50
    test_tiles = []
    test_locations = []

    for cy in range(margin, navigator.fine_map.shape[0] - margin, step):
        for cx in range(margin, navigator.fine_map.shape[1] - margin, step):
            tile = navigator._get_tile(navigator.fine_map, cx, cy, navigator.fine_tile_size)
            test_tiles.append(tile)
            test_locations.append((cx, cy))

    test_tiles = np.array(test_tiles)
    N = len(test_tiles)
    print(f"  Testing {N} tiles...")

    # Результаты
    errors_m = []
    correct = 0

    for i in range(N):
        cx, cy = test_locations[i]
        tile = test_tiles[i]

        # Coarse-to-fine navigation
        pred_loc, score = navigator.navigate(tile)
        pred_cx, pred_cy = pred_loc

        # Ошибка
        error_px = np.sqrt((cx - pred_cx) ** 2 + (cy - pred_cy) ** 2)
        error_m = error_px * navigator.fine_resolution
        errors_m.append(error_m)

        if error_px < navigator.fine_tile_size // 2:
            correct += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{N}] Current accuracy: {correct}/{i+1} = {correct/(i+1)*100:.1f}%")

    errors_m = np.array(errors_m)
    mean_err = np.mean(errors_m)
    median_err = np.median(errors_m)
    max_err = np.max(errors_m)

    print(f"\n{'='*60}")
    print("COARSE-TO-FINE RESULTS")
    print(f"{'='*60}")
    print(f"  Total tiles: {N}")
    print(f"  Mean error: {mean_err:.1f} м")
    print(f"  Median error: {median_err:.1f} м")
    print(f"  Max error: {max_err:.1f} м")
    print(f"  < 15m: {np.sum(errors_m < 15)/N*100:.1f}%")
    print(f"  < 30m: {np.sum(errors_m < 30)/N*100:.1f}%")
    print(f"  < 56m: {np.sum(errors_m < 56)/N*100:.1f}%")
    print(f"  Match accuracy: {correct}/{N} = {correct/N*100:.1f}%")
    print(f"  Target < 15m: {'✓' if median_err < 15 else '✗'}")

    # Distribution
    print(f"\n  Error distribution:")
    for threshold in [10, 20, 30, 50, 100, 200, 500, 1000]:
        pct = np.sum(errors_m < threshold) / N * 100
        print(f"    < {threshold}m: {pct:.1f}%")


if __name__ == '__main__':
    test_coarse_to_fine()
