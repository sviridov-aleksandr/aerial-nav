"""
RANSAC верификация через consistency top-K matchей.

Идея: для каждого query тайла получаем top-K matchей из fine карты.
Если top-K matchи clustered → match reliable.
Если разбросаны → match unreliable (вероятно, ложный).

RANSAC итерации:
  1. Извлечь embedding query
  2. Сравнить со ВСЕМИ fine тайлами → top-K matchей
  3. Найти кластер top-K matchей через RANSAC
  4. Если кластер найден → match accepted, иначе → rejected
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2

Image.MAX_IMAGE_PIXELS = None

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RANSACNavigator:
    """Навигатор с RANSAC верификацией через clustering top-K matchей."""

    def __init__(self, fine_model, fine_embeddings, fine_locations, fine_resolution=0.5,
                 fine_tile_size=224, coarse8_embeddings=None, coarse8_locations=None):
        """
        Args:
            fine_model: fine feature extractor
            fine_embeddings: (N, 256) embeddings fine тайлов
            fine_locations: [(cx, cy), ...] locations fine тайлов
            fine_resolution: м/пиксель
            fine_tile_size: размер тайла
            coarse8_embeddings: (M, 256) embeddings coarse8 тайлов (опционально)
            coarse8_locations: [(cx, cy), ...] locations coarse8 тайлов
        """
        self.fine_model = fine_model
        self.fine_embeddings = fine_embeddings
        self.fine_locations = fine_locations
        self.fine_resolution = fine_resolution
        self.fine_tile_size = fine_tile_size
        self.coarse8_embeddings = coarse8_embeddings
        self.coarse8_locations = coarse8_locations

        # Spatial index
        self.fine_cx_array = None
        self.fine_cy_array = None

    def _build_spatial_index(self, fine_locations):
        """Строим spatial index."""
        self.fine_cx_array = np.array([loc[0] for loc in fine_locations])
        self.fine_cy_array = np.array([loc[1] for loc in fine_locations])

    def _extract_embedding(self, tile, model):
        """Извлечь embedding из тайла."""
        tile = tile.astype(np.float32) / 255.0
        tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model(tensor)
        return emb.squeeze(0).cpu().numpy()

    def _ransac_cluster(self, points, min_cluster_size=3, max_radius_px=30):
        """
        RANSAC clustering: найти наибольший кластер точек.

        Args:
            points: (K, 2) — позиции matchей
            min_cluster_size: минимальный размер кластера
            max_radius_px: максимальный радиус кластера

        Returns:
            cluster_center: (2,) — центр кластера
            cluster_size: int — размер кластера
            inliers: bool — найден ли кластер
        """
        K = len(points)
        if K < min_cluster_size:
            return points[0] if K > 0 else None, 0, False

        best_cluster = None
        best_size = 0

        # RANSAC итерации
        N_ITER = 50
        for _ in range(N_ITER):
            # Random seed point
            seed_idx = np.random.randint(K)
            seed_point = points[seed_idx]

            # Find inliers
            dists = np.sqrt(np.sum((points - seed_point) ** 2, axis=1))
            inliers = dists < max_radius_px
            cluster_size = np.sum(inliers)

            if cluster_size > best_size:
                best_size = cluster_size
                best_cluster = points[inliers]

        if best_size >= min_cluster_size:
            cluster_center = np.mean(best_cluster, axis=0)
            return cluster_center, best_size, True
        else:
            return None, 0, False

    def navigate_ransac(self, query_tile, K=20):
        """
        Навигация с RANSAC верификацией.

        Args:
            query_tile: тайл текущего кадра (H, W, 3)
            K: количество top-K matchей для проверки

        Returns:
            best_location: (cx, cy) в fine-координатах
            confidence: float [0, 1]
            accepted: bool — match принят или отклонён
        """
        # Извлечь embedding query
        query_emb = self._extract_embedding(query_tile, self.fine_model)

        # Сравнить со ВСЕМИ fine тайлами
        query_emb_t = torch.from_numpy(query_emb).unsqueeze(0).to(DEVICE)
        fine_sim = self.fine_embeddings.to(DEVICE) @ query_emb_t.squeeze(0)
        fine_sim = fine_sim.cpu().numpy()

        # Top-K matchей
        top_k_indices = np.argsort(fine_sim)[-K:][::-1]
        top_k_scores = fine_sim[top_k_indices]
        top_k_locs = np.array([self.fine_locations[i] for i in top_k_indices])

        # RANSAC clustering
        cluster_center, cluster_size, accepted = self._ransac_cluster(
            top_k_locs, min_cluster_size=3, max_radius_px=30
        )

        if accepted:
            return cluster_center, float(np.mean(top_k_scores[:cluster_size])), True
        else:
            # Fallback: лучший match
            return self.fine_locations[top_k_indices[0]], float(top_k_scores[0]), False

    def navigate_coarse8_ransac(self, query_tile, K_COARSE8=15, K_FINE=10):
        """
        Coarse8 + RANSAC навигация.

        Args:
            query_tile: тайл текущего кадра (H, W, 3)
            K_COARSE8: количество coarse8 кандидатов
            K_FINE: количество fine matchей для RANSAC

        Returns:
            best_location: (cx, cy) в fine-координатах
            confidence: float
            accepted: bool
        """
        if self.coarse8_embeddings is None:
            return self.navigate_ransac(query_tile, K_FINE)

        # === Coarse8 поиск ===
        coarse8_query = cv2.resize(query_tile, (224, 224), interpolation=cv2.INTER_AREA)
        coarse8_query_224 = cv2.resize(coarse8_query, (224, 224), interpolation=cv2.INTER_AREA)
        coarse8_query_emb = self._extract_embedding(coarse8_query_224, self.fine_model)

        coarse8_query_emb_t = torch.from_numpy(coarse8_query_emb).unsqueeze(0).to(DEVICE)
        coarse8_sim = self.coarse8_embeddings.to(DEVICE) @ coarse8_query_emb_t.squeeze(0)
        coarse8_sim = coarse8_sim.cpu().numpy()

        top_k_coarse8 = np.argsort(coarse8_sim)[-K_COARSE8:][::-1]

        # === Fine поиск в зоне coarse8 кандидатов ===
        all_fine_matches = []
        all_fine_scores = []

        for coarse8_idx in top_k_coarse8:
            coarse8_cx, coarse8_cy = self.coarse8_locations[coarse8_idx]
            fine_cx = coarse8_cx * 16
            fine_cy = coarse8_cy * 16

            zone_half = 20 * self.fine_tile_size
            zone_start_x = max(0, fine_cx - zone_half)
            zone_end_x = min(self.fine_cx_array.max(), fine_cx + zone_half)
            zone_start_y = max(0, fine_cy - zone_half)
            zone_end_y = min(self.fine_cy_array.max(), fine_cy + zone_half)

            x_mask = (self.fine_cx_array >= zone_start_x) & (self.fine_cx_array <= zone_end_x)
            y_mask = (self.fine_cy_array >= zone_start_y) & (self.fine_cy_array <= zone_end_y)
            zone_indices = np.where(x_mask & y_mask)[0]

            if len(zone_indices) == 0:
                continue

            # Извлечь fine embedding
            query_emb = self._extract_embedding(query_tile, self.fine_model)
            query_emb_t = torch.from_numpy(query_emb).to(DEVICE)
            zone_embs = self.fine_embeddings[zone_indices].to(DEVICE)
            fine_sim = (zone_embs @ query_emb_t).cpu().numpy()

            # Top-K в зоне
            top_k_local = np.argsort(fine_sim)[-K_FINE:][::-1]
            for local_idx in top_k_local:
                global_idx = zone_indices[local_idx]
                all_fine_matches.append(self.fine_locations[global_idx])
                all_fine_scores.append(fine_sim[local_idx])

        if not all_fine_matches:
            return None, 0.0, False

        all_fine_matches = np.array(all_fine_matches)
        all_fine_scores = np.array(all_fine_scores)

        # RANSAC clustering
        cluster_center, cluster_size, accepted = self._ransac_cluster(
            all_fine_matches, min_cluster_size=3, max_radius_px=30
        )

        if accepted:
            return cluster_center, float(np.mean(all_fine_scores[:cluster_size])), True
        else:
            # Fallback
            best_idx = np.argmax(all_fine_scores)
            return all_fine_matches[best_idx], float(all_fine_scores[best_idx]), False


def test_ransac():
    """Тест RANSAC навигации."""
    from siamese_network import AerialFeatureExtractor

    print("=" * 60)
    print("RANSAC NAVIGATION TEST")
    print("=" * 60)

    # Загружаем модель
    model = AerialFeatureExtractor(embedding_dim=256).to(DEVICE)
    ckpt = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'siamese_model_kalanchak_v2.pth'),
                      map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Загружаем карты
    print("Loading maps...")
    coarse8_map = np.array(Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map_cache/coarse8_46.2650_33.3732_z14.png')).convert('RGB'))
    fine_map = np.array(Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map_cache/highres/highres_46.2650_33.3732_z18.png')).convert('RGB'))
    print(f"  Coarse8: {coarse8_map.shape}")
    print(f"  Fine: {fine_map.shape}")

    # Извлекаем fine тайлы и embeddings
    print("Extracting fine tiles and embeddings...")
    fine_tile_size = 224
    fine_resolution = 0.5
    stride = fine_tile_size // 2
    margin = fine_tile_size // 2 + 50

    fine_tiles = []
    fine_locations = []

    for cy in range(margin, fine_map.shape[0] - margin, stride):
        for cx in range(margin, fine_map.shape[1] - margin, stride):
            h, w = fine_map.shape[:2]
            x1, y1 = cx - fine_tile_size // 2, cy - fine_tile_size // 2
            x2, y2 = x1 + fine_tile_size, y1 + fine_tile_size
            tile = np.zeros((fine_tile_size, fine_tile_size, 3), dtype=np.uint8)
            mx1, my1 = max(0, x1), max(0, y1)
            mx2, my2 = min(w, x2), min(h, y2)
            if mx2 > mx1 and my2 > my1:
                tx1, ty1 = mx1 - x1, my1 - y1
                tile[ty1:ty1 + (my2 - my1), tx1:tx1 + (mx2 - mx1)] = fine_map[my1:my2, mx1:mx2]
            fine_tiles.append(tile)
            fine_locations.append((cx, cy))

    fine_tiles = np.array(fine_tiles)
    N_FINE = len(fine_tiles)
    print(f"  Fine tiles: {N_FINE}")

    # Extract fine embeddings
    all_embs = []
    batch_size = 256
    for i in range(0, N_FINE, batch_size):
        batch = fine_tiles[i:i+batch_size]
        tensor = torch.from_numpy(batch.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(DEVICE)
        with torch.no_grad():
            embs = model(tensor)
        all_embs.append(embs.cpu())
    fine_embeddings = torch.cat(all_embs, dim=0)
    print(f"  Fine embeddings: {fine_embeddings.shape}")

    # Извлекаем coarse8 тайлы и embeddings
    print("Extracting coarse8 tiles and embeddings...")
    coarse8_tile_size = 224
    coarse8_stride = coarse8_tile_size // 2
    coarse8_margin = coarse8_tile_size // 2 + 50

    coarse8_tiles = []
    coarse8_locations = []

    for cy in range(coarse8_margin, coarse8_map.shape[0] - coarse8_margin, coarse8_stride):
        for cx in range(coarse8_margin, coarse8_map.shape[1] - coarse8_margin, coarse8_stride):
            h, w = coarse8_map.shape[:2]
            x1, y1 = cx - coarse8_tile_size // 2, cy - coarse8_tile_size // 2
            x2, y2 = x1 + coarse8_tile_size, y1 + coarse8_tile_size
            tile = np.zeros((coarse8_tile_size, coarse8_tile_size, 3), dtype=np.uint8)
            mx1, my1 = max(0, x1), max(0, y1)
            mx2, my2 = min(w, x2), min(h, y2)
            if mx2 > mx1 and my2 > my1:
                tx1, ty1 = mx1 - x1, my1 - y1
                tile[ty1:ty1 + (my2 - my1), tx1:tx1 + (mx2 - mx1)] = coarse8_map[my1:my2, mx1:mx2]
            coarse8_tiles.append(tile)
            coarse8_locations.append((cx, cy))

    coarse8_tiles = np.array(coarse8_tiles)
    N_COARSE8 = len(coarse8_tiles)
    print(f"  Coarse8 tiles: {N_COARSE8}")

    # Extract coarse8 embeddings
    all_embs = []
    batch_size = 128
    for i in range(0, N_COARSE8, batch_size):
        batch = coarse8_tiles[i:i+batch_size]
        resized = []
        for tile in batch:
            resized.append(cv2.resize(tile, (224, 224), interpolation=cv2.INTER_AREA))
        resized = np.array(resized)
        tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(DEVICE)
        with torch.no_grad():
            embs = model(tensor)
        all_embs.append(embs.cpu())
    coarse8_embeddings = torch.cat(all_embs, dim=0)
    print(f"  Coarse8 embeddings: {coarse8_embeddings.shape}")

    # Создаём RANSAC навигатор
    ransac_nav = RANSACNavigator(
        fine_model=model,
        fine_embeddings=fine_embeddings,
        fine_locations=fine_locations,
        fine_resolution=fine_resolution,
        fine_tile_size=fine_tile_size,
        coarse8_embeddings=coarse8_embeddings,
        coarse8_locations=coarse8_locations
    )
    ransac_nav._build_spatial_index(fine_locations)

    # Тест
    print("\n" + "=" * 60)
    print("TEST: RANSAC verification on fine tiles")
    print("=" * 60)

    step = 200
    test_tiles = []
    test_locations = []

    for cy in range(margin, fine_map.shape[0] - margin, step):
        for cx in range(margin, fine_map.shape[1] - margin, step):
            h, w = fine_map.shape[:2]
            x1, y1 = cx - fine_tile_size // 2, cy - fine_tile_size // 2
            x2, y2 = x1 + fine_tile_size, y1 + fine_tile_size
            tile = np.zeros((fine_tile_size, fine_tile_size, 3), dtype=np.uint8)
            mx1, my1 = max(0, x1), max(0, y1)
            mx2, my2 = min(w, x2), min(h, y2)
            if mx2 > mx1 and my2 > my1:
                tx1, ty1 = mx1 - x1, my1 - y1
                tile[ty1:ty1 + (my2 - my1), tx1:tx1 + (mx2 - mx1)] = fine_map[my1:my2, mx1:mx2]
            test_tiles.append(tile)
            test_locations.append((cx, cy))

    test_tiles = np.array(test_tiles)
    N = len(test_tiles)
    print(f"  Testing {N} tiles...")

    # Результаты
    errors_m = []
    correct = 0
    accepted_count = 0
    rejected_count = 0

    for i in range(N):
        cx, cy = test_locations[i]
        tile = test_tiles[i]

        # RANSAC navigation
        pred_loc, score, accepted = ransac_nav.navigate_coarse8_ransac(tile, K_COARSE8=15, K_FINE=10)

        if accepted:
            accepted_count += 1
        else:
            rejected_count += 1

        if pred_loc is not None:
            pred_cx, pred_cy = pred_loc
            error_px = np.sqrt((cx - pred_cx) ** 2 + (cy - pred_cy) ** 2)
            error_m = error_px * fine_resolution
            errors_m.append(error_m)

            if error_px < fine_tile_size // 2:
                correct += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{N}] Current accuracy: {correct}/{accepted_count} = {correct/accepted_count*100:.1f}%")

    errors_m = np.array(errors_m)
    mean_err = np.mean(errors_m)
    median_err = np.median(errors_m)
    max_err = np.max(errors_m)

    print(f"\n{'='*60}")
    print("RANSAC RESULTS")
    print(f"{'='*60}")
    print(f"  Total tiles: {N}")
    print(f"  Accepted: {accepted_count} ({accepted_count/N*100:.1f}%)")
    print(f"  Rejected: {rejected_count} ({rejected_count/N*100:.1f}%)")
    print(f"  Mean error (accepted): {mean_err:.1f} м")
    print(f"  Median error (accepted): {median_err:.1f} м")
    print(f"  Max error (accepted): {max_err:.1f} м")
    print(f"  < 15m (accepted): {np.sum(errors_m < 15)/accepted_count*100:.1f}%")
    print(f"  < 30m (accepted): {np.sum(errors_m < 30)/accepted_count*100:.1f}%")
    print(f"  < 56m (accepted): {np.sum(errors_m < 56)/accepted_count*100:.1f}%")
    print(f"  Match accuracy (accepted): {correct}/{accepted_count} = {correct/accepted_count*100:.1f}%")

    # Distribution
    print(f"\n  Error distribution (accepted):")
    for threshold in [10, 20, 30, 50, 100, 200, 500, 1000]:
        pct = np.sum(errors_m < threshold) / accepted_count * 100
        print(f"    < {threshold}m: {pct:.1f}%")

    # Сравнение
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'C2F (8m/px)':<15} {'C2F+RANSAC':<15}")
    print(f"  {'Mean error':<25} {3753.8:<15.1f} {mean_err:<15.1f}")
    print(f"  {'Median error':<25} {3727.1:<15.1f} {median_err:<15.1f}")
    print(f"  {'< 15m':<25} {4.4:<15.1f} {np.sum(errors_m < 15)/accepted_count*100:<15.1f}")
    print(f"  {'< 56m':<25} {8.3:<15.1f} {np.sum(errors_m < 56)/accepted_count*100:<15.1f}")


if __name__ == '__main__':
    test_ransac()