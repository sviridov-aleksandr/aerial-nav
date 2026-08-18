"""
Тест модели через batch processing.
Оптимизировано: все тайлы обрабатываются батчами, сравнение через матричное умножение.
"""

import sys
sys.path.insert(0, '/home/alex/aerial-nav')

import torch
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from siamese_network import AerialFeatureExtractor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = '/home/alex/aerial-nav/siamese_model_kalanchak_v2.pth'
MAP_PATH = '/home/alex/aerial-nav/map_cache/highres/highres_46.2650_33.3732_z18.png'
TILE_SIZE = 224
MAP_RESOLUTION = 0.5
STRIDE = 112

# Load model
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model = AerialFeatureExtractor(embedding_dim=256).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load map
map_img = np.array(Image.open(MAP_PATH).convert('RGB'))
h, w = map_img.shape[:2]

def get_tile(cx, cy, size=TILE_SIZE):
    x1, y1 = cx - size // 2, cy - size // 2
    x2, y2 = x1 + size, y1 + size
    tile = np.zeros((size, size, 3), dtype=np.uint8)
    mx1, my1 = max(0, x1), max(0, y1)
    mx2, my2 = min(w, x2), min(h, y2)
    if mx2 > mx1 and my2 > y1:
        tx1, ty1 = mx1 - x1, my1 - y1
        tile[ty1:ty1 + (my2 - my1), tx1:tx1 + (mx2 - mx1)] = map_img[my1:my2, mx1:mx2]
    return tile

# Collect all tiles
print("Collecting tiles...")
tiles = []
locations = []
margin = TILE_SIZE // 2 + 50
for cy in range(margin, h - margin, STRIDE):
    for cx in range(margin, w - margin, STRIDE):
        tiles.append(get_tile(cx, cy))
        locations.append((cx, cy))

tiles_np = np.array(tiles)  # (N, 224, 224, 3)
N = len(tiles)
print(f"  Total tiles: {N}")

# Extract embeddings in batches
print("Extracting embeddings (batch=256)...")
all_embeddings = []
batch_size = 256
for i in range(0, N, batch_size):
    batch = tiles_np[i:i+batch_size]
    batch_tensor = torch.from_numpy(batch.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(DEVICE)
    with torch.no_grad():
        embs = model(batch_tensor)
    all_embeddings.append(embs.cpu())

all_embs = torch.cat(all_embeddings, dim=0)  # (N, 256)
print(f"  Embeddings shape: {all_embs.shape}")

# Find best match for each tile (matrix multiplication)
print("Finding best matches (matrix mult)...")
# Cosine similarity (embeddings already L2-normalized)
similarity = all_embs @ all_embs.T  # (N, N) — это O(N²) но через GPU

# For each tile, find the best match (excluding self)
errors_m = []
errors_px = []
correct = 0

for i in range(N):
    sim_row = similarity[i]
    # Set self-similarity to -inf
    sim_row[i] = -1.0
    best_j = torch.argmax(sim_row).item()
    
    cx, cy = locations[i]
    best_cx, best_cy = locations[best_j]
    
    error_px = np.sqrt((cx - best_cx)**2 + (cy - best_cy)**2)
    error_m = error_px * MAP_RESOLUTION
    errors_m.append(error_m)
    
    if error_px < STRIDE:
        correct += 1

# Results
errors_m = np.array(errors_m)
mean_err = np.mean(errors_m)
median_err = np.median(errors_m)
max_err = np.max(errors_m)
std_err = np.std(errors_m)

print(f"\n{'='*60}")
print("DIRECT TILE MATCHING RESULTS (batch optimized)")
print(f"{'='*60}")
print(f"  Total tiles: {N}")
print(f"  Mean error: {mean_err:.1f} м")
print(f"  Median error: {median_err:.1f} м")
print(f"  Max error: {max_err:.1f} м")
print(f"  Std error: {std_err:.1f} м")
print(f"  < 15m: {np.sum(errors_m < 15)/N*100:.1f}%")
print(f"  < 30m: {np.sum(errors_m < 30)/N*100:.1f}%")
print(f"  < 56m (1 tile): {np.sum(errors_m < 56)/N*100:.1f}%")
print(f"  Match accuracy: {correct}/{N} = {correct/N*100:.1f}%")
print(f"  Target < 15m: {'✓' if median_err < 15 else '✗'}")

# Distribution
print(f"\n  Error distribution:")
for threshold in [10, 20, 30, 50, 100, 200, 500, 1000]:
    pct = np.sum(errors_m < threshold) / N * 100
    print(f"    < {threshold}m: {pct:.1f}%")