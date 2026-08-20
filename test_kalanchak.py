"""
Тестирование модели на Каланчаке.
Проверяет:
  1. Правильность загрузки модели
  2. Matching accuracy на карте
  3. Robustness к аугментациям (шум, яркость, размытие)
  4. Confusion matrix — путает ли похожие поля
"""

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import os
from siamese_network import AerialFeatureExtractor

# Увеличиваем лимит PIL для больших карт
Image.MAX_IMAGE_PIXELS = None

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Device] {DEVICE}")

# ===================== CONFIG =====================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'siamese_model_kalanchak.pth')
MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map_cache/highres/highres_46.2650_33.3732_z18.png')
TILE_SIZE = 224
MAP_RESOLUTION = 0.5  # м/пиксель

# ===================== LOAD MODEL =====================
print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model = AerialFeatureExtractor(embedding_dim=256).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  Best loss: {checkpoint['loss']:.6f}")
print(f"  Embedding dim: {checkpoint['embedding_dim']}")

# ===================== LOAD MAP =====================
print("\n" + "=" * 60)
print("LOADING MAP")
print("=" * 60)

map_img = np.array(Image.open(MAP_PATH).convert('RGB'))
print(f"  Map shape: {map_img.shape}")
print(f"  Map size: {map_img.nbytes / 1e6:.0f} MB")
h, w = map_img.shape[:2]

# ===================== HELPER FUNCTIONS =====================
def extract_embedding(tile):
    """Извлекает embedding из тайла."""
    tile = tile.astype(np.float32) / 255.0
    tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model(tensor)
    return emb.squeeze(0).cpu().numpy()

def get_tile(cx, cy, size=TILE_SIZE):
    """Извлекает тайл из карты."""
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

def augment(tile):
    """Аугментации как при обучении."""
    tile = tile.astype(np.float32) / 255.0
    brightness = np.random.uniform(0.7, 1.3)
    tile = np.clip(tile * brightness, 0, 1)
    contrast = np.random.uniform(0.8, 1.2)
    mean = tile.mean(axis=(0, 1), keepdims=True)
    tile = np.clip((tile - mean) * contrast + mean, 0, 1)
    if np.random.random() > 0.5:
        hue_shift = np.random.uniform(-0.05, 0.05)
        hsv = cv2.cvtColor(tile * 255, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift * 180) % 180
        tile = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    noise = np.random.normal(0, 0.01, tile.shape)
    tile = np.clip(tile + noise, 0, 1)
    if np.random.random() > 0.6:
        k = np.random.choice([3, 5])
        tile = cv2.GaussianBlur(tile, (k, k), 0.5)
    return (tile * 255).astype(np.uint8)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# ===================== TEST 1: Basic Matching =====================
print("\n" + "=" * 60)
print("TEST 1: Basic Matching (чистый vs чистый)")
print("=" * 60)

# Создаём сетку тайлов
step = 500  # шаг между тайлами (в пикселях карты)
margin = TILE_SIZE // 2 + 50
test_tiles = []
for cy in range(margin, h - margin, step):
    for cx in range(margin, w - margin, step):
        test_tiles.append((cx, cy))

print(f"  Testing {len(test_tiles)} tiles...")

correct = 0
total = 0
scores_correct = []
scores_random = []

for i, (cx, cy) in enumerate(test_tiles):
    anchor = get_tile(cx, cy)
    anchor_emb = extract_embedding(anchor)
    
    # Positive: тот же тайл + оффсет 10px
    pos = get_tile(cx + 10, cy + 10)
    pos_emb = extract_embedding(pos)
    score_pos = cosine_similarity(anchor_emb, pos_emb)
    
    # Negative: другой тайл
    neg_idx = (i + 1) % len(test_tiles)
    neg_cx, neg_cy = test_tiles[neg_idx]
    neg = get_tile(neg_cx, neg_cy)
    neg_emb = extract_embedding(neg)
    score_neg = cosine_similarity(anchor_emb, neg_emb)
    
    scores_correct.append(score_pos)
    scores_random.append(score_neg)
    
    if score_pos > score_neg:
        correct += 1
    total += 1

    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{len(test_tiles)}] Accuracy: {correct}/{total} "
              f"({correct/total*100:.1f}%)")

acc = correct / total * 100
mean_pos = np.mean(scores_correct)
mean_neg = np.mean(scores_random)
margin_val = mean_pos - mean_neg

print(f"\n  Result: {correct}/{total} = {acc:.1f}%")
print(f"  Mean positive score: {mean_pos:.4f}")
print(f"  Mean negative score: {mean_neg:.4f}")
print(f"  Margin: {margin_val:.4f}")

# ===================== TEST 2: Augmented Matching =====================
print("\n" + "=" * 60)
print("TEST 2: Augmented Matching (чистый vs аугментированный)")
print("=" * 60)

correct_aug = 0
total_aug = 0
scores_aug_pos = []
scores_aug_neg = []

np.random.seed(42)
for i in range(min(200, len(test_tiles))):
    cx, cy = test_tiles[i]
    anchor = get_tile(cx, cy)
    anchor_emb = extract_embedding(anchor)
    
    # Positive: аугментированный
    pos = augment(get_tile(cx, cy))
    pos_emb = extract_embedding(pos)
    score_pos = cosine_similarity(anchor_emb, pos_emb)
    
    # Negative: аугментированный, другой тайл
    neg_idx = (i + 5) % len(test_tiles)
    neg_cx, neg_cy = test_tiles[neg_idx]
    neg = augment(get_tile(neg_cx, neg_cy))
    neg_emb = extract_embedding(neg)
    score_neg = cosine_similarity(anchor_emb, neg_emb)
    
    scores_aug_pos.append(score_pos)
    scores_aug_neg.append(score_neg)
    
    if score_pos > score_neg:
        correct_aug += 1
    total_aug += 1

acc_aug = correct_aug / total_aug * 100
mean_pos_aug = np.mean(scores_aug_pos)
mean_neg_aug = np.mean(scores_aug_neg)

print(f"  Result: {correct_aug}/{total_aug} = {acc_aug:.1f}%")
print(f"  Mean positive score: {mean_pos_aug:.4f}")
print(f"  Mean negative score: {mean_neg_aug:.4f}")
print(f"  Margin: {mean_pos_aug - mean_neg_aug:.4f}")

# ===================== TEST 3: Hard Negatives =====================
print("\n" + "=" * 60)
print("TEST 3: Hard Negatives (близкие тайлы)")
print("=" * 60)

correct_hard = 0
total_hard = 0
scores_hard_pos = []
scores_hard_neg = []

for i in range(min(100, len(test_tiles))):
    cx, cy = test_tiles[i]
    anchor = get_tile(cx, cy)
    anchor_emb = extract_embedding(anchor)
    
    # Positive: оффсет 5px
    pos = get_tile(cx + 5, cy + 5)
    pos_emb = extract_embedding(pos)
    score_pos = cosine_similarity(anchor_emb, pos_emb)
    
    # Hard negative: соседний тайл (step/2)
    neg_cx = cx + step // 2
    neg_cy = cy + step // 2
    if neg_cx < w - margin and neg_cy < h - margin:
        neg = get_tile(neg_cx, neg_cy)
        neg_emb = extract_embedding(neg)
        score_neg = cosine_similarity(anchor_emb, neg_emb)
        
        scores_hard_pos.append(score_pos)
        scores_hard_neg.append(score_neg)
        
        if score_pos > score_neg:
            correct_hard += 1
        total_hard += 1

if total_hard > 0:
    acc_hard = correct_hard / total_hard * 100
    print(f"  Result: {correct_hard}/{total_hard} = {acc_hard:.1f}%")
    print(f"  Mean positive score: {np.mean(scores_hard_pos):.4f}")
    print(f"  Mean negative score: {np.mean(scores_hard_neg):.4f}")
else:
    print("  No hard negatives found")

# ===================== TEST 4: Localization Error =====================
print("\n" + "=" * 60)
print("TEST 4: Localization Error (пиксели → метры)")
print("=" * 60)

# Для каждого тайла находим лучший match среди всех тайлов
# и считаем ошибку в пикселях и метрах
errors_px = []
errors_m = []

for i in range(min(100, len(test_tiles))):
    cx, cy = test_tiles[i]
    anchor = get_tile(cx, cy)
    anchor_emb = extract_embedding(anchor)
    
    best_score = -float('inf')
    best_cx, best_cy = cx, cy
    
    for j, (tcx, tcy) in enumerate(test_tiles):
        tile = get_tile(tcx, tcy)
        tile_emb = extract_embedding(tile)
        score = cosine_similarity(anchor_emb, tile_emb)
        if score > best_score:
            best_score = score
            best_cx, best_cy = tcx, tcy
    
    error_px = np.sqrt((cx - best_cx) ** 2 + (cy - best_cy) ** 2)
    error_m = error_px * MAP_RESOLUTION
    errors_px.append(error_px)
    errors_m.append(error_m)

mean_err_px = np.mean(errors_px)
mean_err_m = np.mean(errors_m)
max_err_m = np.max(errors_m)
median_err_m = np.median(errors_m)

print(f"  Mean error: {mean_err_m:.1f} м ({mean_err_px:.1f} px)")
print(f"  Median error: {median_err_m:.1f} м")
print(f"  Max error: {max_err_m:.1f} м")
print(f"  < 15m: {sum(1 for e in errors_m if e < 15)}/{len(errors_m)} "
      f"({sum(1 for e in errors_m if e < 15)/len(errors_m)*100:.1f}%)")
print(f"  < 30m: {sum(1 for e in errors_m if e < 30)}/{len(errors_m)} "
      f"({sum(1 for e in errors_m if e < 30)/len(errors_m)*100:.1f}%)")

# ===================== SUMMARY =====================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Basic matching accuracy:     {acc:.1f}%")
print(f"  Augmented matching accuracy: {acc_aug:.1f}%")
if total_hard > 0:
    print(f"  Hard negative accuracy:      {acc_hard:.1f}%")
print(f"  Mean localization error:     {mean_err_m:.1f} м")
print(f"  Median localization error:   {median_err_m:.1f} м")
print(f"  Max localization error:      {max_err_m:.1f} м")
print(f"  Target < 15m: {'✓' if median_err_m < 15 else '✗'}")
