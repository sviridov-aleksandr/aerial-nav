"""
Тест навигации через симулятор камеры.
Полный пайплайн: симулятор полёта → камера → модель → match → ошибка.
"""

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import cv2
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from siamese_network import AerialFeatureExtractor
from simulator_fixed_wing import FlightSimulatorFixedWing

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Device] {DEVICE}")

# ===================== CONFIG =====================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'siamese_model_kalanchak.pth')
MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map_cache/highres/highres_46.2650_33.3732_z18.png')
TILE_SIZE = 224
MAP_RESOLUTION = 0.5  # м/пиксель
STRIDE = 112  # шаг между тайлами (112px = 56м)

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

# ===================== LOAD MAP =====================
print("\n" + "=" * 60)
print("LOADING MAP")
print("=" * 60)

map_img = np.array(Image.open(MAP_PATH).convert('RGB'))
print(f"  Map shape: {map_img.shape}")
h, w = map_img.shape[:2]

# ===================== REGISTER MAP TILES =====================
print("\n" + "=" * 60)
print("REGISTERING MAP TILES")
print("=" * 60)

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

def extract_embedding(tile):
    """Извлекает embedding из тайла."""
    tile = tile.astype(np.float32) / 255.0
    tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model(tensor)
    return emb.squeeze(0).cpu().numpy()

# Создаём сетку тайлов для регистрации
map_embeddings = {}
map_locations = {}
margin = TILE_SIZE // 2 + 50

tile_count = 0
for cy in range(margin, h - margin, STRIDE):
    for cx in range(margin, w - margin, STRIDE):
        tile = get_tile(cx, cy)
        emb = extract_embedding(tile)
        map_embeddings[(cx, cy)] = emb
        # Конвертируем пиксели карты в метры
        map_locations[(cx, cy)] = (cx * MAP_RESOLUTION, cy * MAP_RESOLUTION)
        tile_count += 1

print(f"  Registered {tile_count} tiles")
print(f"  Grid: {w // STRIDE}×{h // STRIDE} (stride={STRIDE}px={STRIDE*MAP_RESOLUTION}m)")

# ===================== CAMERA PIPELINE =====================
def simulate_camera_frame(map_img, cx, cy, altitude, fov=90.0,
                          cam_w=1920, cam_h=1080, resolution=0.5):
    """
    Полный пайплайн камеры:
    1. Вычислить покрытие на заданной высоте
    2. Извлечь большой регион из карты
    3. Даунскейл до разрешения камеры
    4. Кроп центра (соответствует одному тайлу)
    5. Ресайз до 224x224
    6. Camera effects
    """
    coverage_m = altitude * np.tan(np.radians(fov / 2)) * 2
    coverage_px = int(coverage_m / resolution)
    coverage_px = min(coverage_px, map_img.shape[0], map_img.shape[1])

    # 1. Извлечь большой регион
    region = get_tile(cx, cy, coverage_px)

    # 2. Даунскейл до разрешения камеры
    region = cv2.resize(region, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR)

    # 3. Кроп центра
    gsd = coverage_m / cam_w  # м/пиксель в кадре
    tile_in_frame = int(TILE_SIZE * resolution / gsd)
    tile_in_frame = max(1, min(tile_in_frame, cam_w, cam_h))

    region_h, region_w = region.shape[:2]
    y1 = (region_h - tile_in_frame) // 2
    x1 = (region_w - tile_in_frame) // 2
    crop = region[y1:y1 + tile_in_frame, x1:x1 + tile_in_frame]

    # 4. Ресайз до 224x224
    crop = cv2.resize(crop, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_LINEAR)

    # 5. Camera effects
    crop = crop.astype(np.float32) / 255.0
    crop = np.clip(crop + np.random.normal(0, 0.015, crop.shape), 0, 1)
    crop = np.clip(crop * np.random.uniform(0.85, 1.15), 0, 1)
    if np.random.random() > 0.5:
        crop = cv2.GaussianBlur(crop, (3, 3), 0.5)

    return (crop * 255).astype(np.uint8)

# ===================== NAVIGATION TEST =====================
print("\n" + "=" * 60)
print("NAVIGATION TEST VIA CAMERA SIMULATOR")
print("=" * 60)

# Создаём симулятор
sim = FlightSimulatorFixedWing(
    map_image=map_img,
    resolution=MAP_RESOLUTION,
    fov=90.0,
    width=1920,
    height=1080,
    min_turn_radius=100.0,
    cruise_speed=22.0,
    min_altitude=700.0,
    max_altitude=1000.0
)

# Старт в центре карты
center_x = w * 0.5 * MAP_RESOLUTION
center_y = h * 0.5 * MAP_RESOLUTION
sim.set_position(center_x, center_y, 850.0)

# Waypoints: квадрат 2×2 км
waypoints = [
    (center_x + 1000, center_y, 700.0),
    (center_x + 1000, center_y + 1000, 1000.0),
    (center_x - 1000, center_y + 1000, 850.0),
    (center_x - 1000, center_y - 1000, 700.0),
    (center_x, center_y, 850.0),
]

print(f"  Start: ({center_x:.0f}, {center_y:.0f}) м, alt={850} м")
print(f"  Waypoints: {len(waypoints)} (квадрат 2×2 км)")
print(f"  Duration: 600 кадров (20 сек)")

# Тестирование
errors_m = []
errors_px = []
correct_matches = 0
total_matches = 0
wp_idx = 0

print("\n" + "-" * 70)
print(f"{'Frame':>6} | {'GT_Pos':>18} | {'Est_Pos':>18} | {'Error':>8} | {'WP':>3}")
print("-" * 70)

for frame_idx in range(600):
    # Наведение на waypoint
    if wp_idx < len(waypoints):
        wx, wy, wz = waypoints[wp_idx]
        sim.follow_waypoint(wx, wy)
        sim.set_target_altitude(wz)
        dist = np.sqrt((wx - sim.drone_x)**2 + (wy - sim.drone_y)**2)
        if dist < 50:
            wp_idx += 1

    sim.update(dt=1/30.0)
    gt = sim.get_ground_truth()

    # Кадр с камеры (позиция дрона в метрах → пиксели карты)
    drone_cx = gt['x'] / MAP_RESOLUTION
    drone_cy = gt['y'] / MAP_RESOLUTION
    altitude = gt['z']

    # Симулируем кадр камеры
    camera_frame = simulate_camera_frame(
        map_img, int(drone_cx), int(drone_cy), altitude
    )

    # Извлекаем embedding из кадра
    camera_emb = extract_embedding(camera_frame)

    # Находим лучший match среди зарегистрированных тайлов
    best_score = -float('inf')
    best_tile = None
    for (tcx, tcy), emb in map_embeddings.items():
        score = np.dot(camera_emb, emb)
        if score > best_score:
            best_score = score
            best_tile = (tcx, tcy)

    if best_tile is not None:
        est_cx, est_cy = best_tile
        est_x = est_cx * MAP_RESOLUTION
        est_y = est_cy * MAP_RESOLUTION

        error_px = np.sqrt((drone_cx - est_cx)**2 + (drone_cy - est_cy)**2)
        error_m = error_px * MAP_RESOLUTION
        errors_m.append(error_m)
        errors_px.append(error_px)

        # Считаем правильным, если ошибка < 1 тайла (112px = 56м)
        if error_px < STRIDE:
            correct_matches += 1
        total_matches += 1

    if (frame_idx + 1) % 100 == 0:
        mean_err = np.mean(errors_m) if errors_m else 0
        print(f"{frame_idx + 1:6d} | ({gt['x']:7.0f}, {gt['y']:7.0f}) | "
              f"({est_x:7.0f}, {est_y:7.0f}) | {mean_err:7.1f}м | {wp_idx:3d}")

print("-" * 70)

# ===================== RESULTS =====================
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

if errors_m:
    mean_err = np.mean(errors_m)
    median_err = np.median(errors_m)
    max_err = np.max(errors_m)
    std_err = np.std(errors_m)

    under_15m = sum(1 for e in errors_m if e < 15) / len(errors_m) * 100
    under_30m = sum(1 for e in errors_m if e < 30) / len(errors_m) * 100
    under_56m = sum(1 for e in errors_m if e < 56) / len(errors_m) * 100

    print(f"  Total frames: {total_matches}")
    print(f"  Mean error: {mean_err:.1f} м")
    print(f"  Median error: {median_err:.1f} м")
    print(f"  Max error: {max_err:.1f} м")
    print(f"  Std error: {std_err:.1f} м")
    print(f"  < 15m: {under_15m:.1f}%")
    print(f"  < 30m: {under_30m:.1f}%")
    print(f"  < 56m (1 tile): {under_56m:.1f}%")
    print(f"  Match accuracy (< 56m): {correct_matches}/{total_matches} = {correct_matches/total_matches*100:.1f}%")
    print(f"\n  Target < 15m: {'✓' if median_err < 15 else '✗'}")
else:
    print("  No matches found!")