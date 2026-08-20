"""
Создание карты низкого разрешения (2 м/px) из оригинальной (0.5 м/px).
Coarse карта используется для грубого поиска зоны (500×500м).
"""

import numpy as np
from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = None  # Убираем лимит PIL

MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map_cache/highres/highres_46.2650_33.3732_z18.png')
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map_cache/coarse_46.2650_33.3732_z16.png')

HIGHRES_RESOLUTION = 0.5  # м/пиксель
COARSE_RESOLUTION = 2.0   # м/пиксель
SCALE_FACTOR = int(COARSE_RESOLUTION / HIGHRES_RESOLUTION)  # 4x

print(f"Loading highres map: {MAP_PATH}")
img = np.array(Image.open(MAP_PATH).convert('RGB'))
h, w = img.shape[:2]
print(f"  Highres: {h}x{w}, resolution={HIGHRES_RESOLUTION} м/px")
print(f"  Physical size: {w*HIGHRES_RESOLUTION/1000:.1f}x{h*HIGHRES_RESOLUTION/1000:.1f} km")

# Resize 4x down
coarse = Image.fromarray(img).resize(
    (w // SCALE_FACTOR, h // SCALE_FACTOR),
    Image.LANCZOS
)
coarse_np = np.array(coarse)

print(f"\nCoarse map: {coarse_np.shape[0]}x{coarse_np.shape[1]}")
print(f"  Physical size: {coarse_np.shape[1]*COARSE_RESOLUTION/1000:.1f}x{coarse_np.shape[0]*COARSE_RESOLUTION/1000:.1f} km")
print(f"  Scale factor: {SCALE_FACTOR}x")

# Save
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
Image.fromarray(coarse_np).save(OUTPUT_PATH)
print(f"\nSaved: {OUTPUT_PATH}")
print(f"  Size: {os.path.getsize(OUTPUT_PATH) / 1e6:.1f} MB")