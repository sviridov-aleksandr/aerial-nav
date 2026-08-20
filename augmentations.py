"""
augmentations.py — аугментации для сиамской сети на основе многоуровневых индексов.

Все воздействия привязаны к уровню индекса (высоте полёта):
  - Высота → выбор уровня (масштаб заложен в размере патча, не в аугментации)
  - Поворот → рыскание дрона
  - Наклон камеры → perspective warp
  - Скорость/смещение → сдвиг центра кадра (имитация движения между кадрами)
  - Пропуск кадров → лёгкое размытие движения (motion blur)
  - Шум Калмана → небольшой случайный сдвиг (имитация неточности EKF)
  - Сезонность → снег/листва/выгорание/дождь (цветовые преобразования)
  - Шум матрицы → гауссов + JPEG-сжатие

CURRICULUM-АУГМЕНТАЦИИ (поэтапное усиление):
  level 0: только фотометрия + лёгкий шум (эпохи 1-5)
  level 1: + поворот ±30°, лёгкая перспектива, лёгкая сезоность (эпохи 6-10)
  level 2: полный набор — все искажения на полной мощности (эпохи 11-15)
"""

import numpy as np
import random
import math
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageOps
from io import BytesIO


# ──────────────────────────────────────────────────────────────────────
# Геометрические аугментации
# ──────────────────────────────────────────────────────────────────────

def apply_perspective_warp(img: Image.Image, max_shift: float = 0.12) -> Image.Image:
    """
    Перспективное искажение — имитация наклона камеры в полёте.
    max_shift: максимальное смещение углов (доля от размера).
    """
    w, h = img.size
    shift_x = w * random.uniform(0, max_shift) * random.choice([-1, 1])
    shift_y = h * random.uniform(0, max_shift) * random.choice([-1, 1])

    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    if random.random() < 0.5:
        new_corners = [
            (shift_x, shift_y), (w + shift_x, shift_y),
            (w, h), (0, h)
        ]
    else:
        new_corners = [
            (0, 0), (w + shift_x, shift_y),
            (w + shift_x, h + shift_y), (0, h)
        ]

    img = img.transform(
        (w, h),
        Image.PERSPECTIVE,
        _quad_to_perspective(corners, new_corners),
        Image.BILINEAR
    )
    return img


def _quad_to_perspective(src_quad, dst_quad):
    """Рассчитывает матрицу перспективного преобразования."""
    src = np.array(src_quad, dtype=np.float64)
    dst = np.array(dst_quad, dtype=np.float64)

    A = []
    B = []
    for (sx, sy), (dx, dy) in zip(src, dst):
        A.append([sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy])
        A.append([0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy])
        B.append(dx)
        B.append(dy)

    A = np.array(A)
    B = np.array(B)
    coeffs = np.linalg.solve(A, B)
    return coeffs.tolist()


def apply_small_rotation(img: Image.Image, max_angle: float = 15.0) -> Image.Image:
    """
    Поворот на небольшой угол (рыскание дрона).
    Без чёрных краёв: reflect-padding → поворот → вырезка центра.
    """
    angle = random.uniform(-max_angle, max_angle)
    w, h = img.size
    diag = int(math.ceil(math.sqrt(w**2 + h**2)))
    pad = (diag - min(w, h)) // 2 + 16
    arr = np.array(img)
    arr = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    img = Image.fromarray(arr)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
    cx, cy = img.size[0] // 2, img.size[1] // 2
    img = img.crop((cx - w // 2, cy - h // 2, cx + w - w // 2, cy + h - h // 2))
    return img


def apply_motion_blur(img: Image.Image, max_pixels: int = 15) -> Image.Image:
    """
    Размытие движения — имитация скорости полёта / пропуска кадров.
    При 33 м/с и экспозиции 1/100 с смещение ≈ 0.33 м ≈ 1.6 px на карте.
    При пропуске кадра (1 Гц) смещение ≈ 33 м ≈ 160 px.
    """
    pixels = random.uniform(0, max_pixels)
    if pixels < 1:
        return img

    angle = random.uniform(0, 360)
    kernel_size = int(pixels * 2) + 1
    if kernel_size < 3:
        return img

    # Создаём ядро motion blur
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    rad = math.radians(angle)
    dx, dy = math.cos(rad), math.sin(rad)
    for i in range(kernel_size):
        offset = i - center
        x = int(center + offset * dx)
        y = int(center + offset * dy)
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0
    kernel /= kernel.sum()

    arr = np.array(img).astype(np.float32)
    # Применяем через свёртку по каждому каналу
    from scipy.ndimage import convolve
    for c in range(arr.shape[2]):
        arr[:, :, c] = convolve(arr[:, :, c], kernel, mode='reflect')
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ──────────────────────────────────────────────────────────────────────
# Погодные аугментации
# ──────────────────────────────────────────────────────────────────────

def apply_clouds(img: Image.Image, max_coverage: float = 0.4) -> Image.Image:
    """Облачность — полупрозрачные белые пятна."""
    w, h = img.size
    img = img.convert('RGBA')
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    num_clouds = random.randint(1, 6)
    for _ in range(num_clouds):
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        rx = random.randint(w // 12, w // 4)
        ry = random.randint(h // 16, h // 6)
        alpha = random.randint(30, 160)
        for i in range(3):
            ox = random.randint(-rx // 2, rx // 2)
            oy = random.randint(-ry // 2, ry // 2)
            draw.ellipse([cx + ox - rx, cy + oy - ry,
                          cx + ox + rx, cy + oy + ry],
                         fill=(255, 255, 255, alpha))

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=random.uniform(5, 15)))
    img = Image.alpha_composite(img, overlay)
    return img.convert('RGB')


def apply_fog(img: Image.Image, max_intensity: float = 0.5) -> Image.Image:
    """Туман/дымка — равномерный серый оверлей."""
    intensity = random.uniform(0, max_intensity)
    if intensity < 0.05:
        return img

    fog_color = random.randint(200, 245)
    img = img.convert('RGBA')
    fog = Image.new('RGBA', img.size,
                    (fog_color, fog_color, fog_color,
                     int(intensity * 255)))
    img = Image.alpha_composite(img, fog)

    if intensity > 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=intensity * 3))
    return img.convert('RGB')


def apply_shadow_gradient(img: Image.Image) -> Image.Image:
    """Направленный градиент тени — имитация времени суток/облаков."""
    w, h = img.size
    direction = random.choice(['left', 'right', 'top', 'bottom'])
    strength = random.uniform(0.05, 0.3)

    img = img.convert('RGBA')
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if direction in ('left', 'right'):
        for x in range(0, w, 8):
            t = x / max(1, w - 1)
            alpha = int(strength * 255 * (1 - t)) if direction == 'left' else int(strength * 255 * t)
            draw.line([(x, 0), (x, h)], fill=(0, 0, 0, alpha))
    else:
        for y in range(0, h, 8):
            t = y / max(1, h - 1)
            alpha = int(strength * 255 * (1 - t)) if direction == 'top' else int(strength * 255 * t)
            draw.line([(0, y), (w, y)], fill=(0, 0, 0, alpha))

    img = Image.alpha_composite(img, overlay)
    return img.convert('RGB')


# ──────────────────────────────────────────────────────────────────────
# Сезонные аугментации
# ──────────────────────────────────────────────────────────────────────

def apply_winter(img: Image.Image, intensity: float = 0.5) -> Image.Image:
    """
    Зима: снег (белый оверлей) + снижение насыщенности.
    intensity: 0-1, сила эффекта.
    """
    # Снижение насыщенности
    sat_factor = 1.0 - intensity * 0.6
    img = ImageEnhance.Color(img).enhance(sat_factor)

    # Повышение яркости (снег отражает свет)
    img = ImageEnhance.Brightness(img).enhance(1.0 + intensity * 0.2)

    # Белый оверлей (снежный покров)
    arr = np.array(img).astype(np.float32)
    snow_alpha = intensity * random.uniform(0.15, 0.35)
    snow_tint = np.array([245, 248, 255], dtype=np.float32)
    arr = arr * (1 - snow_alpha) + snow_tint * snow_alpha
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_autumn(img: Image.Image, intensity: float = 0.5) -> Image.Image:
    """
    Осень: жёлто-красный сдвиг листвы.
    """
    arr = np.array(img).astype(np.float32)

    # Усиление красного/жёлтого, ослабление зелёного
    r_shift = intensity * random.uniform(15, 40)
    g_shift = -intensity * random.uniform(10, 25)
    b_shift = -intensity * random.uniform(5, 20)

    arr[:, :, 0] = np.clip(arr[:, :, 0] + r_shift, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + g_shift, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] + b_shift, 0, 255)

    # Лёгкое снижение насыщенности
    img = Image.fromarray(arr.astype(np.uint8))
    img = ImageEnhance.Color(img).enhance(1.0 - intensity * 0.2)
    return img


def apply_summer_scorch(img: Image.Image, intensity: float = 0.5) -> Image.Image:
    """
    Лето: выгоревшая трава (яркость + жёлтый оттенок).
    """
    arr = np.array(img).astype(np.float32)

    # Жёлтый оттенок (усиление R+G, ослабление B)
    yellow_shift = intensity * random.uniform(10, 30)
    arr[:, :, 0] = np.clip(arr[:, :, 0] + yellow_shift * 0.5, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + yellow_shift * 0.3, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] - yellow_shift, 0, 255)

    # Повышение яркости
    img = Image.fromarray(arr.astype(np.uint8))
    img = ImageEnhance.Brightness(img).enhance(1.0 + intensity * 0.15)
    return img


def apply_rain(img: Image.Image, intensity: float = 0.4) -> Image.Image:
    """
    Дождь: затемнение + лёгкие блики/капли.
    """
    # Затемнение
    img = ImageEnhance.Brightness(img).enhance(1.0 - intensity * 0.25)

    # Лёгкое размытие (влажная атмосфера)
    if intensity > 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=intensity * 1.5))

    # Капли (маленькие полупрозрачные точки)
    arr = np.array(img)
    num_drops = int(intensity * random.uniform(50, 200))
    h, w = arr.shape[:2]
    for _ in range(num_drops):
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)
        size = random.randint(1, 3)
        brightness = random.randint(180, 230)
        for dy in range(-size, size + 1):
            for dx in range(-size, size + 1):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and dx*dx + dy*dy <= size*size:
                    blend = 0.3
                    arr[ny, nx] = arr[ny, nx] * (1 - blend) + np.array([brightness]*3) * blend

    return Image.fromarray(arr.astype(np.uint8))


def apply_seasonal(img: Image.Image, intensity: float = 0.5) -> Image.Image:
    """
    Случайная сезонная аугментация.
    """
    season = random.choice(['winter', 'autumn', 'summer', 'rain', 'none'])
    if season == 'winter':
        return apply_winter(img, intensity * random.uniform(0.5, 1.0))
    elif season == 'autumn':
        return apply_autumn(img, intensity * random.uniform(0.5, 1.0))
    elif season == 'summer':
        return apply_summer_scorch(img, intensity * random.uniform(0.5, 1.0))
    elif season == 'rain':
        return apply_rain(img, intensity * random.uniform(0.5, 1.0))
    return img


# ──────────────────────────────────────────────────────────────────────
# Фотометрические и шумовые аугментации
# ──────────────────────────────────────────────────────────────────────

def apply_photometric(img: Image.Image) -> Image.Image:
    """Фотометрические аугментации: яркость, контраст, цвет, гамма."""
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.85, 1.15))
    arr = np.array(img)
    gamma = random.uniform(0.75, 1.25)
    arr = np.power(arr / 255.0, 1.0 / gamma) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_noise(img: Image.Image, sigma_range=(5.0, 25.0)) -> Image.Image:
    """Гауссов шум — имитация матрицы IMX415 при малом свете."""
    arr = np.array(img).astype(np.float32)
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_blur(img: Image.Image, max_radius: float = 2.0) -> Image.Image:
    """Размытие — имитация лёгкой расфокусировки/вибрации."""
    radius = random.uniform(0, max_radius)
    if radius < 0.2:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius))


def apply_jpeg_compression(img: Image.Image, quality_range=(50, 95)) -> Image.Image:
    """JPEG-сжатие — имитация видео-кодека (H.264/H.265)."""
    quality = random.randint(*quality_range)
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


# ──────────────────────────────────────────────────────────────────────
# Конфигурации curriculum-этапов
# ──────────────────────────────────────────────────────────────────────
#
# Этапы идут по нарастающей:
#   0: фотометрия + шум (модель учит базовое сопоставление)
#   1: + поворот + перспектива + лёгкая сезоность
#   2: полный набор (всё на максимальной мощности)
#
# Масштаб НЕ аугментируется — он заложен в размере патча уровня индекса.

CURRICULUM_LEVELS = {
    0: {
        'persp_prob': 0.0,
        'persp_shift': 0.0,
        'rot_prob': 0.0,
        'rot_angle': 0.0,
        'crop_ratio': (0.92, 0.97),
        'cloud_prob': 0.0,
        'cloud_cov': 0.0,
        'fog_prob': 0.0,
        'fog_int': 0.0,
        'shadow_prob': 0.0,
        'seasonal_prob': 0.0,
        'seasonal_int': 0.0,
        'motion_blur_prob': 0.0,
        'motion_blur_px': 0,
        'photo': True,
        'noise_prob': 0.3,
        'noise_sigma': (3.0, 8.0),
        'blur_prob': 0.15,
        'blur_radius': 1.0,
        'jpeg_prob': 0.5,
        'jpeg_quality': (80, 95),
    },
    1: {
        'persp_prob': 0.3,
        'persp_shift': 0.06,
        'rot_prob': 0.8,
        'rot_angle': 30.0,
        'crop_ratio': (0.88, 0.95),
        'cloud_prob': 0.0,
        'cloud_cov': 0.0,
        'fog_prob': 0.0,
        'fog_int': 0.0,
        'shadow_prob': 0.0,
        'seasonal_prob': 0.3,
        'seasonal_int': 0.4,
        'motion_blur_prob': 0.2,
        'motion_blur_px': 8,
        'photo': True,
        'noise_prob': 0.4,
        'noise_sigma': (4.0, 12.0),
        'blur_prob': 0.2,
        'blur_radius': 1.2,
        'jpeg_prob': 0.6,
        'jpeg_quality': (70, 92),
    },
    2: {
        'persp_prob': 0.6,
        'persp_shift': 0.12,
        'rot_prob': 0.8,
        'rot_angle': 30.0,
        'crop_ratio': (0.85, 0.95),
        'cloud_prob': 0.3,
        'cloud_cov': 0.35,
        'fog_prob': 0.2,
        'fog_int': 0.35,
        'shadow_prob': 0.3,
        'seasonal_prob': 0.5,
        'seasonal_int': 0.7,
        'motion_blur_prob': 0.4,
        'motion_blur_px': 15,
        'photo': True,
        'noise_prob': 0.7,
        'noise_sigma': (5.0, 20.0),
        'blur_prob': 0.4,
        'blur_radius': 1.7,
        'jpeg_prob': 0.8,
        'jpeg_quality': (55, 90),
    },
}


def apply_camera_conditions(img: Image.Image, level: int = 2) -> Image.Image:
    """
    Полный набор аугментаций камеры с curriculum-этапом.

    Применяется к anchor-кадру (патч уже прочитан с нужным масштабом уровня).
    Масштаб НЕ применяется здесь — он заложен в размере патча.

    Этапы (level):
      0 — фотометрия + лёгкий шум
      1 — + поворот + перспектива + лёгкая сезоность + motion blur
      2 — полный набор (всё на максимальной мощности)

    НЕ используем flip — зеркальные отражения разрушают пространственную
    структуру и делают anchor неузнаваемым.
    """
    cfg = CURRICULUM_LEVELS.get(level, CURRICULUM_LEVELS[2])

    # 1. Перспектива (наклон камеры)
    if random.random() < cfg['persp_prob']:
        img = apply_perspective_warp(img, cfg['persp_shift'])

    # 2. Поворот (рыскание)
    if random.random() < cfg['rot_prob']:
        img = apply_small_rotation(img, cfg['rot_angle'])

    # 3. Сдвиг/кроп (смещение кадра относительно сетки индекса)
    crop_ratio = random.uniform(*cfg['crop_ratio'])
    w, h = img.size
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)
    x0 = random.randint(0, w - cw)
    y0 = random.randint(0, h - ch)
    img = img.crop((x0, y0, x0 + cw, y0 + ch))
    img = img.resize((w, h), Image.BILINEAR)

    # 4. Motion blur (скорость / пропуск кадров)
    if random.random() < cfg['motion_blur_prob']:
        img = apply_motion_blur(img, cfg['motion_blur_px'])

    # 5. Погодные условия
    if random.random() < cfg['cloud_prob']:
        img = apply_clouds(img, cfg['cloud_cov'])
    if random.random() < cfg['fog_prob']:
        img = apply_fog(img, cfg['fog_int'])
    if random.random() < cfg['shadow_prob']:
        img = apply_shadow_gradient(img)

    # 6. Сезонность
    if random.random() < cfg['seasonal_prob']:
        img = apply_seasonal(img, cfg['seasonal_int'])

    # 7. Фотометрия
    if cfg['photo']:
        img = apply_photometric(img)

    # 8. Шум и артефакты
    if random.random() < cfg['noise_prob']:
        img = apply_noise(img, cfg['noise_sigma'])
    if random.random() < cfg['blur_prob']:
        img = apply_blur(img, cfg['blur_radius'])
    if random.random() < cfg['jpeg_prob']:
        img = apply_jpeg_compression(img, cfg['jpeg_quality'])

    return img
