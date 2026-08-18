"""
augmentations.py — расширенные аугментации для сиамской сети.

Имитируют реальные условия съёмки с дрона (CUAV X7+ Pro, OpenIPC MC800S-V3):
  - Изменение масштаба (высота полёта 700-1200 м)
  - Наклон камеры (perspective warp, нет стабилизации)
  - Рыскание дрона (повороты)
  - Облачность (полупрозрачные пятна)
  - Туман/дымка (серый оверлей)
  - Освещённость (яркость, контраст, тени, гамма)
  - Шум матрицы (гауссов)
  - Сжатие видео-кодека (JPEG)

CURRICULUM-АУГМЕНТАЦИИ:
  ВСЕ аугментации применяются сразу (как раньше) — это ломало сеть:
  модель видела "другой ландшафт" вместо "того же тайла с искажениями"
  (recall падал со 100% до 23% при тесте).

  Теперь сила искажений растёт ПОЭТАПНО (curriculum):
    level 0: только фотометрия + лёгкий шум (эпохи 1-2)
    level 1: + лёгкий масштаб 0.9-1.1, повороты ±5°, лёгкая перспектива
    level 2: + масштаб 0.8-1.25, повороты ±10°, средняя погода
    level 3: полный набор (масштаб 0.7-1.4, повороты ±15°, всё погода)

  Датасет меняет dataset.aug_level по ходу обучения.
"""

import numpy as np
import random
import math
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageOps
from io import BytesIO


def apply_random_scale(img: Image.Image, min_scale: float = 0.5,
                       max_scale: float = 2.0) -> Image.Image:
    """
    Изменение масштаба изображения с возвратом к исходному размеру.
    Имитирует разную высоту полёта: чем выше дрон, тем меньше масштаб.

    scale < 1: объекты меньше (дрон выше) — crop центра + resize вверх
    scale > 1: объекты больше (дрон ниже) — resize вниз + crop
    """
    w, h = img.size
    scale = random.uniform(min_scale, max_scale)

    if scale < 1.0:
        cw, ch = int(w * scale), int(h * scale)
        x0 = (w - cw) // 2 + random.randint(-w // 20, w // 20)
        y0 = (h - ch) // 2 + random.randint(-h // 20, h // 20)
        x0 = max(0, min(w - cw, x0))
        y0 = max(0, min(h - ch, y0))
        img = img.crop((x0, y0, x0 + cw, y0 + ch))
        img = img.resize((w, h), Image.BILINEAR)
    else:
        cw, ch = int(w / scale), int(h / scale)
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        x0 = (img.size[0] - w) // 2 + random.randint(-w // 20, w // 20)
        y0 = (img.size[1] - h) // 2 + random.randint(-h // 20, h // 20)
        x0 = max(0, min(img.size[0] - w, x0))
        y0 = max(0, min(img.size[1] - h, y0))
        img = img.crop((x0, y0, x0 + w, y0 + h))

    return img


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
        # Крен: смещаем верхние углы
        new_corners = [
            (shift_x, shift_y), (w + shift_x, shift_y),
            (w, h), (0, h)
        ]
    else:
        # Тангаж: смещаем правые углы
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
    Камера всегда видит землю целиком, без рамок.
    """
    angle = random.uniform(-max_angle, max_angle)
    w, h = img.size
    # Диагональ — максимальный размер, нужный чтобы после поворота вырезать w×h без пустот
    diag = int(math.ceil(math.sqrt(w**2 + h**2)))
    pad = (diag - min(w, h)) // 2 + 16  # запас, чтобы вырезка не захватывала углы
    # reflect-padding (отражение краёв — как земля вокруг кадра)
    arr = np.array(img)
    arr = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    img = Image.fromarray(arr)
    # Поворот
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
    # Вырезка центра w×h
    cx, cy = img.size[0] // 2, img.size[1] // 2
    img = img.crop((cx - w // 2, cy - h // 2, cx + w - w // 2, cy + h - h // 2))
    return img


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
        x0 = 0 if direction == 'left' else w
        x1 = w if direction == 'left' else 0
        for x in range(0, w, 8):
            t = x / max(1, w - 1)
            alpha = int(strength * 255 * (1 - t)) if direction == 'left' else int(strength * 255 * t)
            draw.line([(x, 0), (x, h)], fill=(0, 0, 0, alpha))
    else:
        y0 = 0 if direction == 'top' else h
        y1 = h if direction == 'top' else 0
        for y in range(0, h, 8):
            t = y / max(1, h - 1)
            alpha = int(strength * 255 * (1 - t)) if direction == 'top' else int(strength * 255 * t)
            draw.line([(0, y), (w, y)], fill=(0, 0, 0, alpha))

    img = Image.alpha_composite(img, overlay)
    return img.convert('RGB')


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


# Конфигурации curriculum-этапов:
#   Этап 1: масштаб (высота полёта) — модель учится инвариантности к масштабу
#   Этап 2: + поворот (рыскание) — добавляем вращение
#   Этап 3: + наклон (перспектива) + погода + шум — полный набор
#
# Каждый этап усиливает искажения постепенно.
CURRICULUM_LEVELS = {
    # Этап 1: только масштаб (0.25× - 2.0×) + лёгкая фотометрия
    0: {
        'scale_range': (0.25, 2.0),      # полный диапазон высот 100-800 м
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
        'photo': True,
        'noise_prob': 0.3,
        'noise_sigma': (3.0, 8.0),
        'blur_prob': 0.15,
        'blur_radius': 1.0,
        'jpeg_prob': 0.5,
        'jpeg_quality': (80, 95),
    },
    # Этап 2: масштаб + поворот (0-30°)
    1: {
        'scale_range': (0.25, 2.0),
        'persp_prob': 0.0,
        'persp_shift': 0.0,
        'rot_prob': 0.8,
        'rot_angle': 30.0,
        'crop_ratio': (0.88, 0.95),
        'cloud_prob': 0.0,
        'cloud_cov': 0.0,
        'fog_prob': 0.0,
        'fog_int': 0.0,
        'shadow_prob': 0.0,
        'photo': True,
        'noise_prob': 0.4,
        'noise_sigma': (4.0, 12.0),
        'blur_prob': 0.2,
        'blur_radius': 1.2,
        'jpeg_prob': 0.6,
        'jpeg_quality': (70, 92),
    },
    # Этап 3: полный набор (масштаб + поворот + перспектива + погода + шум)
    2: {
        'scale_range': (0.25, 2.0),
        'persp_prob': 0.5,
        'persp_shift': 0.10,
        'rot_prob': 0.8,
        'rot_angle': 30.0,
        'crop_ratio': (0.85, 0.95),
        'cloud_prob': 0.3,
        'cloud_cov': 0.35,
        'fog_prob': 0.2,
        'fog_int': 0.35,
        'shadow_prob': 0.3,
        'photo': True,
        'noise_prob': 0.7,
        'noise_sigma': (5.0, 20.0),
        'blur_prob': 0.4,
        'blur_radius': 1.7,
        'jpeg_prob': 0.8,
        'jpeg_quality': (55, 90),
    },
}


def apply_camera_conditions(img: Image.Image, level: int = 2, skip_scale: bool = False) -> Image.Image:
    """
    Полный набор аугментаций камеры с curriculum-этапом.

    Этапы (level):
      0 — масштаб (высота полёта 100-800 м, scale 0.25-2.0)
      1 — масштаб + поворот (рыскание ±30°)
      2 — полный набор (масштаб + поворот + перспектива + погода + шум)

    skip_scale: если True, пропускает масштаб (уже сделан через карту
                в TripletDataset._read_scaled_patch).

    НЕ используем flip — зеркальные отражения разрушают пространственную
    структуру и делают anchor неузнаваемым.
    """
    cfg = CURRICULUM_LEVELS.get(level, CURRICULUM_LEVELS[2])

    # 1. Геометрия: масштаб (высота полёта) — может быть пропущен
    if not skip_scale:
        img = apply_random_scale(img, cfg['scale_range'][0], cfg['scale_range'][1])

    # 2. Перспектива (наклон камеры)
    if random.random() < cfg['persp_prob']:
        img = apply_perspective_warp(img, cfg['persp_shift'])

    # 3. Поворот (рыскание)
    if random.random() < cfg['rot_prob']:
        img = apply_small_rotation(img, cfg['rot_angle'])

    # 4. Сдвиг/кроп (смещение кадра относительно сетки индекса)
    crop_ratio = random.uniform(*cfg['crop_ratio'])
    w, h = img.size
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)
    x0 = random.randint(0, w - cw)
    y0 = random.randint(0, h - ch)
    img = img.crop((x0, y0, x0 + cw, y0 + ch))
    img = img.resize((w, h), Image.BILINEAR)

    # 5. Погодные условия
    if random.random() < cfg['cloud_prob']:
        img = apply_clouds(img, cfg['cloud_cov'])
    if random.random() < cfg['fog_prob']:
        img = apply_fog(img, cfg['fog_int'])
    if random.random() < cfg['shadow_prob']:
        img = apply_shadow_gradient(img)

    # 6. Фотометрия
    if cfg['photo']:
        img = apply_photometric(img)

    # 7. Шум и артефакты
    if random.random() < cfg['noise_prob']:
        img = apply_noise(img, cfg['noise_sigma'])
    if random.random() < cfg['blur_prob']:
        img = apply_blur(img, cfg['blur_radius'])
    if random.random() < cfg['jpeg_prob']:
        img = apply_jpeg_compression(img, cfg['jpeg_quality'])

    return img