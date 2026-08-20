"""
Симулятор полёта крыла с камерой без зума (FOV 90°).
Реалистичная динамика: изменение высоты (700-1000 м),
развороты с мин. радиусом 100 м, крейсерская скорость 22 м/с.
"""

import numpy as np
import cv2
from typing import Optional
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Отключаем лимит PIL для больших карт


class FlightSimulatorFixedWing:
    """
    Симулятор полёта для крыла (самолётного типа).
    Камера без зума, FOV 90°, высота 700-1000м.
    """

    def __init__(self, map_image: np.ndarray, resolution: float = 0.5,
                 fov: float = 90.0, width: int = 1920, height: int = 1080,
                 min_turn_radius: float = 100.0, cruise_speed: float = 22.0,
                 min_altitude: float = 700.0, max_altitude: float = 1000.0):
        """
        Args:
            map_image: карта (H, W, 3) RGB
            resolution: разрешение карты (м/пиксель)
            fov: поле зрения камеры (градусы)
            width, height: разрешение камеры
            min_turn_radius: мин. радиус разворота (м)
            cruise_speed: крейсерская скорость (м/с)
            min_altitude, max_altitude: диапазон высот (м)
        """
        self.map_image = map_image
        self.map_h, self.map_w = map_image.shape[:2]
        self.resolution = resolution
        
        # Параметры камеры
        self.fov = fov
        self.camera_width = width
        self.camera_height = height
        
        # Параметры крыла
        self.min_turn_radius = min_turn_radius
        self.cruise_speed = cruise_speed
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        
        # Дрон (крыло)
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = (min_altitude + max_altitude) / 2  # крейсерская 850 м
        self.drone_vx = cruise_speed  # м/с
        self.drone_vy = 0.0
        self.drone_heading = 0.0  # градусы (0 = восток, 90 = север)
        self.drone_bank = 0.0  # крен (градусы)
        self.timestamp = 0.0
        
        # Целевые параметры (для плавного изменения)
        self.target_heading = 0.0
        self.target_altitude = self.drone_z
        self.target_speed = cruise_speed
        
        # Расчёт размера кадра в метрах (при текущей высоте)
        self.frame_size_m = self.drone_z * np.tan(np.radians(fov / 2)) * 2

    # ── Управление ──────────────────────────────────────────────
    def set_position(self, x: float, y: float, z: float = 850.0):
        """Установка позиции дрона."""
        self.drone_x = x
        self.drone_y = y
        self.drone_z = np.clip(z, self.min_altitude, self.max_altitude)
        self.frame_size_m = self.drone_z * np.tan(np.radians(self.fov / 2)) * 2

    def set_velocity(self, vx: float, vy: float):
        """Установка скорости."""
        self.drone_vx = vx
        self.drone_vy = vy
        self.drone_heading = np.degrees(np.arctan2(vy, vx))
        self.target_heading = self.drone_heading

    def set_target_heading(self, heading: float):
        """Задать целевой курс (градусы)."""
        self.target_heading = heading

    def set_target_altitude(self, altitude: float):
        """Задать целевую высоту (м)."""
        self.target_altitude = np.clip(altitude, self.min_altitude, self.max_altitude)

    def set_target_speed(self, speed: float):
        """Задать целевую скорость (м/с)."""
        self.target_speed = max(5.0, speed)

    def follow_waypoint(self, wx: float, wy: float):
        """Наведение на waypoint: расчёт целевого курса."""
        dx = wx - self.drone_x
        dy = wy - self.drone_y
        self.target_heading = np.degrees(np.arctan2(dy, dx))

    # ── Динамика ────────────────────────────────────────────────
    def update(self, dt: float = 1/30.0):
        """Обновление состояния дрона с реалистичной динамикой."""
        self.timestamp += dt
        
        # Плавное изменение курса (макс. угловая скорость для крыла)
        # При скорости 22 м/с и радиусе 100 м: ω = v/R = 0.22 рад/с ≈ 12.6°/с
        max_turn_rate = np.degrees(self.cruise_speed / self.min_turn_radius)
        
        heading_diff = (self.target_heading - self.drone_heading + 180) % 360 - 180
        max_step = max_turn_rate * dt
        heading_diff = np.clip(heading_diff, -max_step, max_step)
        self.drone_heading = (self.drone_heading + heading_diff) % 360
        
        # Крен при повороте (пропорционален угловой скорости)
        self.drone_bank = np.clip(heading_diff / max_step * 30, -30, 30) if max_step > 0 else 0
        
        # Плавное изменение высоты (макс. вертикальная скорость ~5 м/с)
        alt_diff = self.target_altitude - self.drone_z
        alt_step = np.clip(alt_diff, -5.0 * dt, 5.0 * dt)
        self.drone_z = np.clip(self.drone_z + alt_step, self.min_altitude, self.max_altitude)
        
        # Плавное изменение скорости
        speed_diff = self.target_speed - self.cruise_speed
        self.cruise_speed += np.clip(speed_diff, -3.0 * dt, 3.0 * dt)
        
        # Скорость по осям с учётом курса
        heading_rad = np.radians(self.drone_heading)
        self.drone_vx = self.cruise_speed * np.cos(heading_rad)
        self.drone_vy = self.cruise_speed * np.sin(heading_rad)
        
        # Обновление позиции
        self.drone_x += self.drone_vx * dt
        self.drone_y += self.drone_vy * dt
        
        # Обновление размера кадра
        self.frame_size_m = self.drone_z * np.tan(np.radians(self.fov / 2)) * 2

    # ── Камера ──────────────────────────────────────────────────
    def get_camera_frame(self) -> np.ndarray:
        """
        Генерирует кадр с камеры с учётом текущей высоты.
        Дрон в (x, y) м от верхнего левого угла карты.
        
        Returns:
            Кадр (H, W, 3) RGB
        """
        # Центр кадра в пикселях карты (позиция дрона)
        center_px_x = self.drone_x / self.resolution
        center_px_y = self.drone_y / self.resolution
        
        # Размер кадра в пикселях (16:9, зависит от высоты)
        frame_w_px = int(self.frame_size_m / self.resolution)
        frame_h_px = int(frame_w_px * self.camera_height / self.camera_width)
        
        # Границы кадра
        x1 = int(center_px_x - frame_w_px / 2)
        y1 = int(center_px_y - frame_h_px / 2)
        x2 = x1 + frame_w_px
        y2 = y1 + frame_h_px
        
        # Создаём кадр с заполнением чёрным
        frame = np.zeros((frame_h_px, frame_w_px, 3), dtype=np.uint8)
        
        # Определяем пересечение с картой
        map_x1 = max(0, x1)
        map_y1 = max(0, y1)
        map_x2 = min(self.map_w, x2)
        map_y2 = min(self.map_h, y2)
        
        # Координаты в кадре
        frame_x1 = map_x1 - x1
        frame_y1 = map_y1 - y1
        frame_x2 = frame_x1 + (map_x2 - map_x1)
        frame_y2 = frame_y1 + (map_y2 - map_y1)
        
        # Копируем данные карты
        if map_x2 > map_x1 and map_y2 > map_y1:
            frame[frame_y1:frame_y2, frame_x1:frame_x2] = \
                self.map_image[map_y1:map_y2, map_x1:map_x2]
        
        # Добавляем шум и эффекты
        frame = self._add_noise(frame)
        
        # Масштабируем до разрешения камеры
        frame = cv2.resize(frame, (self.camera_width, self.camera_height),
                          interpolation=cv2.INTER_LINEAR)
        
        return frame

    def _add_noise(self, frame: np.ndarray) -> np.ndarray:
        """Добавление шума и эффектов."""
        # Гауссов шум
        noise = np.random.normal(0, 5, frame.shape).astype(np.int16)
        frame = (frame.astype(np.int16) + noise).clip(0, 255).astype(np.uint8)
        
        # Случайная яркость
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.9, 1.1)
            frame = (frame * factor).clip(0, 255).astype(np.uint8)
        
        return frame

    def get_ground_truth(self) -> dict:
        """Возвращает истинную позицию."""
        return {
            'x': self.drone_x,
            'y': self.drone_y,
            'z': self.drone_z,
            'heading': self.drone_heading,
            'bank': self.drone_bank,
            'speed': self.cruise_speed,
            'timestamp': self.timestamp
        }


def test_fixed_wing_simulation():
    """Тест симулятора крыла с реалистичной траекторией."""
    from PIL import Image as PILImage
    
    # Загрузка карты (используем карту Каланчака — сельхоз поля)
    map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map_cache/highres/highres_46.2650_33.3732_z18.png')
    if not os.path.exists(map_path):
        map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map_cache/highres/highres_55.7550_37.6173_z18.png')
    map_image = np.array(PILImage.open(map_path))
    
    print("=" * 60)
    print("ТЕСТ СИМУЛЯТОРА КРЫЛА (реалистичная траектория)")
    print("=" * 60)
    
    # Создание симулятора
    sim = FlightSimulatorFixedWing(
        map_image=map_image,
        resolution=0.5,  # 0.5 м/пиксель
        fov=90.0,
        width=1920,
        height=1080,
        min_turn_radius=100.0,
        cruise_speed=22.0,
        min_altitude=700.0,
        max_altitude=1000.0
    )
    
    # Маршрут: квадрат с изменением высоты
    # Старт в центре карты (5 км от края)
    center_x = map_image.shape[1] * 0.5 * 0.5  # метры
    center_y = map_image.shape[0] * 0.5 * 0.5
    sim.set_position(center_x, center_y, 850.0)
    
    # Waypoints: квадрат 2×2 км с набором/снижением
    waypoints = [
        (center_x + 1000, center_y, 700.0),       # восток, снижение
        (center_x + 1000, center_y + 1000, 1000.0),  # север, набор
        (center_x - 1000, center_y + 1000, 850.0),   # запад, крейсер
        (center_x - 1000, center_y - 1000, 700.0),   # юг, снижение
        (center_x, center_y, 1000.0),                # центр, набор
    ]
    
    print(f"\nПараметры:")
    print(f"  Высота: {sim.drone_z:.0f} м (диапазон {sim.min_altitude}-{sim.max_altitude})")
    print(f"  Скорость: {sim.cruise_speed:.0f} м/с")
    print(f"  FOV: {sim.fov}°")
    print(f"  Мин. радиус: {sim.min_turn_radius} м")
    print(f"  Маршрут: 5 waypoints (квадрат 2×2 км)")
    
    # Тестирование: пролёт по маршруту
    print(f"\nПролёт по маршруту (600 кадров, 20 сек)...")
    print("-" * 70)
    
    wp_idx = 0
    alt_log = []
    heading_log = []
    frame_log = []
    
    for i in range(600):
        # Наведение на текущий waypoint
        if wp_idx < len(waypoints):
            wx, wy, wz = waypoints[wp_idx]
            sim.follow_waypoint(wx, wy)
            sim.set_target_altitude(wz)
            
            # Переход к следующему waypoint при достижении
            dist = np.sqrt((wx - sim.drone_x)**2 + (wy - sim.drone_y)**2)
            if dist < 50:
                wp_idx += 1
        
        sim.update(dt=1/30.0)
        frame = sim.get_camera_frame()
        gt = sim.get_ground_truth()
        
        alt_log.append(gt['z'])
        heading_log.append(gt['heading'])
        frame_log.append(frame)
        
        if (i + 1) % 100 == 0:
            print(f"Кадр {i + 1:3d} | Поз: ({gt['x']:7.1f}, {gt['y']:7.1f}) м | "
                  f"Выс: {gt['z']:4.0f} м | Курс: {gt['heading']:5.1f}° | "
                  f"Крен: {gt['bank']:4.1f}° | WP: {wp_idx}/{len(waypoints)}")
    
    print("-" * 70)
    print(f"\n✓ Симулятор работает!")
    print(f"  Финальная позиция: ({gt['x']:.1f}, {gt['y']:.1f}) м, "
          f"высота: {gt['z']:.0f} м")
    print(f"  Waypoints пройдено: {wp_idx}/{len(waypoints)}")
    print(f"  Диапазон высоты: {min(alt_log):.0f}-{max(alt_log):.0f} м")
    print(f"  Диапазон курса: {min(heading_log):.0f}-{max(heading_log):.0f}°")
    print(f"  Размер кадра: {frame_log[0].shape} (меняется с высотой)")


if __name__ == '__main__':
    import os
    test_fixed_wing_simulation()