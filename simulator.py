"""
Симуляция полёта дрона над подстилающей поверхностью.
Генерирует кадры с камеры на основе положения и движения дрона.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field
import math
import time


@dataclass
class DroneState:
    """Состояние дрона."""
    x: float = 0.0       # позиция X (метры от начала карты)
    y: float = 0.0       # позиция Y (метры от начала карты)
    z: float = 50.0      # высота (метры)
    vx: float = 0.0      # скорость по X (м/с)
    vy: float = 0.0      # скорость по Y (м/с)
    vz: float = 0.0      # скорость по Z (м/с)
    heading: float = 0.0 # направление (градусы, 0 = север)
    timestamp: float = 0.0


@dataclass
class CameraConfig:
    """Конфигурация камеры."""
    fov: float = 60.0        # поле зрения (градусы)
    width: int = 512         # ширина кадра (пиксели)
    height: int = 512        # высота кадра (пиксели)
    resolution: float = 0.1  # м/пиксель (зависит от высоты)


class FlightSimulator:
    """
    Симулятор полёта дрона.
    Генерирует кадры камеры на основе положения дрона и карты.
    """

    def __init__(self, map_loader, map_region, camera_config: CameraConfig = None):
        self.map_loader = map_loader
        self.map = map_region
        self.camera = camera_config or CameraConfig()
        self.drone = DroneState()
        self.frame_count = 0
        self.start_time = time.time()

    def set_drone_position(self, x: float, y: float, z: float = 50.0):
        """Установить положение дрона."""
        self.drone.x = x
        self.drone.y = y
        self.drone.z = z

    def set_drone_velocity(self, vx: float, vy: float, vz: float = 0.0):
        """Установить скорость дрона."""
        self.drone.vx = vx
        self.drone.vy = vy
        self.drone.vz = vz

    def update(self, dt: float = 1/30.0):
        """
        Обновить состояние дрона.

        Args:
            dt: шаг времени (секунды)
        """
        self.drone.x += self.drone.vx * dt
        self.drone.y += self.drone.vy * dt
        self.drone.z += self.drone.vz * dt

        # Высота не может быть отрицательной
        self.drone.z = max(5.0, self.drone.z)

        self.drone.timestamp = time.time() - self.start_time
        self.frame_count += 1

    def get_camera_frame(self) -> np.ndarray:
        """
        Сгенерировать кадр с камеры дрона.

        Returns:
            Кадр (HxWx3, RGB, uint8)
        """
        # Определяем область карты, видимую камерой
        # Получаем фрагмент карты напрямую из тайла
        region = self.map
        if region is None:
            return np.zeros((self.camera.height, self.camera.width, 3), dtype=np.uint8)

        tile = list(region.tiles.values())[0]
        h, w = tile.image.shape[:2]
        resolution = region.resolution

        # FOV 60° на высоте 50м ≈ видимая область ~57м
        fov_rad = math.radians(self.camera.fov)
        visible_meters = self.drone.z * math.tan(fov_rad / 2) * 2

        # Конвертируем в пиксели
        visible_pixels = int(visible_meters / resolution)
        half = visible_pixels // 2

        # Позиция дрона в пикселях от центра карты
        drone_px_x = w // 2 + int(self.drone.x / resolution)
        drone_px_y = h // 2 + int(self.drone.y / resolution)

        # Вырезаем фрагмент карты вокруг дрона
        x_start = max(0, drone_px_x - half)
        y_start = max(0, drone_px_y - half)
        x_end = min(w, x_start + visible_pixels)
        y_end = min(h, y_start + visible_pixels)

        crop = tile.image[y_start:y_end, x_start:x_end]

        # Дополняем до нужного размера
        if crop.shape[0] < visible_pixels or crop.shape[1] < visible_pixels:
            padded = np.zeros((visible_pixels, visible_pixels, 3), dtype=np.uint8)
            ph = min(crop.shape[0], visible_pixels)
            pw = min(crop.shape[1], visible_pixels)
            padded[:ph, :pw] = crop[:ph, :pw]
            crop = padded

        # Масштабируем до размера камеры
        crop = cv2.resize(crop, (self.camera.width, self.camera.height))

        # Масштабируем до размера камеры
        crop = cv2.resize(crop, (self.camera.width, self.camera.height))

        return crop

    def get_ground_truth(self) -> DroneState:
        """Получить истинное состояние дрона."""
        return DroneState(
            x=self.drone.x,
            y=self.drone.y,
            z=self.drone.z,
            vx=self.drone.vx,
            vy=self.drone.vy,
            vz=self.drone.vz,
            heading=self.drone.heading,
            timestamp=self.drone.timestamp
        )

    def follow_path(self, path: List[Tuple[float, float, float]],
                    speed: float = 5.0):
        """
        Следовать заданному пути.

        Args:
            path: список (x, y, z) — точек пути
            speed: скорость движения (м/с)
        """
        if not path:
            return

        self.path = path
        self.path_index = 0
        self.path_speed = speed

    def update_path_following(self, dt: float = 1/30.0):
        """Обновить движение по пути."""
        if not hasattr(self, 'path') or not self.path:
            return

        target = self.path[self.path_index]
        dx = target[0] - self.drone.x
        dy = target[1] - self.drone.y
        dz = target[2] - self.drone.z
        dist = math.sqrt(dx**2 + dy**2 + dz**2)

        if dist < 1.0:
            self.path_index += 1
            if self.path_index >= len(self.path):
                self.path_index = 0  # зацикливаем
                return

            target = self.path[self.path_index]
            dx = target[0] - self.drone.x
            dy = target[1] - self.drone.y
            dz = target[2] - self.drone.z

        # Направляем к следующей точке
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        self.drone.vx = (dx / dist) * self.path_speed
        self.drone.vy = (dy / dist) * self.path_speed
        self.drone.vz = (dz / dist) * self.path_speed

        self.update(dt)

    def reset(self):
        """Сбросить симуляцию."""
        self.drone = DroneState()
        self.frame_count = 0
        self.start_time = time.time()
