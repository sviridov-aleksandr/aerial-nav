"""
Фильтр Калмана для сглаживания оценок положения и скорости дрона.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class KalmanState:
    """Состояние фильтра Калмана."""
    x: np.ndarray       # состояние [pos_x, pos_y, vel_x, vel_y]
    P: np.ndarray       # ковариация
    A: np.ndarray       # матрица перехода
    H: np.ndarray       # матрица наблюдения
    Q: np.ndarray       # ковариация процесса
    R: np.ndarray       # ковариация измерений


class KalmanFilter2D:
    """
    Фильтр Калмана для 2D навигации.
    
    Состояние: [pos_x, pos_y, vel_x, vel_y]
    Измерение: [pos_x, pos_y]
    """

    def __init__(self, dt: float = 1/30.0, process_noise: float = 1.0,
                 measurement_noise: float = 10.0):
        self.dt = dt
        self.state = None
        self.initialized = False

        # Параметры шума
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def _init_matrices(self):
        """Инициализация матриц фильтра Калмана."""
        # Матрица перехода состояния (x_{t+1} = A * x_t)
        self.A = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)

        # Матрица наблюдения (z = H * x)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)

        # Коэффициент процесса (дрон может менять скорость)
        B = np.array([
            [self.dt**2 / 2, 0],
            [0, self.dt**2 / 2],
            [self.dt, 0],
            [0, self.dt]
        ], dtype=np.float64)
        self.Q = B @ B.T * self.process_noise

        # Коэффициент измерений (шум камеры)
        self.R = np.eye(2) * self.measurement_noise

    def predict(self):
        """Предсказание следующего состояния."""
        if not self.initialized:
            return

        self.state = self.A @ self.state
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Обновление состояния по измерению.

        Args:
            measurement: [pos_x, pos_y] в метрах

        Returns:
            Скорректированное состояние [pos_x, pos_y, vel_x, vel_y]
        """
        if not self.initialized:
            # Инициализация
            self._init_matrices()
            self.state = np.array([
                measurement[0],
                measurement[1],
                0.0,
                0.0
            ], dtype=np.float64)
            self.P = np.eye(4) * 100.0
            self.initialized = True
            return self.state.copy()

        # Предсказание
        self.predict()

        # Обновление
        y = measurement - self.H @ self.state  # инновация
        S = self.H @ self.P @ self.H.T + self.R  # инновационная ковариация
        K = self.P @ self.H.T @ np.linalg.inv(S)  # фильтр Калмана

        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.state.copy()

    def get_position(self) -> np.ndarray:
        """Получить позицию [x, y]."""
        if not self.initialized:
            return np.array([0.0, 0.0])
        return self.state[:2].copy()

    def get_velocity(self) -> np.ndarray:
        """Получить скорость [vx, vy]."""
        if not self.initialized:
            return np.array([0.0, 0.0])
        return self.state[2:].copy()

    def get_speed_ms(self) -> float:
        """Получить модуль скорости (м/с)."""
        v = self.get_velocity()
        return float(np.linalg.norm(v))

    def get_speed_kmh(self) -> float:
        """Получить скорость (км/ч)."""
        return self.get_speed_ms() * 3.6

    def get_heading(self) -> float:
        """Получить направление (градусы)."""
        v = self.get_velocity()
        if np.linalg.norm(v) < 0.01:
            return 0.0
        return float(np.degrees(np.arctan2(v[0], v[1])) % 360)
