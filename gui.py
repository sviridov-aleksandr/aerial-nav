"""
Визуализация полёта дрона и данных навигации.
"""

import numpy as np
import cv2
from typing import Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.collections import LineCollection
from simulator import DroneState


class NavigationGUI:
    """
    Визуализация данных навигации дрона.
    Показывает:
    - Кадр с камеры дрона
    - Карта с положением дрона
    - Графики положения, скорости, направления
    """

    def __init__(self, show_plot: bool = True):
        self.show_plot = show_plot
        self.fig = None
        self.axes = {}
        self.is_open = False

    def open(self):
        """Открыть окно визуализации."""
        if self.show_plot:
            self.fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            self.axes = {
                'camera': axes[0, 0],
                'map': axes[0, 1],
                'position': axes[1, 0],
                'velocity': axes[1, 1]
            }
            self._setup_axes()
            self.is_open = True

    def _setup_axes(self):
        """Настроить оси графиков."""
        self.axes['camera'].set_title('Камера дрона')
        self.axes['camera'].set_axis_off()

        self.axes['map'].set_title('Карта и положение дрона')
        self.axes['map'].set_aspect('equal')

        self.axes['position'].set_title('Положение (м)')
        self.axes['position'].set_xlabel('X (м)')
        self.axes['position'].set_ylabel('Y (м)')

        self.axes['velocity'].set_title('Скорость и направление')
        self.axes['velocity'].set_xlabel('Время (с)')

    def update(self, camera_frame: np.ndarray, map_crop: np.ndarray,
               nav_data: Dict, drone_state: DroneState,
               position_history: list = None,
               velocity_history: list = None,
               heading_history: list = None):
        """
        Обновить визуализацию.

        Args:
            camera_frame: кадр с камеры
            map_crop: фрагмент карты
            nav_data: данные навигации
            drone_state: состояние дрона
            position_history: история положения
            velocity_history: история скорости
            heading_history: история направления
        """
        if not self.is_open:
            return

        if not nav_data:
            nav_data = {}

        # 1. Кадр с камеры
        self.axes['camera'].clear()
        if camera_frame is not None:
            self.axes['camera'].imshow(camera_frame)
        self.axes['camera'].set_title('Камера дрона')
        self.axes['camera'].set_axis_off()

        # 2. Карта
        self.axes['map'].clear()
        if map_crop is not None:
            self.axes['map'].imshow(map_crop)
            # Рисуем положение дрона
            pos = nav_data.get('position') or {}
            pos_x = pos.get('x', 0) or 0
            pos_y = pos.get('y', 0) or 0
            self.axes['map'].plot(pos_x, pos_y, 'ro', markersize=10, label='Дрон')
            # Стрелка направления
            heading = nav_data.get('heading') or 0
            arrow_len = 20
            dx = arrow_len * np.sin(np.radians(heading))
            dy = arrow_len * np.cos(np.radians(heading))
            self.axes['map'].arrow(pos_x, pos_y, dx, dy,
                                   head_width=5, head_length=5,
                                   fc='yellow', ec='orange', linewidth=2)
            self.axes['map'].legend()
        self.axes['map'].set_title(f'Карта | Позиция: ({pos_x:.1f}, {pos_y:.1f}) м')

        # 3. График положения
        self.axes['position'].clear()
        if position_history and len(position_history) > 1:
            x_hist = [p[0] for p in position_history]
            y_hist = [p[1] for p in position_history]
            self.axes['position'].plot(x_hist, y_hist, 'b-', linewidth=1, label='Траектория')
            self.axes['position'].plot(x_hist[-1], y_hist[-1], 'ro', markersize=8, label='Текущая')
            self.axes['position'].set_xlabel('X (м)')
            self.axes['position'].set_ylabel('Y (м)')
            self.axes['position'].legend()
            self.axes['position'].set_title('Траектория')

        # 4. График скорости и направления
        self.axes['velocity'].clear()
        if velocity_history and len(velocity_history) > 1:
            t = np.arange(len(velocity_history))
            self.axes['velocity'].plot(t, velocity_history, 'g-', linewidth=1, label='Скорость (м/с)')
            self.axes['velocity'].set_xlabel('Время (кадров)')
            self.axes['velocity'].set_ylabel('Скорость (м/с)')
            self.axes['velocity'].legend()
            self.axes['velocity'].set_title('Скорость')

        # Добавляем текстовую информацию
        vel = nav_data.get('velocity') or {}
        pos = nav_data.get('position') or {}
        speed = vel.get('speed_kmh', 0) or 0
        heading = nav_data.get('heading') or 0
        confidence = nav_data.get('confidence') or 0
        matches = nav_data.get('num_matches') or 0

        info_text = (f"Скорость: {speed:.1f} км/ч\n"
                     f"Направление: {heading:.1f}°\n"
                     f"Уверенность: {confidence:.2f}\n"
                     f"Совпадений: {matches}")

        self.fig.text(0.5, 0.02, info_text, ha='center', va='bottom',
                      fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.pause(0.01)

    def close(self):
        """Закрыть визуализацию."""
        if self.fig is not None:
            plt.close(self.fig)
            self.is_open = False

    def show_static(self, camera_frame: np.ndarray, map_crop: np.ndarray,
                    nav_data: Dict):
        """Показать статический снимок (без анимации)."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        if camera_frame is not None:
            axes[0].imshow(camera_frame)
            axes[0].set_title('Камера дрона')
            axes[0].set_axis_off()

        if map_crop is not None:
            axes[1].imshow(map_crop)
            pos_x = nav_data.get('position', {}).get('x', 0)
            pos_y = nav_data.get('position', {}).get('y', 0)
            axes[1].plot(pos_x, pos_y, 'ro', markersize=10)
            axes[1].set_title(f'Карта ({pos_x:.1f}, {pos_y:.1f})')
            axes[1].set_axis_off()

        # График данных
        speed = nav_data.get('velocity', {}).get('speed_kmh', 0)
        heading = nav_data.get('heading', 0)
        confidence = nav_data.get('confidence', 0)

        axes[2].text(0.5, 0.7, f"Скорость: {speed:.1f} км/ч",
                     transform=axes[2].transAxes, fontsize=14, va='center')
        axes[2].text(0.5, 0.5, f"Направление: {heading:.1f}°",
                     transform=axes[2].transAxes, fontsize=14, va='center')
        axes[2].text(0.5, 0.3, f"Уверенность: {confidence:.2f}",
                     transform=axes[2].transAxes, fontsize=14, va='center')
        axes[2].text(0.5, 0.1, f"Совпадений: {nav_data.get('num_matches', 0)}",
                     transform=axes[2].transAxes, fontsize=14, va='center')
        axes[2].set_axis_off()

        plt.tight_layout()
        plt.show()
