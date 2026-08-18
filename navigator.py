"""
Главный модуль навигации.
Объединяет карту, извлечение признаков, сопоставление и оценку позы.
"""

import numpy as np
from typing import Dict, Optional, List
from collections import deque
from map_loader import MapLoader
from feature_extractor import SuperPointFeatureExtractor, SuperGlueMatcher, PoseEstimator
from kalman_filter import KalmanFilter2D
from simulator import FlightSimulator, DroneState, CameraConfig


class AerialNavigator:
    """
    Автономный навигатор дрона по подстилающей поверхности.
    
    Работает в цикле:
    1. Загружает карту местности
    2. Получает кадр с камеры дрона
    3. Извлекает ключевые точки из кадра и карты
    4. Сопоставляет точки
    5. Оценивает положение, скорость, направление
    6. Сглаживает оценки через фильтр Калмана
    """

    def __init__(self, map_loader: MapLoader, resolution: float = 0.1,
                 use_cuda: bool = True):
        self.map_loader = map_loader
        self.resolution = resolution

        # Компоненты
        self.feature_extractor = SuperPointFeatureExtractor(use_cuda=use_cuda)
        self.matcher = SuperGlueMatcher(use_cuda=use_cuda)
        self.pose_estimator = PoseEstimator(resolution=resolution)

        # Фильтр Калмана для сглаживания
        self.kalman = KalmanFilter2D(dt=1/30.0, process_noise=1.0, measurement_noise=5.0)

        # Текущее состояние
        self.current_pose: Optional[Dict] = None
        self.is_initialized = False
        self.base_lat = 55.7550
        self.base_lon = 37.6173

    def initialize(self, map_path: str = None, lat: float = 55.7550,
                   lon: float = 37.6173, synthetic: bool = True,
                   init_x: float = 0.0, init_y: float = 0.0):
        """
        Инициализировать навигатор, загрузив карту.

        Args:
            map_path: путь к файлу карты (если None, создаётся синтетическая)
            lat: широта центра карты
            lon: долгота центра карты
            synthetic: если True, создаётся синтетическая карта
            init_x: начальная позиция X (метры от центра карты)
            init_y: начальная позиция Y (метры от центра карты)
        """
        if synthetic and map_path is None:
            print("[Navigator] Создаю синтетическую карту...")
            self.map_loader.generate_synthetic_map(size=1024, resolution=self.resolution)
        elif map_path:
            print(f"[Navigator] Загружаю карту из {map_path}...")
            self.map_loader.load_from_image(map_path, lat, lon, self.resolution)
        else:
            print("[Navigator] Создаю синтетическую карту...")
            self.map_loader.generate_synthetic_map(size=1024, resolution=self.resolution)

        self.is_initialized = True
        self.init_x = init_x
        self.init_y = init_y
        self.kalman = KalmanFilter2D(dt=1/30.0, process_noise=1.0, measurement_noise=5.0)
        # Инициализируем фильтр Калмана начальной позицией
        self.kalman.update(np.array([init_x, init_y]))
        print(f"[Navigator] Карта загружена. Разрешение: {self.resolution} м/пиксель")
        print(f"[Navigator] Начальная позиция: ({init_x:.1f}, {init_y:.1f}) м")

    def process_frame(self, camera_frame: np.ndarray,
                      timestamp: float = None) -> Dict:
        """
        Обработать один кадр с камеры.

        Args:
            camera_frame: кадр с камеры дрона (HxWx3, RGB)
            timestamp: время кадра

        Returns:
            Словарь с оценками положения, скорости, направления
        """
        if not self.is_initialized:
            raise RuntimeError("Навигатор не инициализирован. Вызовите initialize()")

        # 1. Извлекаем ключевые точки из кадра камеры
        cam_kps = self.feature_extractor.extract(camera_frame)

        # 2. Получаем фрагмент карты в текущем положении
        # Для этого используем предыдущую оценку положения
        map_crop = self._get_map_crop_for_current_position()

        # 3. Извлекаем ключевые точки из карты
        map_kps = self.feature_extractor.extract(map_crop)

        # 4. Сопоставляем точки
        matches1, matches2 = self.matcher.match(
            map_kps, cam_kps, map_crop, camera_frame
        )

        # 5. Оцениваем позу
        raw_pose = self.pose_estimator.estimate_pose(
            matches1, matches2, map_kps, cam_kps,
            map_crop, camera_frame, timestamp
        )

        # 6. Фильтр Калмана
        if raw_pose and raw_pose.get('position') and matches1.size > 0:
            pos_meters = raw_pose['position']['meters']
            kalman_state = self.kalman.update(pos_meters)
            
            # Обновляем pose сглаженными значениями
            raw_pose['position']['meters'] = kalman_state[:2]
            raw_pose['position']['x_meters'] = kalman_state[0]
            raw_pose['position']['y_meters'] = kalman_state[1]
            raw_pose['velocity'] = {
                'speed_ms': float(np.linalg.norm(kalman_state[2:])),
                'speed_kmh': float(np.linalg.norm(kalman_state[2:]) * 3.6),
                'vx': kalman_state[2],
                'vy': kalman_state[3]
            }
            raw_pose['heading'] = float(np.degrees(np.arctan2(kalman_state[2], kalman_state[1])) % 360) if np.linalg.norm(kalman_state[2:]) > 0.01 else 0.0

        self.current_pose = raw_pose
        return raw_pose

    def _get_map_crop_for_current_position(self) -> np.ndarray:
        """Получить фрагмент карты для текущего положения.
        
        Работает напрямую с пиксельными смещениями от центра карты.
        """
        if self.current_pose and self.current_pose.get('position'):
            pos = self.current_pose['position']['meters']
            x_m, y_m = pos[0], pos[1]
            # Конвертируем метры в пиксели
            x_px = x_m / self.resolution
            y_px = y_m / self.resolution
        else:
            x_px = 0
            y_px = 0

        # Получаем фрагмент карты вокруг центра + смещение
        region = self.map_loader.get_region()
        if region is None:
            return None

        tile = list(region.tiles.values())[0]
        h, w = tile.image.shape[:2]

        # Размер фрагмента в пикселях
        crop_size_px = 256
        half = crop_size_px // 2

        # Центрируем на текущей оценке положения
        center_x = w // 2 + int(x_px)
        center_y = h // 2 + int(y_px)

        x_start = max(0, center_x - half)
        y_start = max(0, center_y - half)
        x_end = min(w, x_start + crop_size_px)
        y_end = min(h, y_start + crop_size_px)

        crop = tile.image[y_start:y_end, x_start:x_end]

        # Дополняем до нужного размера если вышли за границы
        if crop.shape[0] < crop_size_px or crop.shape[1] < crop_size_px:
            padded = np.zeros((crop_size_px, crop_size_px, 3), dtype=np.uint8)
            ph = min(crop.shape[0], crop_size_px)
            pw = min(crop.shape[1], crop_size_px)
            padded[:ph, :pw] = crop[:ph, :pw]
            crop = padded

        return crop

    def get_navigation_data(self) -> Dict:
        """Получить все данные навигации."""
        if not self.current_pose:
            return {
                'position': {'x': 0, 'y': 0},
                'velocity': {'speed': 0},
                'heading': 0,
                'confidence': 0
            }

        pose = self.current_pose
        return {
            'position': {
                'x': pose.get('position', {}).get('x_meters', 0),
                'y': pose.get('position', {}).get('y_meters', 0),
                'confidence': pose.get('confidence', 0)
            },
            'velocity': {
                'speed_ms': pose.get('velocity', {}).get('speed_ms', 0),
                'speed_kmh': pose.get('velocity', {}).get('speed_kmh', 0)
            },
            'heading': pose.get('heading', 0),
            'num_matches': pose.get('num_matches', 0)
        }

    def reset(self):
        """Сбросить навигатор."""
        self.pose_estimator.reset()
        self.kalman = KalmanFilter2D(dt=1/30.0, process_noise=1.0, measurement_noise=5.0)
        self.current_pose = None
        print("[Navigator] Навигатор сброшен")
