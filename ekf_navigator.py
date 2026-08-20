"""
EKF Navigator — навигация с fusion IMU odometry + Siamese map matching.

Архитектура:
  State: [x, y, vx, vy, heading]
  Predict: IMU odometry (каждые 0.1с)
  Update: Siamese map matching (каждые 5с)

Использование:
  navigator = EKFNavigator(local_matcher, map_resolution=0.5)
  
  # Инициализация
  navigator.init_from_gps(x, y)  # начальная позиция
  
  # В цикле полёта:
  navigator.predict(velocity, heading, dt=0.1)  # IMU
  if should_correct():
      tile = camera.capture()
      match = navigator.update(tile, search_radius=50)
      if match.confidence > threshold:
          navigator.correct(match.position, match.confidence)
"""

import torch
import numpy as np
from PIL import Image
import cv2
import os

Image.MAX_IMAGE_PIXELS = None

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EKFNavigator:
    """Extended Kalman Filter навигатор с fusion IMU + Siamese map matching."""

    def __init__(self, local_matcher, map_resolution=0.5, imu_noise=0.5, map_noise=2.0):
        """
        Args:
            local_matcher: LocalMatcher instance
            map_resolution: м/пиксель
            imu_noise: шум IMU (м/с² для accel, рад/с для gyro)
            map_noise: шум map matching (м)
        """
        self.matcher = local_matcher
        self.resolution = map_resolution
        self.imu_noise = imu_noise
        self.map_noise = map_noise

        # State: [x, y, vx, vy, heading]
        # x, y — позиция в пикселях карты
        # vx, vy — скорость в пикселях/с
        # heading — направление в радианах
        self.state = np.zeros(5)
        self.cov = np.eye(5) * 100  # начальная неопределённость

        # IMU odometry
        self.last_imu_time = None
        self.odom_x = 0.0
        self.odom_y = 0.0

        # Map matching
        self.last_match_time = None
        self.match_interval = 5.0  # коррекция каждые 5 сек

        # Statistics
        self.match_count = 0
        self.reject_count = 0

    def init_from_gps(self, gps_x, gps_y):
        """
        Инициализация из GPS координат.

        Args:
            gps_x, gps_y: позиция в пикселях карты
        """
        self.state[0] = gps_x  # x
        self.state[1] = gps_y  # y
        self.state[2] = 0.0    # vx
        self.state[3] = 0.0    # vy
        self.state[4] = 0.0    # heading
        self.cov = np.eye(5) * 10  # низкая неопределённость
        self.last_imu_time = None
        self.last_match_time = None
        print(f"[EKF] Initialized at ({gps_x:.0f}, {gps_y:.0f})")

    def init_from_map_match(self, tile, center_x, center_y):
        """
        Инициализация из map matching (первый кадр).

        Args:
            tile: тайл текущего кадра
            center_x, center_y: примерная позиция в пикселях
        """
        # Используем local matcher для точной инициализации
        match_loc, match_score, _, _ = self.matcher.match_local(
            tile, center_x, center_y, radius_m=50
        )
        self.init_from_gps(match_loc[0], match_loc[1])
        print(f"[EKF] Init from map match: ({match_loc[0]:.0f}, {match_loc[1]:.0f}), score={match_score:.3f}")

    def predict(self, velocity, heading, dt):
        """
        Предсказание позиции по IMU odometry.

        Args:
            velocity: скорость м/с
            heading: направление рад (от севера, по часовой)
            dt: время между шагами сек
        """
        if self.last_imu_time is None:
            self.last_imu_time = dt
        else:
            self.last_imu_time += dt

        # heading: 0 = север (y+), π/2 = восток (x+)
        # В пикселях: x = east, y = south
        heading_rad = heading  # уже в радианах

        # Преобразование: velocity (м/с) → pixels/s
        vel_px = velocity / self.resolution

        # Обновление state
        self.state[0] += vel_px * np.cos(heading_rad) * dt  # x
        self.state[1] += vel_px * np.sin(heading_rad) * dt  # y
        self.state[2] = vel_px * np.cos(heading_rad)  # vx
        self.state[3] = vel_px * np.sin(heading_rad)  # vy
        self.state[4] = heading_rad  # heading

        # Увеличение covariance (drift)
        self.cov[0, 0] += 0.1 * dt  # x drift
        self.cov[1, 1] += 0.1 * dt  # y drift
        self.cov[4, 4] += 0.01 * dt  # heading drift

    def update(self, query_tile, search_radius_m=50):
        """
        Map matching с Siamese моделью.

        Args:
            query_tile: тайл текущего кадра (H, W, 3)
            search_radius_m: радиус поиска в метрах

        Returns:
            match: dict с position, confidence или None
        """
        if self.last_match_time is None:
            self.last_match_time = 0
        else:
            self.last_match_time += self.match_interval

        # Предсказанная позиция в пикселях
        pred_x = self.state[0]
        pred_y = self.state[1]

        # Local matching
        match_loc, match_score, candidates, scores = self.matcher.match_local(
            query_tile, pred_x, pred_y, radius_m=search_radius_m
        )

        # Проверка confidence
        if match_score < 0.5:
            self.reject_count += 1
            return None

        # Вычисляем ошибку match
        error_px = np.sqrt((pred_x - match_loc[0])**2 + (pred_y - match_loc[1])**2)
        error_m = error_px * self.resolution

        # Если match близок к предсказанию — принимаем
        if error_m < search_radius_m * 0.5:
            self.match_count += 1
            return {
                'position': match_loc,
                'confidence': match_score,
                'error_m': error_m,
                'candidates': len(candidates)
            }
        else:
            self.reject_count += 1
            return None

    def correct(self, match):
        """
        EKF correction step.

        Args:
            match: dict с position, confidence
        """
        match_x, match_y = match['position']
        confidence = match['confidence']

        # Measurement noise зависит от confidence
        measurement_noise = self.map_noise * (1.0 - confidence)
        R = np.eye(2) * (measurement_noise ** 2)

        # Innovation
        z = np.array([match_x, match_y])
        h = np.array([self.state[0], self.state[1]])
        innovation = z - h

        # Kalman gain
        S = self.cov[:2, :2] + R
        K = self.cov[:2, :2] @ np.linalg.inv(S)

        # Update state
        self.state[:2] += K @ innovation

        # Update covariance
        self.cov[:2, :2] -= K @ S @ K.T

        # Clamp covariance
        self.cov = np.maximum(self.cov, 0.01)

    def get_position(self):
        """Получить текущую позицию в пикселях."""
        return self.state[0], self.state[1]

    def get_velocity(self):
        """Получить текущую скорость м/с."""
        vx = self.state[2] * self.resolution
        vy = self.state[3] * self.resolution
        return np.sqrt(vx**2 + vy**2), self.state[4]

    def get_uncertainty(self):
        """Получить неопределённость позиции в метрах."""
        cov_x = self.cov[0, 0] * self.resolution**2
        cov_y = self.cov[1, 1] * self.resolution**2
        return np.sqrt(cov_x), np.sqrt(cov_y)

    def get_statistics(self):
        """Статистика matchей."""
        total = self.match_count + self.reject_count
        return {
            'matches': self.match_count,
            'rejects': self.reject_count,
            'accept_rate': self.match_count / total * 100 if total > 0 else 0
        }


def test_ekf_navigator():
    """Тест EKF навигатора на симуляции."""
    from local_matcher import LocalMatcher

    print("=" * 60)
    print("EKF NAVIGATOR TEST (SIMULATION)")
    print("=" * 60)

    # Загружаем matcher
    matcher = LocalMatcher(
        model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'siamese_model_kalanchak_v2.pth'),
        map_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map_cache/highres/highres_46.2650_33.3732_z18.png'),
        tile_size=224,
        resolution=0.5
    )

    # Создаём EKF навигатор
    ekf = EKFNavigator(matcher, map_resolution=0.5, imu_noise=0.5, map_noise=2.0)

    # Симуляция полёта: маршрут по карте
    print("\n" + "=" * 60)
    print("SIMULATION: Flight along a route")
    print("=" * 60)

    # Загружаем карту для извлечения тайлов
    map_img = np.array(Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map_cache/highres/highres_46.2650_33.3732_z18.png')).convert('RGB'))
    h, w = map_img.shape[:2]

    # Создаём маршрут: линия из точки A в точку B
    start_cx, start_cy = 5000, 5000  # начальная позиция
    end_cx, end_cy = 10000, 8000     # конечная позиция

    # Генерируем waypoints
    num_waypoints = 200
    route_x = np.linspace(start_cx, end_cx, num_waypoints)
    route_y = np.linspace(start_cy, end_cy, num_waypoints)

    # Скорость: 10 м/с = 20 пикселей/с
    velocity_ms = 10.0
    velocity_px = velocity_ms / 0.5  # 20 px/s

    # EKF навигация
    errors_m = []
    match_accepted = 0
    match_rejected = 0

    # Инициализация
    ekf.init_from_gps(start_cx, start_cy)

    dt = 0.1  # 100мс шаг
    total_time = 0
    match_counter = 0

    for i in range(num_waypoints):
        target_x = route_x[i]
        target_y = route_y[i]

        # Вычисляем heading к цели
        dx = target_x - ekf.state[0]
        dy = target_y - ekf.state[1]
        heading = np.arctan2(dy, dx)

        # IMU predict (с реалистичным шумом)
        # CUAV 7+ Pro IMU: gyro bias ~0.1°/s, accel bias ~0.01g
        gyro_noise = np.radians(0.1)  # 0.1°/s
        accel_noise = 0.01 * 9.81 / 0.5  # 0.01g → м/с² → пиксели/с²
        imu_heading = heading + np.random.normal(0, gyro_noise)
        imu_velocity = velocity_ms + np.random.normal(0, 0.5)  # 0.5 м/с speed noise
        ekf.predict(imu_velocity, imu_heading, dt)

        total_time += dt

        # Map matching каждые 1 секунду (чаще!)
        if total_time % 1.0 < dt:
            match_counter += 1
            # Извлекаем тайл из карты (симуляция камеры)
            cx, cy = int(ekf.state[0]), int(ekf.state[1])
            tile = matcher._get_tile(cx, cy, 224)

            # Map matching
            match = ekf.update(tile, search_radius_m=50)
            if match is not None:
                ekf.correct(match)
                match_accepted += 1
                errors_m.append(match['error_m'])
            else:
                match_rejected += 1

        if (i + 1) % 40 == 0:
            pos_x, pos_y = ekf.get_position()
            target_dist = np.sqrt((target_x - pos_x)**2 + (target_y - pos_y)**2)
            print(f"  [{i+1}/{num_waypoints}] Pos: ({pos_x:.0f}, {pos_y:.0f}), "
                  f"Target dist: {target_dist:.1f}m, "
                  f"Matches: {match_accepted}, Rejects: {match_rejected}")

    # Финальные результаты
    errors_m = np.array(errors_m) if errors_m else np.array([0])
    mean_err = np.mean(errors_m)
    median_err = np.median(errors_m)
    max_err = np.max(errors_m)

    print(f"\n{'='*60}")
    print("EKF NAVIGATION RESULTS")
    print(f"{'='*60}")
    print(f"  Total time: {total_time:.1f} сек")
    print(f"  Distance: {num_waypoints * 10 * 0.1 * 10 / 1000:.1f} км")
    print(f"  Matches accepted: {match_accepted}")
    print(f"  Matches rejected: {match_rejected}")
    print(f"  Accept rate: {match_accepted/(match_accepted+match_rejected)*100:.1f}%")
    print(f"\n  Map matching errors:")
    print(f"    Mean: {mean_err:.1f} м")
    print(f"    Median: {median_err:.1f} м")
    print(f"    Max: {max_err:.1f} м")
    print(f"    < 15м: {np.sum(errors_m < 15)/len(errors_m)*100:.1f}%")
    print(f"    < 30м: {np.sum(errors_m < 30)/len(errors_m)*100:.1f}%")
    print(f"    < 50м: {np.sum(errors_m < 50)/len(errors_m)*100:.1f}%")

    # Сравнение
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Full Search':<15} {'Local (±50m)':<15} {'EKF+Local':<15}")
    print(f"  {'Mean error':<20} {3753.8:<15.1f} {27.5:<15.1f} {mean_err:<15.1f}")
    print(f"  {'Median error':<20} {3727.1:<15.1f} {25.6:<15.1f} {median_err:<15.1f}")
    print(f"  {'< 15m':<20} {4.4:<15.1f} {20.6:<15.1f} {np.sum(errors_m < 15)/len(errors_m)*100:<15.1f}")
    print(f"  {'< 56m':<20} {8.3:<15.1f} {98.0:<15.1f} {np.sum(errors_m < 56)/len(errors_m)*100:<15.1f}")


if __name__ == '__main__':
    test_ekf_navigator()
