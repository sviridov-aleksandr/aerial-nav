"""
GPS-Denied Navigation — Тест на реальном маршруте.

Преобразование GPS → пиксели карты:
  Карта: highres_46.2650_33.3732_z18.png
  Размер: 17920×17920 px
  Центр: (46.2650, 33.3732)
  Разрешение: 0.5 м/px

  x = (lon - 33.3732) * 17920 / 0.1164 + 8960
  y = (46.2650 - lat) * 17920 / 0.0807 + 8960
"""

import torch
import numpy as np
from PIL import Image
import cv2
import os

Image.MAX_IMAGE_PIXELS = None

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Координаты маршрута
ROUTE_GPS = [
    (46.264929, 33.372986),  # Точка взлёта
    (46.279537, 33.371246),  # 1
    (46.277625, 33.352702),  # 2
    (46.259422, 33.322994),  # 3
    (46.252893, 33.298535),  # 4
    (46.262919, 33.286605),  # 5
    (46.281208, 33.274915),  # 6
]

# Параметры карты
MAP_PATH = '/home/alex/aerial-nav/map_cache/highres/highres_46.2650_33.3732_z18.png'
MAP_CENTER_LAT = 46.2650
MAP_CENTER_LON = 33.3732
MAP_SIZE = 17920  # px
MAP_RESOLUTION = 0.5  # m/px

# Преобразование GPS → пиксели
# 1° широты ≈ 111320 м
# 1° долготы ≈ 111320 * cos(lat) м
def gps_to_pixels(lat, lon):
    """Преобразование GPS координат в пиксели карты."""
    # meters per degree
    lat_m_per_deg = 111320.0
    lon_m_per_deg = 111320.0 * np.cos(np.radians(lat))
    
    # meters from center
    dy = (MAP_CENTER_LAT - lat) * lat_m_per_deg  # y increases south
    dx = (lon - MAP_CENTER_LON) * lon_m_per_deg  # x increases east
    
    # pixels from center
    center_px = MAP_SIZE // 2
    x = int(center_px + dx / MAP_RESOLUTION)
    y = int(center_px - dy / MAP_RESOLUTION)  # y axis inverted
    
    return x, y


def interpolate_route(gps_points, step_m=100):
    """
    Интерполяция маршрута с шагом step_m метров.
    
    Returns:
        route: [(x_px, y_px), ...]
        distances: [distance_from_start_m, ...]
    """
    route = []
    distances = []
    total_dist = 0
    
    for i in range(len(gps_points) - 1):
        lat1, lon1 = gps_points[i]
        lat2, lon2 = gps_points[i + 1]
        
        x1, y1 = gps_to_pixels(lat1, lon1)
        x2, y2 = gps_to_pixels(lat2, lon2)
        
        # Distance between points
        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx**2 + dy**2) * MAP_RESOLUTION
        
        # Interpolate
        num_steps = int(dist / step_m)
        for j in range(num_steps + 1):
            t = j / num_steps
            x = x1 + dx * t
            y = y1 + dy * t
            route.append((x, y))
            distances.append(total_dist + dist * t)
        
        total_dist += dist
    
    return route, distances


def test_route():
    """Тест навигации на реальном маршруте."""
    from local_matcher import LocalMatcher
    from ekf_navigator import EKFNavigator
    
    print("=" * 70)
    print("GPS-DENIED NAVIGATION — REAL ROUTE TEST")
    print("=" * 70)
    
    # 1. Показать маршрут
    print("\n[1/4] Route:")
    print(f"  {'Point':<10} {'Lat':<15} {'Lon':<15} {'X(px)':<10} {'Y(px)':<10}")
    print(f"  {'-'*60}")
    
    for i, (lat, lon) in enumerate(ROUTE_GPS):
        x, y = gps_to_pixels(lat, lon)
        print(f"  {i:<10} {lat:<15.6f} {lon:<15.6f} {x:<10} {y:<10}")
    
    # 2. Интерполяция
    print("\n[2/4] Interpolating route...")
    route, distances = interpolate_route(ROUTE_GPS, step_m=50)
    total_dist = distances[-1]
    print(f"  Total distance: {total_dist/1000:.1f} км")
    print(f"  Waypoints: {len(route)}")
    
    # 3. Загрузка компонентов
    print("\n[3/4] Loading components...")
    matcher = LocalMatcher(
        model_path='/home/alex/aerial-nav/siamese_model_kalanchak_v2.pth',
        map_path=MAP_PATH,
        tile_size=224,
        resolution=MAP_RESOLUTION
    )
    
    ekf = EKFNavigator(matcher, map_resolution=MAP_RESOLUTION, imu_noise=0.5, map_noise=2.0)
    
    # 4. Симуляция
    print("\n[4/4] Simulating flight...")
    
    # Инициализация
    start_x, start_y = gps_to_pixels(ROUTE_GPS[0][0], ROUTE_GPS[0][1])
    ekf.init_from_gps(start_x, start_y)
    
    velocity_ms = 10.0  # 36 км/ч
    dt = 0.1
    total_time = 0
    match_accepted = 0
    match_rejected = 0
    errors_m = []
    waypoints_reached = 0
    
    # Для каждой waypoint-цели
    waypoint_idx = 1
    target_x, target_y = gps_to_pixels(ROUTE_GPS[waypoint_idx][0], ROUTE_GPS[waypoint_idx][1])
    
    for i, (rx, ry) in enumerate(route):
        # Heading к цели
        dx = target_x - ekf.state[0]
        dy = target_y - ekf.state[1]
        heading = np.arctan2(dy, dx)
        
        # IMU predict
        gyro_noise = np.radians(0.1)
        imu_heading = heading + np.random.normal(0, gyro_noise)
        imu_velocity = velocity_ms + np.random.normal(0, 0.5)
        ekf.predict(imu_velocity, imu_heading, dt)
        
        total_time += dt
        
        # Map matching каждые 1 сек
        if total_time % 1.0 < dt:
            cx, cy = int(ekf.state[0]), int(ekf.state[1])
            tile = matcher._get_tile(cx, cy, 224)
            match = ekf.update(tile, search_radius_m=50)
            if match is not None:
                ekf.correct(match)
                match_accepted += 1
                errors_m.append(match['error_m'])
            else:
                match_rejected += 1
        
        # Проверка достижения waypoint
        if waypoint_idx < len(ROUTE_GPS) - 1:
            dist_to_waypoint = np.sqrt((target_x - ekf.state[0])**2 + (target_y - ekf.state[1])**2) * MAP_RESOLUTION
            if dist_to_waypoint < 100:  # 100м
                waypoints_reached += 1
                waypoint_idx += 1
                target_x, target_y = gps_to_pixels(ROUTE_GPS[waypoint_idx][0], ROUTE_GPS[waypoint_idx][1])
        
        if (i + 1) % 500 == 0:
            pos_x, pos_y = ekf.get_position()
            dist_to_target = np.sqrt((target_x - pos_x)**2 + (target_y - pos_y)**2) * MAP_RESOLUTION
            print(f"  [{i+1}/{len(route)}] Pos: ({pos_x:.0f}, {pos_y:.0f}), "
                  f"Dist to target: {dist_to_target:.0f}m, "
                  f"Matches: {match_accepted}, Rejects: {match_rejected}, "
                  f"Waypoints: {waypoints_reached}/{len(ROUTE_GPS)-1}")
    
    # 5. Результаты
    errors_m = np.array(errors_m) if errors_m else np.array([0])
    mean_err = np.mean(errors_m)
    median_err = np.median(errors_m)
    max_err = np.max(errors_m)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Total distance: {total_dist/1000:.1f} км")
    print(f"  Total time: {total_time:.1f} сек ({total_time/60:.1f} мин)")
    print(f"  Waypoints reached: {waypoints_reached}/{len(ROUTE_GPS)-1}")
    print(f"  Matches: {match_accepted} accepted, {match_rejected} rejected")
    print(f"  Accept rate: {match_accepted/(match_accepted+match_rejected)*100:.1f}%")
    print(f"\n  Map matching errors:")
    print(f"    Mean: {mean_err:.1f} м")
    print(f"    Median: {median_err:.1f} м")
    print(f"    Max: {max_err:.1f} м")
    print(f"    < 15м: {np.sum(errors_m < 15)/len(errors_m)*100:.1f}%")
    print(f"    < 30м: {np.sum(errors_m < 30)/len(errors_m)*100:.1f}%")
    print(f"    < 50м: {np.sum(errors_m < 50)/len(errors_m)*100:.1f}%")
    
    # Итог
    print(f"\n{'='*70}")
    if median_err < 15:
        print(f"  ✓ ЦЕЛЬ ДОСТИГНУТА: median error {median_err:.1f}м < 15м")
    else:
        print(f"  ✗ ЦЕЛЬ НЕ ДОСТИГНУТА: median error {median_err:.1f}м >= 15м")
    print(f"\n{'='*70}")
    
    return median_err < 15


if __name__ == '__main__':
    test_route()
