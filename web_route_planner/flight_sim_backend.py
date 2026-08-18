"""
flight_sim_backend — симуляция полёта дрона по маршруту для веб-визуализации.

Симулирует полёт крыла (CUAV X7+ Pro) по waypoint'ам из flight_mission.json
с реалистичной динамикой: скорость, развороты, высота.
Состояние доступно через API для отображения в браузере.
"""

import math
import threading
import time
import json
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MISSION_FILE = os.path.join(PROJECT_DIR, 'flight_mission.json')


def latlon_to_xy(lat, lon, ref_lat, ref_lon):
    """Конвертация широты/долготы в локальные метры от точки отсчёта."""
    x = (lon - ref_lon) * 111320.0 * math.cos(math.radians(ref_lat))
    y = (lat - ref_lat) * 110574.0
    return x, y


def xy_to_latlon(x, y, ref_lat, ref_lon):
    """Обратная конвертация метров в широту/долготу."""
    lon = x / (111320.0 * math.cos(math.radians(ref_lat))) + ref_lon
    lat = y / 110574.0 + ref_lat
    return lat, lon


def haversine(lat1, lon1, lat2, lon2):
    """Расстояние между двумя точками в метрах."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class FlightSimulation:
    """
    Потоковая симуляция полёта дрона по маршруту.

    Параметры дрона из drone_config.py:
      - Скорость: 15-33 м/с (крейсерская 22)
      - Высота: 700-1200 м (крейсерская 1000)
      - Мин. радиус разворота: 100 м
      - Камера: FOV 90°, 4K
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.running = False
        self.paused = False
        self.speed_mult = 1.0  # множитель скорости симуляции

        self.waypoints = []  # [(lat, lon), ...]
        self.wp_xy = []      # [(x, y), ...] в метрах
        self.ref_lat = 0.0
        self.ref_lon = 0.0

        # Состояние дрона
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_alt = 1000.0
        self.drone_heading = 0.0  # градусы, 0 = восток, 90 = север
        self.drone_speed = 22.0   # м/с
        self.drone_bank = 0.0     # крен
        self.wp_index = 0
        self.sim_time = 0.0
        self.trail = []           # [(lat, lon), ...] — пройденный путь
        self.trail_max = 2000

        # Параметры
        self.cruise_speed = 22.0
        self.min_turn_radius = 100.0
        self.min_alt = 700.0
        self.max_alt = 1200.0
        self.fov_h = 90.0
        self.cam_w = 3840
        self.cam_h = 2160

        self._load_mission()

    def _load_mission(self):
        """Загружает маршрут из flight_mission.json."""
        if not os.path.exists(MISSION_FILE):
            return
        with open(MISSION_FILE) as f:
            data = json.load(f)
        pts = data.get('points', [])
        if not pts:
            return
        self.waypoints = [(p[0], p[1]) for p in pts]
        self.ref_lat = pts[0][0]
        self.ref_lon = pts[0][1]
        self.wp_xy = [latlon_to_xy(lat, lon, self.ref_lat, self.ref_lon)
                      for lat, lon in self.waypoints]

    def reset(self):
        """Сброс симуляции в начальную точку."""
        with self.lock:
            self._load_mission()
            if self.wp_xy:
                self.drone_x = self.wp_xy[0][0]
                self.drone_y = self.wp_xy[0][1]
            self.drone_alt = 1000.0
            self.drone_heading = 0.0
            self.drone_speed = self.cruise_speed
            self.drone_bank = 0.0
            self.wp_index = 1 if len(self.wp_xy) > 1 else 0
            self.sim_time = 0.0
            self.trail = []

    def start(self):
        """Запуск симуляции."""
        if not self.wp_xy:
            self._load_mission()
        if not self.running:
            self.reset()
            self.running = True
            self.paused = False
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def pause(self):
        """Пауза/возобновление."""
        self.paused = not self.paused

    def stop(self):
        """Остановка симуляции."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None

    def set_speed_mult(self, mult):
        """Установка множителя скорости симуляции (1-10)."""
        self.speed_mult = max(0.1, min(20.0, mult))

    def _run(self):
        """Главный цикл симуляции."""
        dt_real = 0.1  # 100 мс реального времени
        while self.running:
            if not self.paused:
                dt = dt_real * self.speed_mult
                self._step(dt)
            time.sleep(dt_real)

    def _step(self, dt):
        """Один шаг симуляции."""
        with self.lock:
            if self.wp_index >= len(self.wp_xy):
                # Маршрут пройден — кружим
                self.drone_heading = (self.drone_heading + 5.0 * dt) % 360
                self.drone_bank = 15.0
            else:
                tx, ty = self.wp_xy[self.wp_index]
                dx = tx - self.drone_x
                dy = ty - self.drone_y
                dist = math.sqrt(dx * dx + dy * dy)

                # Целевой курс
                target_heading = math.degrees(math.atan2(dy, dx))

                # Плавный разворот (ограничение угловой скорости)
                max_turn_rate = math.degrees(self.cruise_speed / self.min_turn_radius)
                heading_diff = (target_heading - self.drone_heading + 180) % 360 - 180
                max_step = max_turn_rate * dt
                turn = max(-max_step, min(max_step, heading_diff))
                self.drone_heading = (self.drone_heading + turn) % 360

                # Крен пропорционален скорости разворота
                self.drone_bank = max(-30, min(30, (turn / max_step * 30) if max_step > 0 else 0))

                # Высота: плавное изменение между waypoint'ами
                if self.wp_index < len(self.waypoints):
                    target_alt = 700 + (self.wp_index % 3) * 250  # 700/950/1200
                    target_alt = min(self.max_alt, max(self.min_alt, target_alt))
                    alt_diff = target_alt - self.drone_alt
                    self.drone_alt += max(-5.0 * dt, min(5.0 * dt, alt_diff))

                # Переход к следующему waypoint
                if dist < 80:
                    self.wp_index += 1

            # Движение по курсу
            hr = math.radians(self.drone_heading)
            self.drone_x += self.drone_speed * math.cos(hr) * dt
            self.drone_y += self.drone_speed * math.sin(hr) * dt
            self.sim_time += dt

            # Трек пути
            lat, lon = xy_to_latlon(self.drone_x, self.drone_y, self.ref_lat, self.ref_lon)
            self.trail.append((lat, lon))
            if len(self.trail) > self.trail_max:
                self.trail = self.trail[-self.trail_max:]

    def get_state(self):
        """Возвращает текущее состояние для API."""
        with self.lock:
            lat, lon = xy_to_latlon(self.drone_x, self.drone_y, self.ref_lat, self.ref_lon)

            # Расчёт FOV конуса на карте
            footprint = 2 * self.drone_alt * math.tan(math.radians(self.fov_h / 2))
            # Полуширина конуса в метрах
            half_w = footprint / 2
            # Длина конуса (упрощённо — footprint, т.к. камера смотрит вниз)
            depth = footprint * (self.cam_h / self.cam_w)

            hr = math.radians(self.drone_heading)
            # Три точки конуса: нос и два крыла
            nose_x = self.drone_x + (depth / 2) * math.cos(hr)
            nose_y = self.drone_y + (depth / 2) * math.sin(hr)
            tail_x = self.drone_x - (depth / 2) * math.cos(hr)
            tail_y = self.drone_y - (depth / 2) * math.sin(hr)
            # Перпендикуляр
            perp_x = -math.sin(hr)
            perp_y = math.cos(hr)
            left_x = tail_x + half_w * perp_x
            left_y = tail_y + half_w * perp_y
            right_x = tail_x - half_w * perp_x
            right_y = tail_y - half_w * perp_y

            nose_lat, nose_lon = xy_to_latlon(nose_x, nose_y, self.ref_lat, self.ref_lon)
            left_lat, left_lon = xy_to_latlon(left_x, left_y, self.ref_lat, self.ref_lon)
            right_lat, right_lon = xy_to_latlon(right_x, right_y, self.ref_lat, self.ref_lon)

            # Расстояние до следующего waypoint
            dist_to_wp = 0
            if self.wp_index < len(self.wp_xy):
                tx, ty = self.wp_xy[self.wp_index]
                dist_to_wp = math.sqrt((tx - self.drone_x) ** 2 + (ty - self.drone_y) ** 2)

            # Пройденный путь
            total_dist = 0
            for i in range(len(self.wp_xy) - 1):
                x1, y1 = self.wp_xy[i]
                x2, y2 = self.wp_xy[i + 1]
                total_dist += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            remaining = 0
            for i in range(self.wp_index, len(self.wp_xy) - 1):
                x1, y1 = self.wp_xy[i]
                x2, y2 = self.wp_xy[i + 1]
                remaining += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            remaining += dist_to_wp
            progress = max(0, min(100, (1 - remaining / total_dist) * 100)) if total_dist > 0 else 0

            return {
                'running': self.running and not self.paused,
                'paused': self.paused,
                'drone': {
                    'lat': lat,
                    'lon': lon,
                    'alt': round(self.drone_alt, 1),
                    'heading': round(self.drone_heading, 1),
                    'speed': round(self.drone_speed, 1),
                    'bank': round(self.drone_bank, 1),
                },
                'waypoint': {
                    'index': self.wp_index,
                    'total': len(self.waypoints),
                    'dist_to_wp': round(dist_to_wp, 1),
                    'lat': self.waypoints[self.wp_index][0] if self.wp_index < len(self.waypoints) else None,
                    'lon': self.waypoints[self.wp_index][1] if self.wp_index < len(self.waypoints) else None,
                },
                'fov_cone': [[nose_lat, nose_lon], [left_lat, left_lon], [right_lat, right_lon]],
                'trail': self.trail[-500:],
                'progress': round(progress, 1),
                'sim_time': round(self.sim_time, 1),
                'speed_mult': self.speed_mult,
            }


# Глобальный экземпляр
sim = FlightSimulation()
