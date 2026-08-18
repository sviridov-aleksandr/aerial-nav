#!/usr/bin/env python3
"""
mavlink_bridge.py — мост между полётным контроллером CUAV X7+ Pro и
навигационным пайплайном (Siamese + EKF) на Jetson Orin Nano.

Архитектура:
  ┌─────────┐  MAVLink2   ┌──────────────┐  кадр   ┌────────────┐
  │ CUAV X7+ │◄──── UART ──►│ mavlink_     │◄───────│ OpenIPC RTSP │
  │ (FC)     │  921600 бод  │ bridge.py    │ 5 Гц   │ (камера)     │
  └─────────┘              └──────┬───────┘         └────────────┘
      ▲                           │
      │ VISION_POSITION_ESTIMATE  │ EKFNavigator
      │                           ▼
      │                    ┌──────────────┐
      │                    │ Siamese model │
      │                    │ + индекс карты│
      │                    └──────────────┘

Режимы:
  - SITL: UDP подключение к ArduPilot Simulator (127.0.0.1:14540)
  - REAL: UART /dev/ttyTHS1 (или другой), 921600 бод

Поток данных (5 Гц):
  1. FC → bridge: ATTITUDE (рыскание), VFR_HUD (скорость), GLOBAL_POSITION_INT (GPS init)
  2. Камера → Siamese: кадр → embedding → локальный поиск по карте
  3. EKF predict (IMU/скорость) → EKF update (map matching)
  4. bridge → FC: VISION_POSITION_ESTIMATE (x,y,z + attitude + covariance)

Запуск (SITL):
  /home/alex/my_project_env/bin/python mavlink_bridge.py --sitl

Запуск (реальное железо):
  /home/alex/my_project_env/bin/python mavlink_bridge.py --device /dev/ttyTHS1 --baud 921600
"""

import os
import sys
import time
import math
import argparse
import threading
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field

# pymavlink
from pymavlink import mavutil

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from drone_config import DroneConfig


@dataclass
class TelemetryState:
    """Текущая телеметрия от FC."""
    roll: float = 0.0       # рад
    pitch: float = 0.0      # рад
    yaw: float = 0.0        # рад (рыскание, от севера)
    speed_ms: float = 0.0   # м/с
    altitude_m: float = 0.0 # м (барометр)
    gps_lat: float = 0.0    # град (если есть GPS)
    gps_lon: float = 0.0    # град
    gps_fix: bool = False
    timestamp: float = 0.0


@dataclass
class NavigationConfig:
    """Конфигурация навигационного пайплайна."""
    # MAVLink
    device: str = "udp:127.0.0.1:14540"
    baud: int = 921600
    # Навигация
    map_resolution: float = 0.206  # м/px (zoom 19 Google)
    search_radius_m: float = 50.0  # радиус локального поиска
    match_confidence_threshold: float = 0.5
    # Частоты
    nav_rate_hz: float = 5.0       # частота цикла навигации
    predict_rate_hz: float = 5.0   # частота EKF predict (IMU)
    # Модель
    model_path: str = "region_model.pth"
    map_path: str = "map_cache/antiuav_route_strip.tif"
    # Координаты карты (центр)
    map_origin_lat: float = 46.34
    map_origin_lon: float = 31.95
    # Камера
    camera_rtsp_url: str = "rtsp://192.168.1.10:8554/stream"
    camera_width: int = 3840
    camera_height: int = 2160
    tile_size: int = 512


class MAVLinkBridge:
    """
    Мост MAVLink между FC и навигационным пайплайном.

    Отвечает за:
    - Приём телеметрии (ATTITUDE, VFR_HUD, GLOBAL_POSITION_INT)
    - Отправку позиции (VISION_POSITION_ESTIMATE)
    - Синхронизацию времени с FC
    """

    def __init__(self, config: NavigationConfig):
        self.config = config
        self.telemetry = TelemetryState()
        self.connected = False
        self._lock = threading.Lock()

        # Статистика
        self.msg_count = 0
        self.vision_sent = 0
        self.last_heartbeat = 0

    def _auto_detect_port(self) -> str:
        """
        Автодетект USB-порта полётного контроллера CUAV.

        Ищет /dev/ttyACM* и /dev/ttyUSB*, проверяет MAVLink heartbeat
        и наличие телеметрии (ATTITUDE/VFR_HUD). Возвращает лучший порт.
        """
        import glob

        candidates = sorted(
            glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')
        )
        if not candidates:
            raise RuntimeError("[MAVLink] Не найдено USB-портов (/dev/ttyACM*, /dev/ttyUSB*)")

        print(f"[MAVLink] Поиск FC среди портов: {candidates}")

        best_port = None
        best_score = -1

        for port in candidates:
            try:
                conn = mavutil.mavlink_connection(port, baud=115200, dialect="ardupilotmega")
                msg = conn.wait_heartbeat(timeout=3)
                if msg is None:
                    conn.close()
                    print(f"[MAVLink] {port}: нет heartbeat")
                    continue

                # Запрашиваем потоки и считаем телеметрию
                conn.mav.request_data_stream_send(
                    msg.get_srcSystem(), msg.get_srcComponent(),
                    mavutil.mavlink.MAV_DATA_STREAM_EXTRA1, 5, 1)
                conn.mav.request_data_stream_send(
                    msg.get_srcSystem(), msg.get_srcComponent(),
                    mavutil.mavlink.MAV_DATA_STREAM_EXTRA2, 5, 1)

                got = {}
                t0 = time.time()
                while time.time() - t0 < 4:
                    m = conn.recv_match(
                        type=['ATTITUDE', 'VFR_HUD', 'GLOBAL_POSITION_INT'],
                        blocking=True, timeout=1)
                    if m:
                        got[m.get_type()] = got.get(m.get_type(), 0) + 1
                    if len(got) >= 3:
                        break

                score = sum(got.values())
                print(f"[MAVLink] {port}: heartbeat OK, телеметрия {got} (score={score})")
                conn.close()

                if score > best_score:
                    best_score = score
                    best_port = port

            except Exception as e:
                print(f"[MAVLink] {port}: ошибка {e}")

        if best_port is None:
            raise RuntimeError("[MAVLink] Не найден порт с MAVLink heartbeat")

        print(f"[MAVLink] Выбран порт: {best_port} (score={best_score})")
        return best_port

    def connect(self):
        """Подключение к FC (UART или UDP для SITL)."""
        # Автодетект порта, если не указан явно
        if not self.config.device.startswith("udp") and not self.config.device.startswith("/dev/"):
            self.config.device = self._auto_detect_port()

        print(f"[MAVLink] Подключение к {self.config.device}...")

        if self.config.device.startswith("udp"):
            self.conn = mavutil.mavlink_connection(
                self.config.device,
                input=True,
                dialect="ardupilotmega"
            )
        else:
            self.conn = mavutil.mavlink_connection(
                self.config.device,
                baud=self.config.baud,
                dialect="ardupilotmega"
            )

        # Ждём heartbeat
        print("[MAVLink] Ожидание heartbeat...")
        msg = self.conn.wait_heartbeat(timeout=10)
        if msg is None:
            raise RuntimeError("[MAVLink] Нет heartbeat — FC не отвечает")

        self.system_id = msg.get_srcSystem()
        self.component_id = msg.get_srcComponent()
        print(f"[MAVLink] Подключено: sys={self.system_id}, comp={self.component_id}")
        print(f"[MAVLink] Тип: {msg.type}, Autopilot: {msg.autopilot}")
        self.connected = True
        self.last_heartbeat = time.time()

    def start_telemetry_thread(self):
        """Запуск фонового потока чтения телеметрии."""
        self._running = True
        self._thread = threading.Thread(target=self._telemetry_loop, daemon=True)
        self._thread.start()
        print("[MAVLink] Поток телеметрии запущен")

    def _telemetry_loop(self):
        """Фоновый цикл чтения телеметрии (неблокирующий)."""
        while self._running:
            msg = self.conn.recv_match(
                type=['ATTITUDE', 'VFR_HUD', 'GLOBAL_POSITION_INT',
                      'HEARTBEAT', 'SYS_STATUS'],
                blocking=True,
                timeout=0.1
            )
            if msg is None:
                continue

            self.msg_count += 1

            with self._lock:
                if msg.get_type() == 'ATTITUDE':
                    self.telemetry.roll = msg.roll
                    self.telemetry.pitch = msg.pitch
                    self.telemetry.yaw = msg.yaw
                    self.telemetry.timestamp = time.time()

                elif msg.get_type() == 'VFR_HUD':
                    self.telemetry.speed_ms = msg.groundspeed
                    self.telemetry.altitude_m = msg.alt

                elif msg.get_type() == 'GLOBAL_POSITION_INT':
                    if msg.lat != 0 and msg.lon != 0:
                        self.telemetry.gps_lat = msg.lat * 1e-7
                        self.telemetry.gps_lon = msg.lon * 1e-7
                        self.telemetry.gps_fix = True

                elif msg.get_type() == 'HEARTBEAT':
                    self.last_heartbeat = time.time()

    def get_telemetry(self) -> TelemetryState:
        """Получить текущую телеметрию (потокобезопасно)."""
        with self._lock:
            return TelemetryState(
                roll=self.telemetry.roll,
                pitch=self.telemetry.pitch,
                yaw=self.telemetry.yaw,
                speed_ms=self.telemetry.speed_ms,
                altitude_m=self.telemetry.altitude_m,
                gps_lat=self.telemetry.gps_lat,
                gps_lon=self.telemetry.gps_lon,
                gps_fix=self.telemetry.gps_fix,
                timestamp=self.telemetry.timestamp
            )

    def send_vision_position(self, x_m: float, y_m: float, z_m: float,
                              roll: float, pitch: float, yaw: float,
                              covariance: Optional[np.ndarray] = None):
        """
        Отправка позиции в FC через VISION_POSITION_ESTIMATE.

        Args:
            x_m, y_m, z_m: позиция в метрах (NED: x=North, y=East, z=Down)
            roll, pitch, yaw: ориентация в радианах
            covariance: 21-элементный массив ковариации (или None)
        """
        if not self.connected:
            return

        if covariance is None:
            # Ковариация по умолчанию: 1 м по позиции, 0.1 рад по ориентации
            covariance = np.zeros(21)
            covariance[0] = 1.0   # var(x)
            covariance[6] = 1.0   # var(y)
            covariance[12] = 1.0  # var(z)
            covariance[15] = 0.1  # var(roll)
            covariance[18] = 0.1  # var(pitch)
            covariance[20] = 0.1  # var(yaw)

        cov_flat = [float(c) for c in covariance[:21]]

        self.conn.mav.vision_position_estimate_send(
            int(time.time() * 1e6),  # usec
            x_m, y_m, z_m,
            roll, pitch, yaw,
            cov_flat
        )
        self.vision_sent += 1

    def send_heartbeat(self):
        """Отправка heartbeat от компаньона (тип: onboard computer)."""
        self.conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            0,  # custom_mode
            mavutil.mavlink.MAV_STATE_ACTIVE
        )

    def request_data_streams(self):
        """Запрос потоков телеметрии с нужной частотой."""
        streams = [
            (mavutil.mavlink.MAV_DATA_STREAM_EXTRA1, 10),   # ATTITUDE 10 Гц
            (mavutil.mavlink.MAV_DATA_STREAM_EXTRA2, 5),    # VFR_HUD 5 Гц
            (mavutil.mavlink.MAV_DATA_STREAM_EXTRA3, 2),    # AHRS 2 Гц
            (mavutil.mavlink.MAV_DATA_STREAM_POSITION, 5),  # GPS 5 Гц
        ]

        for stream_id, rate in streams:
            self.conn.mav.request_data_stream_send(
                self.system_id,
                self.component_id,
                stream_id,
                rate,
                1  # start
            )
        print(f"[MAVLink] Запрошены потоки: {streams}")

        # Отправляем heartbeat компаньона
        self.send_heartbeat()

    def stop(self):
        """Остановка моста."""
        self._running = False
        if hasattr(self, '_thread'):
            self._thread.join(timeout=2)
        self.connected = False
        print("[MAVLink] Мост остановлен")


class NavigationPipeline:
    """
    Полный навигационный конвейер: камера → Siamese → EKF → MAVLink.

    Объединяет:
    - MAVLinkBridge (телеметрия от FC, отправка позиции)
    - EKFNavigator (fusion IMU + map matching)
    - LocalMatcher (Siamese embedding + локальный поиск)
    - Захват кадра с камеры (RTSP или симуляция)
    """

    def __init__(self, config: NavigationConfig):
        self.config = config
        self.bridge = MAVLinkBridge(config)
        self.ekf = None
        self.matcher = None
        self.cap = None
        self.running = False

        # Статистика
        self.frames_processed = 0
        self.matches_accepted = 0
        self.matches_rejected = 0
        self.last_position = None

    def initialize(self, use_simulated_camera: bool = False):
        """
        Инициализация компонентов пайплайна.

        Args:
            use_simulated_camera: если True, кадры берутся из карты (тест)
        """
        # 1. MAVLink
        self.bridge.connect()
        self.bridge.request_data_streams()
        self.bridge.start_telemetry_thread()

        # 2. Siamese matcher
        from local_matcher import LocalMatcher
        model_path = os.path.join(PROJECT_DIR, self.config.model_path)
        map_path = os.path.join(PROJECT_DIR, self.config.map_path)

        print(f"[Pipeline] Загрузка модели: {model_path}")
        self.matcher = LocalMatcher(
            model_path=model_path,
            map_path=map_path,
            tile_size=self.config.tile_size,
            resolution=self.config.map_resolution
        )

        # 3. EKF Navigator
        from ekf_navigator import EKFNavigator
        self.ekf = EKFNavigator(
            local_matcher=self.matcher,
            map_resolution=self.config.map_resolution,
            imu_noise=0.5,
            map_noise=2.0
        )

        # 4. Камера
        self.use_simulated_camera = use_simulated_camera
        if not use_simulated_camera:
            import cv2
            self.cap = cv2.VideoCapture(self.config.camera_rtsp_url, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                raise RuntimeError(f"[Pipeline] Не удалось открыть RTSP: {self.config.camera_rtsp_url}")
            print(f"[Pipeline] Камера подключена: {self.config.camera_rtsp_url}")

        # 5. Инициализация позиции из GPS (если есть)
        time.sleep(2)  # ждём первую телеметрию
        tel = self.bridge.get_telemetry()
        if tel.gps_fix:
            init_x, init_y = self._gps_to_map_px(tel.gps_lat, tel.gps_lon)
            self.ekf.init_from_gps(init_x, init_y)
            print(f"[Pipeline] Инициализация из GPS: ({init_x:.0f}, {init_y:.0f}) px")
        else:
            print("[Pipeline] ⚠️ Нет GPS — ожидается инициализация по первому кадру")

        print("[Pipeline] Инициализация завершена")

    def _gps_to_map_px(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Конвертация GPS → пиксели карты.

        Args:
            lat, lon: широта/долгота

        Returns:
            (x_px, y_px) — позиция в пикселях карты
        """
        # Расчёт через гео-привязку карты
        # Для zoom 19: разрешение ~0.206 м/px
        # Используем простую проекцию (плоская аппроксимация для малых площадей)
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(math.radians(self.config.map_origin_lat))

        dx_m = (lon - self.config.map_origin_lon) * meters_per_deg_lon
        dy_m = (lat - self.config.map_origin_lat) * meters_per_deg_lat

        x_px = dx_m / self.config.map_resolution
        y_px = dy_m / self.config.map_resolution

        return x_px, y_px

    def _map_px_to_gps(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """Конвертация пиксели карты → GPS."""
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(math.radians(self.config.map_origin_lat))

        dx_m = x_px * self.config.map_resolution
        dy_m = y_px * self.config.map_resolution

        lon = self.config.map_origin_lon + dx_m / meters_per_deg_lon
        lat = self.config.map_origin_lat + dy_m / meters_per_deg_lat

        return lat, lon

    def _capture_frame(self) -> Optional[np.ndarray]:
        """
        Захват кадра с камеры.

        Returns:
            кадр (H, W, 3) RGB или None
        """
        if self.use_simulated_camera:
            # Симуляция: берём тайл из карты вокруг текущей позиции EKF
            x, y = self.ekf.get_position()
            tile = self.matcher._get_tile(int(x), int(y), self.config.tile_size)
            return tile

        import cv2
        ret, frame = self.cap.read()
        if not ret:
            return None

        # BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Центральная вырезка 512×512 (имитация nadir)
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        half = self.config.tile_size // 2
        crop = frame[cy-half:cy+half, cx-half:cx+half]

        return crop

    def _rotate_frame_by_yaw(self, frame: np.ndarray, yaw_rad: float) -> np.ndarray:
        """
        Поворот кадра на угол рыскания (компенсация ориентации дрона).

        Использует reflect-padding для устранения чёрных краёв.
        """
        from PIL import Image, ImageOps
        angle_deg = math.degrees(yaw_rad)

        img = Image.fromarray(frame)
        w, h = img.size
        diag = int(math.ceil(math.sqrt(w**2 + h**2)))
        pad_w = (diag - w) // 2
        pad_h = (diag - h) // 2
        img = ImageOps.expand(img, border=(pad_w, pad_h), fill=0)
        img = img.rotate(angle_deg, resample=Image.BILINEAR, fillcolor=0)
        cx, cy = img.size[0] // 2, img.size[1] // 2
        img = img.crop((cx - w // 2, cy - h // 2, cx + w - w // 2, cy + h - h // 2))
        return np.array(img)

    def run(self):
        """Главный цикл навигации (5 Гц)."""
        self.running = True
        dt = 1.0 / self.config.nav_rate_hz
        match_interval = 1.0 / self.config.nav_rate_hz
        last_match_time = 0

        print(f"[Pipeline] Запуск навигации ({self.config.nav_rate_hz} Гц)")
        print(f"[Pipeline] Ctrl+C для остановки")

        try:
            while self.running:
                loop_start = time.time()

                # 1. Получаем телеметрию
                tel = self.bridge.get_telemetry()

                # 2. EKF predict (по скорости и heading из FC)
                if tel.timestamp > 0:
                    self.ekf.predict(tel.speed_ms, tel.yaw, dt)

                # 3. Map matching (каждый цикл или реже)
                now = time.time()
                if now - last_match_time >= match_interval:
                    frame = self._capture_frame()
                    if frame is not None and frame.max() > 0:
                        # Компенсируем рыскание
                        frame_aligned = self._rotate_frame_by_yaw(frame, tel.yaw)

                        # Локальный поиск
                        pred_x, pred_y = self.ekf.get_position()
                        match = self.ekf.update(frame_aligned, self.config.search_radius_m)

                        if match is not None:
                            self.ekf.correct(match)
                            self.matches_accepted += 1
                        else:
                            self.matches_rejected += 1

                    last_match_time = now
                    self.frames_processed += 1

                # 4. Отправка позиции в FC
                x_px, y_px = self.ekf.get_position()
                x_m = x_px * self.config.map_resolution
                y_m = y_px * self.config.map_resolution
                z_m = -tel.altitude_m  # NED: z вниз

                # Ковариация из EKF
                unc_x, unc_y = self.ekf.get_uncertainty()
                cov = np.zeros(21)
                cov[0] = unc_x ** 2
                cov[6] = unc_y ** 2
                cov[12] = 1.0
                cov[15] = 0.01
                cov[18] = 0.01
                cov[20] = 0.01

                self.bridge.send_vision_position(
                    x_m, y_m, z_m,
                    tel.roll, tel.pitch, tel.yaw,
                    covariance=cov
                )

                self.last_position = (x_m, y_m, z_m)

                # 5. Логирование (каждые 5 секунд)
                if self.frames_processed % 25 == 0 and self.frames_processed > 0:
                    lat, lon = self._map_px_to_gps(x_px, y_px)
                    speed, heading = self.ekf.get_velocity()
                    stats = self.ekf.get_statistics()
                    print(f"[Nav] Позиция: ({x_m:.1f}, {y_m:.1f}) м | "
                          f"GPS: {lat:.5f}, {lon:.5f} | "
                          f"Скорость: {speed:.1f} м/с | "
                          f"Unc: ±{unc_x:.1f} м | "
                          f"Match: {stats['accept_rate']:.0f}%")

                # 6. Heartbeat компаньона (раз в секунду)
                if int(now) != int(self.bridge.last_heartbeat):
                    self.bridge.send_heartbeat()

                # 7. Соблюдение частоты
                elapsed = time.time() - loop_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[Pipeline] Остановка по Ctrl+C")
        finally:
            self.stop()

    def stop(self):
        """Остановка пайплайна."""
        self.running = False
        self.bridge.stop()
        if self.cap is not None:
            self.cap.release()
        print(f"[Pipeline] Статистика: кадров={self.frames_processed}, "
              f"match принят={self.matches_accepted}, "
              f"отклонён={self.matches_rejected}")


def main():
    parser = argparse.ArgumentParser(
        description="MAVLink-мост: Siamese навигация → CUAV X7+ Pro"
    )
    parser.add_argument('--sitl', action='store_true',
                        help='Режим SITL (UDP 127.0.0.1:14540)')
    parser.add_argument('--device', type=str, default=None,
                        help='UART устройство (например /dev/ttyACM0). '
                             'Если не указано — автодетект USB-порта CUAV')
    parser.add_argument('--baud', type=int, default=115200,
                        help='Скорость UART (USB CDC: 115200, телеметрия: 57600)')
    parser.add_argument('--model', type=str, default='region_model.pth',
                        help='Путь к модели')
    parser.add_argument('--map', type=str,
                        default='map_cache/antiuav_route_strip.tif',
                        help='Путь к карте GeoTIFF')
    parser.add_argument('--sim-camera', action='store_true',
                        help='Симулированная камера (кадры из карты)')
    parser.add_argument('--rtsp', type=str,
                        default='rtsp://192.168.1.10:8554/stream',
                        help='URL RTSP камеры')
    args = parser.parse_args()

    # Конфигурация
    config = NavigationConfig(
        model_path=args.model,
        map_path=args.map,
        camera_rtsp_url=args.rtsp,
    )

    if args.sitl:
        config.device = "udp:127.0.0.1:14540"
    elif args.device:
        config.device = args.device
        config.baud = args.baud
    else:
        # Автодетект USB-порта CUAV
        config.device = "auto"

    # Запуск
    pipeline = NavigationPipeline(config)
    pipeline.initialize(use_simulated_camera=args.sim_camera)
    pipeline.run()


if __name__ == '__main__':
    main()
