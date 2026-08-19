#!/usr/bin/env python3
"""
sim_flight_fc.py — симуляция полёта на реальном полётном контроллере.

Генерирует кадры камеры из карты вдоль маршрута, определяет позицию
через Siamese-модель и отправляет координаты в FC (CUAV X7+ Pro)
через MAVLink как VISION_POSITION_ESTIMATE. QGroundControl подключается
к FC и отображает движение дрона по маршруту.

Схема:
  Карта (GeoTIFF) → кадры вдоль маршрута → Siamese matching →
  координаты (lat/lon) → MAVLink VISION_POSITION_ESTIMATE →
  FC (ArduPilot EKF3) → QGroundControl

Безопасность:
  - FC на столе, пропеллеры сняты
  - Режим FC: MANUAL/STABILIZE (не AUTO)
  - Мы только отправляем позицию, не управляем полётом

Запуск:
  /home/alex/my_project_env/bin/python sim_flight_fc.py --altitude 1000 --speed 22

  --altitude  высота полёта (м), по умолчанию 1000
  --speed     скорость (м/с), по умолчанию 22
  --model     путь к модели (region_model.pth)
  --device    UART устройство (по умолчанию автодетект)
  --route-reverse  обратный маршрут
"""

import os
import sys
import time
import math
import argparse
import threading
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from PIL import Image
from pymavlink import mavutil

Image.MAX_IMAGE_PIXELS = None

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from siamese_network import AerialFeatureExtractor

# --- Константы ---
MAP_PATH = os.path.join(PROJECT_DIR, 'map_cache/antiuav_route_strip.tif')
COORDS_PATH = os.path.join(PROJECT_DIR, 'training_data/route_dataset/positive_coords.npy')
RESOLUTION = 0.206  # м/px
TILE_SIZE = 512
CAM_W = 3840
CAM_FOV_H = 90.0

# Гео-привязка карты (центр маршрута)
MAP_ORIGIN_LAT = 46.265
MAP_ORIGIN_LON = 33.250

# Параметры FC
FC_BAUD = 115200
NAV_RATE_HZ = 5.0


def load_model(model_path):
    """Загружает Siamese-модель."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AerialFeatureExtractor(embedding_dim=256).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        epoch = ckpt.get('epoch', '?')
        print(f"[Sim] Модель: {model_path} (epoch={epoch})")
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model, device


def normalize(arr):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    out = arr.astype(np.float32) / 255.0
    out = (out - mean) / std
    return out


def read_tile(src, x, y, size=512):
    win = Window(int(x), int(y), size, size)
    try:
        data = src.read(window=win)
        tile = np.transpose(data, (1, 2, 0))
        if tile.shape[2] == 4:
            tile = tile[:, :, :3]
        h, w = tile.shape[:2]
        if h < size or w < size:
            padded = np.zeros((size, size, 3), dtype=np.uint8)
            padded[:h, :w] = tile
            tile = padded
        return tile
    except Exception:
        return np.zeros((size, size, 3), dtype=np.uint8)


def camera_frame_from_map(src, center_px, altitude, tile_size=512):
    """Имитация кадра камеры с высоты altitude м."""
    footprint_w = 2 * altitude * math.tan(math.radians(CAM_FOV_H / 2))
    gsd_cam = footprint_w / CAM_W
    patch_m = gsd_cam * tile_size
    patch_px_map = int(patch_m / RESOLUTION)

    cx, cy = center_px
    x1 = int(cx - patch_px_map / 2)
    y1 = int(cy - patch_px_map / 2)

    win = Window(x1, y1, patch_px_map, patch_px_map)
    try:
        data = src.read(window=win)
        map_crop = np.transpose(data, (1, 2, 0))
        if map_crop.shape[2] == 4:
            map_crop = map_crop[:, :, :3]
    except Exception:
        return np.zeros((tile_size, tile_size, 3), dtype=np.uint8)

    h, w = map_crop.shape[:2]
    if h < patch_px_map or w < patch_px_map:
        padded = np.zeros((patch_px_map, patch_px_map, 3), dtype=np.uint8)
        padded[:h, :w] = map_crop
        map_crop = padded

    img = Image.fromarray(map_crop)
    img = img.resize((tile_size, tile_size), Image.BILINEAR)
    return np.array(img)


def map_px_to_gps(x_px, y_px):
    """Конвертация пикселей карты → GPS."""
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(MAP_ORIGIN_LAT))
    dx_m = x_px * RESOLUTION
    dy_m = y_px * RESOLUTION
    lon = MAP_ORIGIN_LON + dx_m / meters_per_deg_lon
    lat = MAP_ORIGIN_LAT + dy_m / meters_per_deg_lat
    return lat, lon


def gps_to_map_px(lat, lon):
    """Конвертация GPS → пиксели карты."""
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(MAP_ORIGIN_LAT))
    dx_m = (lon - MAP_ORIGIN_LON) * meters_per_deg_lon
    dy_m = (lat - MAP_ORIGIN_LAT) * meters_per_deg_lat
    return dx_m / RESOLUTION, dy_m / RESOLUTION


class FlightSimulator:
    """
    Симулятор полёта: генерирует кадры вдоль маршрута.

    Маршрут — последовательность тайлов (coords.npy).
    Скорость определяет, как быстро движемся по маршруту.
    """

    def __init__(self, coords, altitude, speed, reverse=False):
        self.coords = coords if not reverse else coords[::-1]
        self.altitude = altitude
        self.speed = speed  # м/с
        self.current_idx = 0
        self.current_progress = 0.0  # дробная часть между тайлами

        # Шаг между тайлами в метрах
        dx = (self.coords[1] - self.coords[0])
        self.step_m = math.sqrt(dx[0]**2 + dx[1]**2) * RESOLUTION
        self.dt_step = self.step_m / speed  # время между тайлами

        print(f"[Sim] Маршрут: {len(coords)} точек, шаг={self.step_m:.1f} м")
        print(f"[Sim] Высота: {altitude} м, скорость: {speed} м/с")
        print(f"[Sim] Время между точками: {self.dt_step:.1f} с")
        print(f"[Sim] Общее время: {len(coords) * self.dt_step / 60:.1f} мин")

    def step(self, dt):
        """Продвинуться по маршруту на dt секунд."""
        progress_inc = dt / self.dt_step
        self.current_progress += progress_inc

        while self.current_progress >= 1.0:
            self.current_progress -= 1.0
            self.current_idx += 1
            if self.current_idx >= len(self.coords) - 1:
                self.current_idx = len(self.coords) - 1
                self.current_progress = 0.0
                return False  # маршрут завершён

        return True  # продолжаем

    def get_center_px(self):
        """Текущая позиция в пикселях карты (центр кадра)."""
        c0 = self.coords[self.current_idx]
        c1 = self.coords[min(self.current_idx + 1, len(self.coords) - 1)]
        t = self.current_progress
        cx = c0[0] + (c1[0] - c0[0]) * t + TILE_SIZE / 2
        cy = c0[1] + (c1[1] - c0[1]) * t + TILE_SIZE / 2
        return cx, cy

    def get_heading(self):
        """Направление движения в радианах."""
        c0 = self.coords[self.current_idx]
        c1 = self.coords[min(self.current_idx + 1, len(self.coords) - 1)]
        dx = c1[0] - c0[0]
        dy = c1[1] - c0[1]
        return math.atan2(dy, dx)


class MAVLinkSender:
    """Отправка VISION_POSITION_ESTIMATE в FC."""

    def __init__(self, device='auto', baud=115200):
        self.device = device
        self.baud = baud
        self.connected = False
        self.vision_sent = 0
        self.telemetry = {}

    def _auto_detect_port(self):
        import glob
        candidates = sorted(glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*'))
        if not candidates:
            raise RuntimeError("Не найдено USB-портов")

        print(f"[MAVLink] Поиск FC: {candidates}")
        for port in candidates:
            try:
                conn = mavutil.mavlink_connection(port, baud=self.baud, dialect="ardupilotmega")
                msg = conn.wait_heartbeat(timeout=3)
                if msg:
                    print(f"[MAVLink] {port}: heartbeat OK (sys={msg.get_srcSystem()})")
                    conn.close()
                    return port
                conn.close()
            except Exception:
                pass
        raise RuntimeError("FC не найден")

    def connect(self):
        if self.device == 'auto':
            self.device = self._auto_detect_port()

        print(f"[MAVLink] Подключение к {self.device}...")
        self.conn = mavutil.mavlink_connection(
            self.device, baud=self.baud, dialect="ardupilotmega"
        )

        msg = self.conn.wait_heartbeat(timeout=10)
        if msg is None:
            raise RuntimeError("Нет heartbeat от FC")

        self.sys_id = msg.get_srcSystem()
        self.comp_id = msg.get_srcComponent()
        print(f"[MAVLink] FC: sys={self.sys_id}, comp={self.comp_id}, "
              f"type={msg.type}, autopilot={msg.autopilot}")
        self.connected = True

        # Запрашиваем телеметрию
        self._request_streams()

    def _request_streams(self):
        streams = [
            (mavutil.mavlink.MAV_DATA_STREAM_EXTRA1, 10),
            (mavutil.mavlink.MAV_DATA_STREAM_EXTRA2, 5),
            (mavutil.mavlink.MAV_DATA_STREAM_POSITION, 5),
        ]
        for stream_id, rate in streams:
            self.conn.mav.request_data_stream_send(
                self.sys_id, self.comp_id, stream_id, rate, 1
            )
        print(f"[MAVLink] Потоки запрошены")

    def start_telemetry_thread(self):
        self._running = True
        self._thread = threading.Thread(target=self._tel_loop, daemon=True)
        self._thread.start()

    def _tel_loop(self):
        while self._running:
            msg = self.conn.recv_match(
                type=['ATTITUDE', 'VFR_HUD', 'GLOBAL_POSITION_INT', 'HEARTBEAT'],
                blocking=True, timeout=0.1
            )
            if msg:
                t = msg.get_type()
                if t == 'ATTITUDE':
                    self.telemetry['yaw'] = msg.yaw
                    self.telemetry['roll'] = msg.roll
                    self.telemetry['pitch'] = msg.pitch
                elif t == 'VFR_HUD':
                    self.telemetry['alt'] = msg.alt
                elif t == 'GLOBAL_POSITION_INT':
                    if msg.lat != 0 and msg.lon != 0:
                        self.telemetry['gps_lat'] = msg.lat * 1e-7
                        self.telemetry['gps_lon'] = msg.lon * 1e-7

    def send_vision_position(self, lat, lon, alt_m, yaw_rad=0.0):
        """Отправка позиции в FC (lat/lon → NED)."""
        # Конвертация lat/lon → локальные NED координаты
        # Используем MAP_ORIGIN как точку отсчёта
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(math.radians(MAP_ORIGIN_LAT))

        x_north = (lat - MAP_ORIGIN_LAT) * meters_per_deg_lat
        y_east = (lon - MAP_ORIGIN_LON) * meters_per_deg_lon
        z_down = -alt_m

        # Ковариация: 1 м по позиции
        cov = [0.0] * 21
        cov[0] = 1.0   # var(x)
        cov[6] = 1.0   # var(y)
        cov[12] = 1.0  # var(z)
        cov[15] = 0.01  # var(roll)
        cov[18] = 0.01  # var(pitch)
        cov[20] = 0.01  # var(yaw)

        self.conn.mav.vision_position_estimate_send(
            int(time.time() * 1e6),
            x_north, y_east, z_down,
            0.0, 0.0, yaw_rad,
            cov
        )
        self.vision_sent += 1

    def send_heartbeat(self):
        self.conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            0,
            mavutil.mavlink.MAV_STATE_ACTIVE
        )

    def close(self):
        self._running = False
        self.connected = False


class FlightSimulationFC:
    """
    Полный пайплайн симуляции:
      маршрут → кадры → Siamese matching → координаты → FC → QGC
    """

    def __init__(self, model, device, src, coords, altitude, speed,
                 mav_sender, reverse=False, index_step=4):
        self.model = model
        self.device = device
        self.src = src
        self.simulator = FlightSimulator(coords, altitude, speed, reverse)
        self.mav = mav_sender

        # Индекс карты: эмбеддинги всех тайлов (прореженный для скорости)
        self.index_step = index_step
        self.index_embs = None
        self.index_locs = None
        self.index_coords = None

        # Статистика
        self.frames_processed = 0
        self.matches_correct = 0
        self.matches_total = 0
        self.errors_m = []

    def build_index(self):
        """Построение индекса эмбеддингов тайлов карты."""
        coords = np.load(COORDS_PATH)
        index_coords = coords[::self.index_step]

        print(f"[Sim] Индексация {len(index_coords)} тайлов (step={self.index_step})...")

        batch_size = 32
        all_embs = []
        all_locs = []

        batch_tiles = []
        batch_locs = []

        for i, (x, y) in enumerate(index_coords):
            tile = read_tile(self.src, x, y)
            if tile.max() == 0:
                continue
            batch_tiles.append(tile)
            batch_locs.append((x, y))

            if len(batch_tiles) >= batch_size:
                embs = self._embed_batch(batch_tiles)
                all_embs.append(embs)
                all_locs.extend(batch_locs)
                batch_tiles = []
                batch_locs = []

            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(index_coords)}]")

        if batch_tiles:
            embs = self._embed_batch(batch_tiles)
            all_embs.append(embs)
            all_locs.extend(batch_locs)

        self.index_embs = torch.cat(all_embs, dim=0)
        self.index_locs = np.array(all_locs)
        self.index_coords = index_coords

        print(f"[Sim] Индекс: {self.index_embs.shape[0]} тайлов, "
              f"{self.index_embs.shape[1]}D")

    @torch.no_grad()
    def _embed_batch(self, tiles):
        arr = normalize(np.array(tiles))
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device)
        return model(tensor)

    @torch.no_grad()
    def _embed_single(self, tile):
        arr = normalize(np.array([tile]))
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device)
        return model(tensor)[0]

    def find_position(self, frame):
        """
        Поиск позиции кадра в индексе.

        Returns:
            (best_x, best_y) — пиксели карты
            confidence — cosine similarity
            error_m — ошибка до истинной позиции (если известна)
        """
        query_emb = self._embed_single(frame)
        d = torch.cdist(query_emb.unsqueeze(0), self.index_embs).squeeze(0)
        nearest = d.argmin().item()
        best_x, best_y = self.index_locs[nearest]
        confidence = 1.0 - d[nearest].item()

        return (best_x, best_y), confidence, nearest

    def run(self):
        """Главный цикл симуляции."""
        dt = 1.0 / NAV_RATE_HZ
        running = True
        last_hb = 0

        print(f"\n[Sim] Запуск симуляции ({NAV_RATE_HZ} Гц)")
        print(f"[Sim] Ctrl+C для остановки")
        print(f"[Sim] Открой QGroundControl и подключись к FC для просмотра позиции\n")

        try:
            while running:
                loop_start = time.time()

                # 1. Шаг симулятора
                running = self.simulator.step(dt)
                cx, cy = self.simulator.get_center_px()
                heading = self.simulator.get_heading()

                # 2. Генерируем кадр камеры
                frame = camera_frame_from_map(self.src, (cx, cy),
                                              self.simulator.altitude, TILE_SIZE)
                if frame.max() == 0:
                    time.sleep(dt)
                    continue

                # 3. Siamese matching
                (pred_x, pred_y), conf, nearest_idx = self.find_position(frame)

                # Истинная позиция (для статистики)
                true_x = cx - TILE_SIZE / 2
                true_y = cy - TILE_SIZE / 2
                err_px = math.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
                err_m = err_px * RESOLUTION
                self.errors_m.append(err_m)
                self.matches_total += 1
                if err_m < 30:
                    self.matches_correct += 1

                # 4. Конвертация в GPS
                lat, lon = map_px_to_gps(pred_x + TILE_SIZE / 2,
                                         pred_y + TILE_SIZE / 2)

                # 5. Отправка в FC
                self.mav.send_vision_position(lat, lon, self.simulator.altitude, heading)

                self.frames_processed += 1

                # 6. Логирование (каждые 5 секунд)
                if self.frames_processed % 25 == 0:
                    med_err = float(np.median(self.errors_m[-25:]))
                    acc = self.matches_correct / self.matches_total * 100
                    print(f"[Sim] Точка {self.simulator.current_idx}/"
                          f"{len(self.simulator.coords)} | "
                          f"GPS: {lat:.5f}, {lon:.5f} | "
                          f"Err: {err_m:.0f} м (мед={med_err:.0f}) | "
                          f"Conf: {conf:.2f} | "
                          f"Acc: {acc:.0f}% | "
                          f"Vision: {self.mav.vision_sent}")

                # 7. Heartbeat компаньона
                now = time.time()
                if int(now) != int(last_hb):
                    self.mav.send_heartbeat()
                    last_hb = now

                # 8. Соблюдение частоты
                elapsed = time.time() - loop_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[Sim] Остановка по Ctrl+C")

        # Итоги
        print(f"\n{'='*60}")
        print("ИТОГИ СИМУЛЯЦИИ")
        print(f"{'='*60}")
        print(f"  Кадров: {self.frames_processed}")
        print(f"  Vision отправлено: {self.mav.vision_sent}")
        if self.matches_total > 0:
            acc = self.matches_correct / self.matches_total * 100
            med = float(np.median(self.errors_m)) if self.errors_m else 0
            print(f"  Точность (<30 м): {acc:.0f}%")
            print(f"  Медиана ошибки: {med:.0f} м")
            print(f"  Средняя ошибка: {float(np.mean(self.errors_m)):.0f} м")


def main():
    parser = argparse.ArgumentParser(
        description="Симуляция полёта на FC с передачей координат в QGC"
    )
    parser.add_argument('--altitude', type=float, default=1000.0,
                        help='Высота полёта (м), по умолчанию 1000')
    parser.add_argument('--speed', type=float, default=22.0,
                        help='Скорость (м/с), по умолчанию 22')
    parser.add_argument('--model', type=str, default='region_model.pth',
                        help='Путь к модели')
    parser.add_argument('--device', type=str, default='auto',
                        help='UART устройство (по умолчанию автодетект)')
    parser.add_argument('--route-reverse', action='store_true',
                        help='Обратный маршрут')
    parser.add_argument('--index-step', type=int, default=4,
                        help='Шаг прореживания индекса (4=1559 тайлов)')
    args = parser.parse_args()

    model_path = os.path.join(PROJECT_DIR, args.model)

    # 1. Загрузка модели
    model, device = load_model(model_path)

    # 2. Открытие карты
    print(f"[Sim] Открытие карты: {MAP_PATH}")
    src = rasterio.open(MAP_PATH)
    coords = np.load(COORDS_PATH)
    print(f"[Sim] Карта: {src.width}x{src.height} px, {len(coords)} тайлов маршрута")

    # 3. Подключение к FC
    mav = MAVLinkSender(device=args.device, baud=FC_BAUD)
    mav.connect()
    mav.start_telemetry_thread()

    # 4. Создание симуляции
    sim = FlightSimulationFC(
        model=model,
        device=device,
        src=src,
        coords=coords,
        altitude=args.altitude,
        speed=args.speed,
        mav_sender=mav,
        reverse=args.route_reverse,
        index_step=args.index_step,
    )

    # 5. Индексация тайлов
    sim.build_index()

    # 6. Запуск
    sim.run()

    # 7. Завершение
    mav.close()
    src.close()
    print("[Sim] Завершено")


if __name__ == '__main__':
    main()
