#!/usr/bin/env python3
"""
web_route_planner — веб-интерфейс планирования маршрута и загрузки карт.

Возможности:
  - Интерактивная карта (Leaflet) с выбором точек маршрута кликом
  - Импорт маршрута из ссылки Яндекс.Карт
  - Выбор поставщика тайлов (ESRI, OSM, Google, Yandex)
  - Превью области загрузки (подсветка тайлов)
  - Сохранение маршрута в JSON
  - Запуск скачивания карты (фоновый процесс)

Запуск:
  /home/alex/my_project_env/bin/python app.py
  Открыть http://localhost:8080
"""

import os
import sys
import json
import math
import signal
import threading
import subprocess
import re
import urllib.request
import urllib.parse

from flask import Flask, request, jsonify, send_from_directory

# Путь к проекту
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from route_editor import parse_yandex_route, haversine
from flight_sim_backend import sim

app = Flask(__name__)

# Состояние
ROUTE_FILE = os.path.join(PROJECT_DIR, 'flight_mission.json')
DOWNLOAD_LOG = os.path.join(PROJECT_DIR, 'download.log')
DOWNLOAD_PID_FILE = os.path.join(PROJECT_DIR, 'download.pid')

# Поставщики тайлов
TILE_PROVIDERS = {
    'esri': {
        'name': 'ESRI World Imagery',
        'url': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        'attribution': '© ESRI',
        'max_zoom': 19,
    },
    'osm': {
        'name': 'OpenStreetMap',
        'url': 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
        'attribution': '© OpenStreetMap',
        'max_zoom': 19,
    },
    'google': {
        'name': 'Google Satellite',
        'url': 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        'attribution': '© Google',
        'max_zoom': 20,
    },
    'yandex': {
        'name': 'Yandex Satellite',
        'url': 'https://core-renderer-tiles.maps.yandex.net/tiles?l=sat&x={x}&y={y}&z={z}',
        'attribution': '© Yandex',
        'max_zoom': 19,
    },
}


def load_route():
    """Загружает маршрут из JSON."""
    if os.path.exists(ROUTE_FILE):
        with open(ROUTE_FILE) as f:
            data = json.load(f)
        return data
    return {'name': 'Маршрут без названия', 'source': 'manual', 'points': []}


def save_route(data):
    """Сохраняет маршрут в JSON."""
    with open(ROUTE_FILE, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def route_length(points):
    """Общая длина маршрута в метрах."""
    total = 0
    for i in range(len(points) - 1):
        total += haversine(points[i][0], points[i][1],
                           points[i+1][0], points[i+1][1])
    return total


def estimate_tiles(points, zoom=19, width_m=500, patch_km=1.0):
    """Оценка количества тайлов для загрузки."""
    if not points:
        return 0
    # Площадь: квадраты у точек + полосы между ними
    total_area = 0
    for i in range(len(points)):
        total_area += (patch_km * 1000) ** 2
    for i in range(len(points) - 1):
        d = haversine(points[i][0], points[i][1],
                      points[i+1][0], points[i+1][1])
        total_area += d * width_m
    # Площадь тайла на zoom 19 ≈ 0.5 м/px → 256*0.5 = 128 м
    tile_m = 256 * (156543.03 * math.cos(math.radians(46)) / (2 ** zoom))
    return int(total_area / (tile_m ** 2)) + 100


def kill_download():
    """Убивает процесс скачивания карты, если он запущен."""
    if not os.path.exists(DOWNLOAD_PID_FILE):
        return
    try:
        with open(DOWNLOAD_PID_FILE) as f:
            pid = int(f.read().strip())
        # Проверяем, жив ли процесс
        os.kill(pid, 0)
        # Убиваем всю группу процессов (shell + python + воркеры)
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)
        print(f"[WebPlanner] Процесс скачивания {pid} остановлен")
    except (ProcessLookupError, ValueError, FileNotFoundError):
        pass
    finally:
        try:
            os.remove(DOWNLOAD_PID_FILE)
        except FileNotFoundError:
            pass


def cleanup_on_exit():
    """Останавливает скачивание при выходе из сервера."""
    kill_download()
    print("[WebPlanner] Сервер остановлен, хвостов не осталось")


# Регистрируем очистку при выходе
import atexit
atexit.register(cleanup_on_exit)


# ------------------------------------------------------------
# API
# ------------------------------------------------------------

@app.route('/')
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')


@app.route('/api/route', methods=['GET'])
def api_get_route():
    data = load_route()
    points = data.get('points', [])
    return jsonify({
        'name': data.get('name', ''),
        'source': data.get('source', ''),
        'points': points,
        'length_km': round(route_length(points) / 1000, 2),
    })


@app.route('/api/route', methods=['POST'])
def api_save_route():
    data = request.get_json()
    points = data.get('points', [])
    # Валидация
    clean = []
    for p in points:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            lat, lon = float(p[0]), float(p[1])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                clean.append([round(lat, 6), round(lon, 6)])
    route_data = {
        'name': data.get('name', 'Маршрут без названия'),
        'source': data.get('source', 'manual'),
        'points': clean,
    }
    save_route(route_data)
    return jsonify({'ok': True, 'points': len(clean),
                    'length_km': round(route_length(clean) / 1000, 2)})


@app.route('/api/import/yandex', methods=['POST'])
def api_import_yandex():
    data = request.get_json()
    text = data.get('text', '')
    points = parse_yandex_route(text)
    if points:
        return jsonify({'ok': True, 'points': points})
    return jsonify({'ok': False, 'error': 'Не удалось извлечь координаты'})


@app.route('/api/route_by_road', methods=['POST'])
def api_route_by_road():
    """Маршрутизация по дорогам через OSRM (public demo server)."""
    data = request.get_json()
    waypoints = data.get('waypoints', [])
    profile = data.get('profile', 'driving')  # driving, foot, cycling

    if len(waypoints) < 2:
        return jsonify({'ok': False, 'error': 'Нужно минимум 2 точки'})

    # Формируем URL для OSRM API
    coords_str = ';'.join(f'{lon},{lat}' for lat, lon in waypoints)
    url = (f'https://router.project-osrm.org/route/v1/{profile}/{coords_str}'
           f'?overview=full&geometries=geojson')

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'AerialNav/1.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())

        if result.get('code') != 'Ok':
            return jsonify({'ok': False, 'error': 'OSRM: ' + result.get('message', 'ошибка')})

        route = result['routes'][0]
        geometry = route['geometry']['coordinates']  # [[lon, lat], ...]
        distance_km = route['distance'] / 1000

        # Конвертируем в [lat, lon]
        points = [[coord[1], coord[0]] for coord in geometry]

        return jsonify({
            'ok': True,
            'points': points,
            'distance_km': round(distance_km, 2),
            'waypoints_count': len(waypoints),
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/providers', methods=['GET'])
def api_providers():
    return jsonify(TILE_PROVIDERS)


@app.route('/api/estimate', methods=['POST'])
def api_estimate():
    data = request.get_json()
    points = data.get('points', [])
    zoom = int(data.get('zoom', 19))
    width_m = float(data.get('width_m', 500))
    patch_km = float(data.get('patch_km', 1.0))
    tiles = estimate_tiles(points, zoom, width_m, patch_km)
    length = route_length(points)
    return jsonify({
        'tiles': tiles,
        'length_km': round(length / 1000, 2),
        'zoom': zoom,
    })


@app.route('/api/download', methods=['POST'])
def api_download():
    """Запускает скачивание карты в фоне."""
    data = request.get_json()
    points = data.get('points', [])
    source = data.get('source', 'esri')
    resolution = float(data.get('resolution', 0.5))
    width_m = float(data.get('width_m', 500))
    patch_km = float(data.get('patch_km', 1.0))

    if len(points) < 2:
        return jsonify({'ok': False, 'error': 'Нужно минимум 2 точки'})

    # Проверяем, не идёт ли уже загрузка
    if os.path.exists(DOWNLOAD_PID_FILE):
        try:
            with open(DOWNLOAD_PID_FILE) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            return jsonify({'ok': False, 'error': f'Загрузка уже идёт (PID {pid})'})
        except (ProcessLookupError, ValueError):
            pass

    # Формируем команду
    route_args = ' '.join(f'{lat},{lon}' for lat, lon in points)
    num_points = len(points)
    num_segments = len(points) - 1
    total_parts = num_points + num_segments
    cmd = (
        f'cd {PROJECT_DIR} && '
        f'PYTHONUNBUFFERED=1 '
        f'/home/alex/my_project_env/bin/python route_strip_map.py '
        f'--route {route_args} '
        f'--source {source} '
        f'--resolution {resolution} '
        f'--seg-width {width_m} '
        f'--point-size {patch_km} '
        f'--output map_cache/route_map_{source}.png '
        f'> {DOWNLOAD_LOG} 2>&1 & echo $! > {DOWNLOAD_PID_FILE}'
    )
    subprocess.Popen(cmd, shell=True)
    return jsonify({'ok': True, 'message': 'Загрузка запущена'})


@app.route('/api/download/stop', methods=['POST'])
def api_download_stop():
    """Принудительная остановка скачивания."""
    kill_download()
    return jsonify({'ok': True, 'message': 'Скачивание остановлено'})


@app.route('/api/download/status', methods=['GET'])
def api_download_status():
    """Статус фоновой загрузки с парсингом прогресса по точкам."""
    running = False
    if os.path.exists(DOWNLOAD_PID_FILE):
        try:
            with open(DOWNLOAD_PID_FILE) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            running = True
        except (ProcessLookupError, ValueError):
            running = False

    log_tail = ''
    progress = None
    stages = []
    if os.path.exists(DOWNLOAD_LOG):
        with open(DOWNLOAD_LOG) as f:
            lines = f.readlines()
        log_tail = ''.join(lines[-30:])

        # Парсим прогресс по точкам/сегментам
        # Формат: [Точка 1] [████████░░░░] 50.0% (200/400) 5.0/s  ETA: 40s
        #         [Сегмент 2] [DONE] 15/30 (120.5s)
        current_stage = None
        for line in lines:
            m = re.search(r'\[([^\]]+)\].*?(\d+(?:\.\d+)?)%\s*\((\d+)/(\d+)\)', line)
            if m:
                stage_name = m.group(1)
                current_stage = {
                    'name': stage_name,
                    'current': int(m.group(3)),
                    'total': int(m.group(4)),
                    'percent': int(float(m.group(2))),
                    'done': False,
                }
                # Обновляем или добавляем
                found = False
                for s in stages:
                    if s['name'] == stage_name:
                        s.update(current_stage)
                        found = True
                        break
                if not found:
                    stages.append(current_stage)

            # DONE = этап завершён
            m_done = re.search(r'\[([^\]]+)\].*\[DONE\]', line)
            if m_done:
                stage_name = m_done.group(1)
                for s in stages:
                    if s['name'] == stage_name:
                        s['done'] = True
                        s['percent'] = 100

        # Текущий прогресс (последний незавершённый)
        for s in reversed(stages):
            if not s['done']:
                progress = s
                break

    return jsonify({
        'running': running,
        'log': log_tail,
        'progress': progress,
        'stages': stages,
    })


# ------------------------------------------------------------
# API симуляции полёта
# ------------------------------------------------------------

@app.route('/simulation')
def simulation_page():
    """Страница визуализации полёта."""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'simulation.html')


@app.route('/api/sim/start', methods=['POST'])
def api_sim_start():
    sim.start()
    return jsonify({'ok': True, 'state': sim.get_state()})


@app.route('/api/sim/pause', methods=['POST'])
def api_sim_pause():
    sim.pause()
    return jsonify({'ok': True, 'state': sim.get_state()})


@app.route('/api/sim/reset', methods=['POST'])
def api_sim_reset():
    sim.stop()
    sim.reset()
    return jsonify({'ok': True, 'state': sim.get_state()})


@app.route('/api/sim/speed', methods=['POST'])
def api_sim_speed():
    data = request.get_json() or {}
    mult = float(data.get('mult', 1.0))
    sim.set_speed_mult(mult)
    return jsonify({'ok': True, 'speed_mult': sim.speed_mult})


@app.route('/api/sim/state', methods=['GET'])
def api_sim_state():
    return jsonify(sim.get_state())


if __name__ == '__main__':
    # Обработка Ctrl+C / kill — корректная остановка
    signal.signal(signal.SIGINT, lambda s, f: (cleanup_on_exit(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda s, f: (cleanup_on_exit(), sys.exit(0)))

    print("Web Route Planner запущен: http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)