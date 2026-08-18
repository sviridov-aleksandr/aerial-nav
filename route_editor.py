#!/usr/bin/env python3
"""
route_editor.py — интерактивный редактор маршрутного задания.

Возможности:
  - Интерактивный ввод точек маршрута (широта, долгота)
  - Импорт маршрута из Яндекс.Карт (вставка ссылки или координат)
  - Импорт/экспорт в JSON
  - Расчёт километража между точками (хаверсинус)
  - Удаление, вставка, изменение точек
  - Сохранение в единый файл: flight_mission.json

Использование:
  python3 route_editor.py              # новая сессия
  python3 route_editor.py --file f.json # продолжить с файлом

Формат JSON:
  {
    "name": "Описание маршрута",
    "source": "yandex|manual|file",
    "points": [
      [lat, lon],
      ...
    ]
  }
"""

import os
import sys
import re
import json
import math
import argparse


# ------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------

def haversine(lat1, lon1, lat2, lon2):
    """Расстояние в метрах между двумя точками (GPS)."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def parse_coord(s):
    """Парсит координату. Принимает '46.123' или '46,123'."""
    s = s.replace(',', '.').strip()
    try:
        v = float(s)
    except ValueError:
        return None
    if not -90.0 <= v <= 90.0:
        return None
    return v


def parse_lon(s):
    """Парсит долготу (диапазон -180..180)."""
    s = s.replace(',', '.').strip()
    try:
        v = float(s)
    except ValueError:
        return None
    if not -180.0 <= v <= 180.0:
        return None
    return v


def parse_yandex_route(text):
    """
    Извлекает координаты из текста/ссылки Яндекс.Карт.

    Поддерживает:
      - Ссылка: https://yandex.ru/maps/?ll=32.353548%2C46.344208&...
      - Ссылка с маршрутом: https://yandex.ru/maps/.../?rtext=46.344208,32.353548~46.358646,32.332859
      - Просто координаты: "46.344208 32.353548" или "46.344208,32.353548"
      - Несколько точек, разделённых пробелами/~
    """
    points = []

    # 1. rtext (пешеходный/авто маршрут): rtext=lat,lon~lat,lon~...
    m = re.search(r'rtext=([0-9.,_~\-\s]+)', text)
    if m:
        raw = m.group(1)
        for chunk in raw.replace('~', ' ').split():
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = chunk.replace(',', ' ').split()
            if len(parts) == 2:
                lat = parse_coord(parts[0])
                lon = parse_lon(parts[1])
                if lat is not None and lon is not None:
                    points.append((lat, lon))
        if points:
            return points

    # 2. Координаты в URL: ll=lon%2Clat  (или &ll=lon,lat)
    #    Для маршрута может быть несколько параметров ll / pt
    for param in ['ll', 'pt']:
        for m in re.finditer(param + r'=([-0-9.]+)(?:%2C|,)([-0-9.]+)', text):
            lat = parse_coord(m.group(2))
            lon = parse_lon(m.group(1))
            if lat is not None and lon is not None:
                # избегаем дубликатов
                if (round(lat, 6), round(lon, 6)) not in [(round(p[0], 6), round(p[1], 6)) for p in points]:
                    points.append((lat, lon))
    if points:
        return points

    # 3. Просто координаты в тексте
    pairs = list(re.finditer(r'([-]?\d+[.,]\d+)\s*[,; ]\s*([-]?\d+[.,]\d+)', text))
    for m in pairs:
        lat = parse_coord(m.group(1))
        lon = parse_lon(m.group(2))
        if lat is not None and lon is not None:
            # избегаем дубликатов
            if (round(lat, 6), round(lon, 6)) not in [(round(p[0], 6), round(p[1], 6)) for p in points]:
                points.append((lat, lon))

    return points


def format_point(p):
    return f"({p[0]:.6f}, {p[1]:.6f})"


# ------------------------------------------------------------
# Основной редактор
# ------------------------------------------------------------

class RouteEditor:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.name = "Маршрут без названия"
        self.source = "manual"
        self.points = []

        if filepath and os.path.exists(filepath):
            self._load(filepath)

    def _load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.name = data.get('name', self.name)
        self.source = data.get('source', self.source)
        self.points = [(float(p[0]), float(p[1])) for p in data.get('points', [])]
        print(f"Загружен маршрут: '{self.name}' ({len(self.points)} точек) из {path}")

    def save(self, path=None):
        path = path or self.filepath
        if not path:
            path = 'flight_mission.json'
            self.filepath = path
        data = {
            'name': self.name,
            'source': self.source,
            'points': [[round(lat, 6), round(lon, 6)] for lat, lon in self.points]
        }
        with open(path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✓ Сохранено: {path}")

    def _print_menu(self):
        total_dist = self._total_distance()
        print()
        print(f"Маршрут: '{self.name}'  |  Источник: {self.source}  |  Точек: {len(self.points)}  |  Длина: {total_dist/1000:.1f} км")
        print("-" * 60)
        for i, (lat, lon) in enumerate(self.points):
            marker = "  " if i < len(self.points) - 1 else "«"
            print(f"  [{i}] {format_point((lat, lon))}{marker}")
            if i < len(self.points) - 1:
                d = haversine(lat, lon, self.points[i+1][0], self.points[i+1][1])
                print(f"      └─→ {d/1000:.2f} км")
        print("-" * 60)
        print("Команды:")
        print("  add <lat> <lon>     — добавить точку в конец")
        print("  ins <idx> <lat> <lon> — вставить точку на позицию")
        print("  set <idx> <lat> <lon> — изменить точку")
        print("  del <idx>           — удалить точку")
        print("  yandex <Url/текст>  — импорт маршрута из ссылки Яндекс.Карт")
        print("  name <название>     — задать название маршрута")
        print("  src <источник>      — задать источник (yandex/manual/file/...):")
        print("  list                — показать все точки")
        print("  dist                — общая длина маршрута")
        print("  load <файл>         — загрузить JSON")
        print("  save [<файл>]       — сохранить")
        print("  quit/exit/q         — выход")
        print()

    def _total_distance(self):
        total = 0
        for i in range(len(self.points) - 1):
            total += haversine(
                self.points[i][0], self.points[i][1],
                self.points[i+1][0], self.points[i+1][1]
            )
        return total

    def run(self):
        print("Интерактивный редактор маршрутного задания")
        print("Введите 'help' для подсказки, 'quit' для выхода")
        self._print_menu()

        while True:
            try:
                cmd = input("→ ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                print("Выход.")
                break

            if not cmd:
                continue

            parts = cmd.split()
            action = parts[0].lower()

            if action in ('quit', 'exit', 'q'):
                if self.points and input("Сохранить перед выходом? (y/n): ").lower().startswith('y'):
                    self.save()
                break

            elif action in ('help', '?'):
                self._print_menu()

            elif action == 'add':
                if len(parts) < 3:
                    print("Формат: add <lat> <lon>")
                    continue
                lat, lon = parse_coord(parts[1]), parse_lon(parts[2])
                if lat is None or lon is None:
                    print("✗ Некорректные координаты")
                    continue
                self.points.append((lat, lon))
                print(f"✓ Добавлена точка {format_point((lat, lon))}")

            elif action == 'ins':
                if len(parts) < 4:
                    print("Формат: ins <idx> <lat> <lon>")
                    continue
                idx = int(parts[1])
                lat, lon = parse_coord(parts[2]), parse_lon(parts[3])
                if lat is None or lon is None or not (0 <= idx <= len(self.points)):
                    print("✗ Некорректные данные")
                    continue
                self.points.insert(idx, (lat, lon))
                print(f"✓ Вставлена точка на позицию {idx}")

            elif action == 'set':
                if len(parts) < 4:
                    print("Формат: set <idx> <lat> <lon>")
                    continue
                idx = int(parts[1])
                lat, lon = parse_coord(parts[2]), parse_lon(parts[3])
                if lat is None or lon is None or not (0 <= idx < len(self.points)):
                    print("✗ Некорректные данные")
                    continue
                self.points[idx] = (lat, lon)
                print(f"✓ Точка {idx} изменена на {format_point((lat, lon))}")

            elif action == 'del':
                if len(parts) < 2:
                    print("Формат: del <idx>")
                    continue
                idx = int(parts[1])
                if 0 <= idx < len(self.points):
                    removed = self.points.pop(idx)
                    print(f"✓ Удалена точка {format_point(removed)}")
                else:
                    print("✗ Индекс вне диапазона")

            elif action == 'yandex':
                rest = cmd[len('yandex'):].strip()
                if not rest:
                    print("Вставьте ссылку или координаты из Яндекс.Карт:")
                    rest = input("  → ").strip()
                pts = parse_yandex_route(rest)
                if pts:
                    self.points = pts
                    self.source = 'yandex'
                    print(f"✓ Импортировано {len(pts)} точек из Яндекс.Карт")
                else:
                    print("✗ Не удалось извлечь координаты. Скопируйте ссылку с параметром rtext или ll.")

            elif action == 'name':
                self.name = cmd[len('name'):].strip() or self.name
                print(f"✓ Название: '{self.name}'")

            elif action == 'src':
                self.source = cmd[len('src'):].strip() or self.source
                print(f"✓ Источник: '{self.source}'")

            elif action == 'list':
                for i, p in enumerate(self.points):
                    print(f"  [{i}] {format_point(p)}")

            elif action == 'dist':
                print(f"Длина маршрута: {self._total_distance()/1000:.1f} км ({self._total_distance():.0f} м)")

            elif action == 'load':
                path = ' '.join(parts[1:]).strip() or self.filepath
                if path and os.path.exists(path):
                    self._load(path)
                else:
                    print(f"✗ Файл не найден: {path}")

            elif action == 'save':
                path = ' '.join(parts[1:]).strip() or self.filepath
                self.save(path)

            else:
                print(f"Неизвестная команда: {action}. Введите 'help' для списка команд.")

            self._print_menu()


def main():
    parser = argparse.ArgumentParser(description='Интерактивный редактор маршрута')
    parser.add_argument('--file', '-f', help='Файл маршрута (JSON) для загрузки/сохранения')
    parser.add_argument('--from-yandex', '-y', help='Вставить URL Яндекс.Карт для импорта сразу')
    parser.add_argument('--points', '-p', nargs='+',
                        help='Точки на старте: lat lon lat lon ... или lat,lon lat,lon ...')
    parser.add_argument('--save-to', '-o', help='Сохранить сразу и выйти')
    args = parser.parse_args()

    editor = RouteEditor(args.file)

    if args.from_yandex:
        pts = parse_yandex_route(args.from_yandex)
        if pts:
            editor.points = pts
            editor.source = 'yandex'
            print(f"Импортировано {len(pts)} точек из Яндекс.Карт")

    if args.points:
        flat = []
        for p in args.points:
            if ',' in p:
                a, b = p.split(',')
                flat += [a, b]
            else:
                flat.append(p)
        pts = []
        for i in range(0, len(flat) - 1, 2):
            lat = parse_coord(flat[i])
            lon = parse_lon(flat[i+1])
            if lat is not None and lon is not None:
                pts.append((lat, lon))
        if pts:
            editor.points = pts
            print(f"Импортировано {len(pts)} точек из аргументов")

    if args.save_to:
        editor.save(args.save_to)
        return

    editor.run()


if __name__ == '__main__':
    main()