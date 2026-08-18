"""
Планировщик траекторий для дрона с обходом препятствий.
Использует A* алгоритм на сетке карты.
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional, Dict
from enum import Enum
import math


class ObstacleType(Enum):
    """Типы препятствий."""
    CLEAR = 0
    BUILDING = 1
    TREE = 2
    WATER = 3
    UNKNOWN = 4


class Waypoint:
    """Вспомогательная точка траектории."""

    def __init__(self, x: float, y: float, z: float = 50.0,
                 heading: float = 0.0, speed: float = 5.0):
        self.x = x
        self.y = y
        self.z = z
        self.heading = heading
        self.speed = speed

    def distance_to(self, other: 'Waypoint') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def __repr__(self):
        return f"Waypoint({self.x:.1f}, {self.y:.1f}, z={self.z:.1f})"


class TrajectoryPlanner:
    """
    Планировщик траекторий с обходом препятствий.
    
    Алгоритм:
    1. A* для поиска пути на сетке
    2. Сглаживание траектории (B-сплайны)
    3. Расчёт скоростей и ускорений
    """

    def __init__(self, resolution: float = 2.35,
                 min_altitude: float = 30.0,
                 max_altitude: float = 100.0,
                 max_speed: float = 15.0,
                 obstacle_buffer: float = 10.0):
        """
        Args:
            resolution: разрешение карты (м/пиксель)
            min_altitude: минимальная высота (м)
            max_altitude: максимальная высота (м)
            max_speed: максимальная скорость (м/с)
            obstacle_buffer: буфер вокруг препятствий (м)
        """
        self.resolution = resolution
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        self.max_speed = max_speed
        self.obstacle_buffer = obstacle_buffer

        # Карта препятствий
        self.obstacle_map = None
        self.map_shape = None
        self.map_origin = (0.0, 0.0)

    def set_obstacle_map(self, obstacle_map: np.ndarray,
                         origin: Tuple[float, float] = (0.0, 0.0)):
        """
        Устанавливает карту препятствий.
        
        Args:
            obstacle_map: 2D массив (H, W), значения 0-4 (ObstacleType)
            origin: координаты левого верхнего угла (x, y) в метрах
        """
        self.obstacle_map = obstacle_map.astype(np.uint8)
        self.map_shape = obstacle_map.shape
        self.map_origin = origin
        print(f"[Planner] Карта препятствий: {obstacle_map.shape}, "
              f"origin={origin}")

    def plan(self, start: Tuple[float, float, float],
             goal: Tuple[float, float, float],
             current_altitude: float = 50.0) -> List[Waypoint]:
        """
        Планирует траекторию от start до goal.
        
        Args:
            start: (x, y, z) начальная позиция
            goal: (x, y, z) целевая позиция
            current_altitude: текущая высота
        
        Returns:
            Список Waypoint
        """
        print(f"\n[Planner] Планирование: {start} → {goal}")

        # 1. Проверка границ
        if not self._is_in_bounds(start) or not self._is_in_bounds(goal):
            print("[Planner] ОШИБКА: точка вне границ карты!")
            return []

        # 2. A* поиск пути
        path = self._astar(start[:2], goal[:2])
        if not path:
            print("[Planner] ОШИБКА: путь не найден!")
            return []

        print(f"[Planner] A* нашёл путь: {len(path)} точек")

        # 3. Сглаживание
        smoothed = self._smooth_path(path)

        # 4. Генерация waypoint
        waypoints = self._generate_waypoints(smoothed, start[2], goal[2],
                                             current_altitude)

        print(f"[Planner] Траектория: {len(waypoints)} waypoint")
        return waypoints

    def _astar(self, start: Tuple[float, float],
               goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """A* алгоритм поиска пути."""
        if self.obstacle_map is None:
            # Если нет карты препятствий — прямая линия
            return self._direct_line(start, goal)

        # Конвертация в пиксели
        start_px = self._meters_to_pixels(start)
        goal_px = self._meters_to_pixels(goal)

        if not self._is_valid_pixel(start_px) or not self._is_valid_pixel(goal_px):
            return []

        start_h = self._heuristic(start_px, goal_px)
        open_set = [(start_h, 0, start_px)]
        came_from = {}
        g_score = {start_px: 0}
        closed_set = set()

        while open_set:
            f, g, current = heapq.heappop(open_set)

            if current == goal_px:
                # Восстановление пути
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return [self._pixels_to_meters(p) for p in path]

            if current in closed_set:
                continue
            closed_set.add(current)

            # Соседи (8 направлений)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)

                if not self._is_valid_pixel(neighbor):
                    continue
                if neighbor in closed_set:
                    continue

                # Проверка препятствий
                if self._is_obstacle(neighbor):
                    continue

                tentative_g = g + (1.0 if dx == 0 or dy == 0 else 1.414)

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = self._heuristic(neighbor, goal_px)
                    heapq.heappush(open_set, (tentative_g + h, tentative_g, neighbor))

        return []

    def _heuristic(self, a: Tuple[int, int],
                   b: Tuple[int, int]) -> float:
        """Эвристика (евклидово расстояние)."""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def _is_obstacle(self, pixel: Tuple[int, int]) -> bool:
        """Проверка, является ли пиксель препятствием."""
        if (0 <= pixel[1] < self.map_shape[0] and
                0 <= pixel[0] < self.map_shape[1]):
            obstacle = self.obstacle_map[pixel[1], pixel[0]]
            return obstacle != ObstacleType.CLEAR.value
        return True

    def _direct_line(self, start: Tuple[float, float],
                     goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Прямая линия между двумя точками."""
        steps = max(10, int(math.sqrt((goal[0] - start[0])**2 +
                                       (goal[1] - start[1])**2) / 5))
        path = []
        for i in range(steps + 1):
            t = i / steps
            x = start[0] + t * (goal[0] - start[0])
            y = start[1] + t * (goal[1] - start[1])
            path.append((x, y))
        return path

    def _smooth_path(self, path: List[Tuple[float, float]],
                     smoothing_factor: float = 0.3) -> List[Tuple[float, float]]:
        """Сглаживание пути (простой метод)."""
        if len(path) < 3:
            return path

        smoothed = [path[0]]

        for i in range(1, len(path) - 1):
            prev = np.array(path[i - 1])
            curr = np.array(path[i])
            next_p = np.array(path[i + 1])

            # Среднее между текущей и следующей точкой
            new_point = curr * (1 - smoothing_factor) + next_p * smoothing_factor
            smoothed.append(tuple(new_point))

        smoothed.append(path[-1])
        return smoothed

    def _generate_waypoints(self, path: List[Tuple[float, float]],
                            start_z: float, goal_z: float,
                            current_z: float) -> List[Waypoint]:
        """Генерация waypoint из пути."""
        waypoints = []
        total_dist = 0

        for i, (x, y) in enumerate(path):
            # Высота: интерполяция
            if i == 0:
                z = start_z
            elif i == len(path) - 1:
                z = goal_z
            else:
                # Держим постоянную высоту
                z = max(current_z, self.min_altitude)

            # Heading
            if i > 0:
                dx = x - path[i - 1][0]
                dy = y - path[i - 1][1]
                heading = math.degrees(math.atan2(dy, dx))
            else:
                heading = 0.0

            # Скорость
            speed = self.max_speed

            wp = Waypoint(x, y, z, heading, speed)
            waypoints.append(wp)

            if i > 0:
                total_dist += waypoints[-1].distance_to(waypoints[-2])

        print(f"[Planner] Общая дистанция: {total_dist:.1f} м")
        return waypoints

    def _is_in_bounds(self, point: Tuple[float, float, float]) -> bool:
        """Проверка, находится ли точка в пределах карты."""
        x, y = point[0], point[1]
        if self.obstacle_map is None:
            return True

        px, py = self._meters_to_pixels((x, y))
        return (0 <= py < self.map_shape[0] and
                0 <= px < self.map_shape[1])

    def _is_valid_pixel(self, pixel: Tuple[int, int]) -> bool:
        """Проверка валидности пикселя."""
        return (0 <= pixel[1] < self.map_shape[0] and
                0 <= pixel[0] < self.map_shape[1])

    def _meters_to_pixels(self, meters: Tuple[float, float]) -> Tuple[int, int]:
        """Конвертация метров в пиксели."""
        px = int((meters[0] - self.map_origin[0]) / self.resolution)
        py = int((meters[1] - self.map_origin[1]) / self.resolution)
        return (px, py)

    def _pixels_to_meters(self, pixels: Tuple[int, int]) -> Tuple[float, float]:
        """Конвертация пикселей в метры."""
        x = pixels[0] * self.resolution + self.map_origin[0]
        y = pixels[1] * self.resolution + self.map_origin[1]
        return (x, y)

    def visualize(self, waypoints: List[Waypoint],
                  output_path: str = 'trajectory.png'):
        """Визуализация траектории."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Карта препятствий
        if self.obstacle_map is not None:
            ax.imshow(self.obstacle_map, cmap='viridis', origin='lower',
                      extent=[self.map_origin[0],
                              self.map_origin[0] + self.map_shape[1] * self.resolution,
                              self.map_origin[1],
                              self.map_origin[1] + self.map_shape[0] * self.resolution])

        # Waypoints
        if waypoints:
            xs = [wp.x for wp in waypoints]
            ys = [wp.y for wp in waypoints]
            ax.plot(xs, ys, 'b-', linewidth=2, label='Trajectory')
            ax.plot(xs[0], ys[0], 'go', markersize=15, label='Start')
            ax.plot(xs[-1], ys[-1], 'rx', markersize=15, label='Goal')

            # Подписи waypoint
            for i, wp in enumerate(waypoints[::max(1, len(waypoints) // 10)]):
                ax.annotate(f'#{i}', (wp.x, wp.y), fontsize=8)

        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('Trajectory Planning')
        ax.legend()
        ax.grid(True)
        ax.axis('equal')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"[Planner] График сохранён: {output_path}")
        plt.show()


def test_trajectory_planning():
    """Тест планировщика траекторий."""
    from map_loader import MapLoader

    print("=" * 60)
    print("ТЕСТ ПЛАНИРОВЩИКА ТРАЕКТОРИЙ")
    print("=" * 60)

    # Загрузка карты
    loader = MapLoader()
    loader.generate_synthetic_map(size=1024, resolution=0.1)
    region = loader.get_region()
    tile = list(region.tiles.values())[0]

    # Создание карты препятствий (синтетическая)
    obstacle_map = np.zeros_like(tile.image[:, :, 0], dtype=np.uint8)

    # Добавляем "здания" (прямоугольники)
    for i in range(5):
        x = np.random.randint(100, 900)
        y = np.random.randint(100, 900)
        w = np.random.randint(30, 80)
        h = np.random.randint(30, 80)
        obstacle_map[y:y+h, x:x+w] = ObstacleType.BUILDING.value

    # Инициализация планировщика
    planner = TrajectoryPlanner(
        resolution=region.resolution,
        min_altitude=30.0,
        max_altitude=100.0,
        max_speed=10.0,
        obstacle_buffer=10.0
    )
    planner.set_obstacle_map(obstacle_map, origin=(0.0, 0.0))

    # Планирование
    start = (10.0, 10.0, 50.0)
    goal = (90.0, 90.0, 50.0)

    waypoints = planner.plan(start, goal, current_altitude=50.0)

    if waypoints:
        print(f"\nТраектория ({len(waypoints)} waypoint):")
        for i, wp in enumerate(waypoints[::max(1, len(waypoints) // 10)]):
            print(f"  #{i}: ({wp.x:.1f}, {wp.y:.1f}, z={wp.z:.1f}, "
                  f"heading={wp.heading:.1f}°, speed={wp.speed:.1f} м/с)")

        # Визуализация
        planner.visualize(waypoints, output_path='/home/alex/aerial-nav/trajectory.png')
    else:
        print("[Planner] Путь не найден!")


if __name__ == '__main__':
    test_trajectory_planning()  