#!/usr/bin/env python3
"""
Главная точка входа — симуляция автономного полёта дрона
с навигацией по подстилающей поверхности.
"""

import sys
import os
import time
import numpy as np
from map_loader import MapLoader
from real_map_loader import RealMapLoader
from navigator import AerialNavigator
from simulator import FlightSimulator, CameraConfig
from gui import NavigationGUI


def run_simulation(num_frames: int = 300, show_gui: bool = True,
                   map_path: str = None, synthetic: bool = True):
    """
    Запустить симуляцию полёта дрона.

    Args:
        num_frames: количество кадров для симуляции
        show_gui: показывать GUI визуализацию
        map_path: путь к карте (если None, синтетическая)
        synthetic: использовать синтетическую карту
    """
    print("=" * 60)
    print("АВТОНОМНАЯ НАВИГАЦИЯ ДРОНА ПО ПОДСТИЛАЮЩЕЙ ПОВЕРХНОСТИ")
    print("=" * 60)

    # 1. Создаём навигатор
    print("\n[1/5] Инициализация навигатора...")
    navigator = AerialNavigator(
        map_loader=MapLoader(),
        resolution=0.1,
        use_cuda=False  # False если нет GPU
    )

    # 2. Загружаем карту
    print("\n[2/5] Загрузка карты...")
    navigator.initialize(
        map_path=map_path,
        lat=55.7550,
        lon=37.6173,
        synthetic=synthetic
    )

    # 3. Создаём симулятор
    print("\n[3/5] Настройка симулятора...")
    camera_config = CameraConfig(
        fov=60.0,
        width=512,
        height=512,
        resolution=0.1
    )

    simulator = FlightSimulator(
        map_loader=navigator.map_loader,
        map_region=navigator.map_loader.get_region(),
        camera_config=camera_config
    )

    # Задаём начальный путь (спираль)
    path = []
    for i in range(50):
        angle = np.radians(i * 15)
        radius = 5 + i * 0.5
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 50 + np.sin(i * 0.2) * 10
        path.append((x, y, z))

    simulator.follow_path(path, speed=3.0)

    # 4. Инициализируем GUI
    print("\n[4/5] Инициализация визуализации...")
    gui = NavigationGUI(show_plot=show_gui)
    if show_gui:
        gui.open()

    # 5. Запускаем симуляцию
    print("\n[5/5] Запуск симуляции...")
    print("-" * 60)

    position_history = []
    velocity_history = []
    heading_history = []

    try:
        for frame in range(num_frames):
            # Обновляем состояние дрона
            simulator.update_path_following(dt=1/30.0)
            drone_state = simulator.get_ground_truth()

            # Получаем кадр с камеры
            camera_frame = simulator.get_camera_frame()

            # Обрабатываем кадр навигатором
            nav_data = navigator.process_frame(
                camera_frame,
                timestamp=drone_state.timestamp
            ) or {}

            # Собираем историю для визуализации
            if nav_data.get('position'):
                position_history.append(
                    [nav_data['position'].get('x', 0) or 0,
                     nav_data['position'].get('y', 0) or 0]
                )
            if nav_data.get('velocity'):
                velocity_history.append(
                    nav_data['velocity'].get('speed_ms', 0) or 0
                )
            if nav_data.get('heading') is not None:
                heading_history.append(nav_data['heading'])

            # Получаем фрагмент карты для визуализации
            map_crop = navigator.map_loader.get_map_crop(
                lat=55.7550 + drone_state.x / 100,
                lon=37.6173 + drone_state.y / 100,
                width_meters=100,
                height_meters=100
            )

            # Обновляем GUI
            if show_gui and frame % 3 == 0:  # обновляем каждые 3 кадра
                gui.update(
                    camera_frame=camera_frame,
                    map_crop=map_crop,
                    nav_data=nav_data,
                    drone_state=drone_state,
                    position_history=position_history,
                    velocity_history=velocity_history,
                    heading_history=heading_history
                )

            # Печатаем статистику каждые 30 кадров
            if frame % 30 == 0:
                vel = nav_data.get('velocity') or {}
                pos = nav_data.get('position') or {}
                speed = vel.get('speed_kmh', 0) or 0
                heading = nav_data.get('heading') or 0
                confidence = nav_data.get('confidence') or 0
                matches = nav_data.get('num_matches') or 0
                print(f"Кадр {frame:4d} | "
                      f"Скорость: {speed:5.1f} км/ч | "
                      f"Направление: {heading:5.1f}° | "
                      f"Уверенность: {confidence:.2f} | "
                      f"Совпадений: {matches}")

    except KeyboardInterrupt:
        print("\n\nСимуляция прервана пользователем")
    finally:
        if show_gui:
            gui.close()

    # Итоговая статистика
    print("\n" + "=" * 60)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60)

    if position_history:
        x_final = position_history[-1][0]
        y_final = position_history[-1][1]
        total_distance = np.sqrt(x_final**2 + y_final**2)
        print(f"Итоговое положение: ({x_final:.1f}, {y_final:.1f}) м")
        print(f"Общее расстояние: {total_distance:.1f} м")

    if velocity_history:
        avg_speed = np.mean(velocity_history)
        max_speed = np.max(velocity_history)
        print(f"Средняя скорость: {avg_speed:.2f} м/с ({avg_speed*3.6:.1f} км/ч)")
        print(f"Максимальная скорость: {max_speed:.2f} м/с ({max_speed*3.6:.1f} км/ч)")

    if heading_history:
        print(f"Направление: {heading_history[-1]:.1f}°")

    print(f"\nВсего кадров обработано: {num_frames}")
    print("=" * 60)


def run_demo():
    """Запустить демо с одним кадром для проверки."""
    print("ДЕМО: обработка одного кадра")
    print("-" * 40)

    # Создаём навигатор
    navigator = AerialNavigator(
        map_loader=MapLoader(),
        resolution=0.1,
        use_cuda=False
    )

    # Создаём синтетическую карту
    navigator.initialize(synthetic=True)

    # Создаём тестовый кадр
    test_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    test_frame[:, :] = [34, 139, 34]  # зелёный фон
    # Добавляем "здания"
    test_frame[100:200, 100:200] = [128, 128, 128]
    test_frame[300:400, 250:350] = [128, 128, 128]

    # Обрабатываем кадр
    nav_data = navigator.process_frame(test_frame, timestamp=0.0)

    print(f"Положение: x={nav_data.get('position', {}).get('x_meters', 0):.2f}, "
          f"y={nav_data.get('position', {}).get('y_meters', 0):.2f} м")
    print(f"Скорость: {nav_data.get('velocity', {}).get('speed_ms', 0):.2f} м/с")
    print(f"Направление: {nav_data.get('heading', 0):.1f}°")
    print(f"Уверенность: {nav_data.get('confidence', 0):.2f}")
    print(f"Совпадений: {nav_data.get('num_matches', 0)}")

    # Показываем результат
    navigator.show_static(
        camera_frame=test_frame,
        map_crop=test_frame,
        nav_data=nav_data
    )


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        run_demo()
    elif len(sys.argv) > 1 and sys.argv[1] == 'real':
        # Загрузка реальной карты
        lat = float(sys.argv[2]) if len(sys.argv) > 2 else 55.7550  # Москва
        lon = float(sys.argv[3]) if len(sys.argv) > 3 else 37.6173
        radius = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
        zoom_str = sys.argv[5] if len(sys.argv) > 5 else 'medium'

        zoom_levels = {'low': 12, 'medium': 14, 'high': 16, 'ultra': 18}
        zoom = zoom_levels.get(zoom_str, 14)

        print("=" * 60)
        print("ЗАГРУЗКА РЕАЛЬНОЙ КАРТЫ")
        print("=" * 60)
        print(f"Координаты: {lat}, {lon}")
        print(f"Радиус: {radius} км")
        print(f"Детализация: {zoom_str} (zoom={zoom})")

        real_loader = RealMapLoader(tile_source='esri')
        region = real_loader.load_real_map(lat, lon, radius, zoom)

        # Сохраняем карту
        output_path = os.path.join(os.path.dirname(__file__), 'map_cache', 'real_map.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        real_loader.save_map_to_file(output_path)

        print(f"\nКарта сохранена в {output_path}")
        print(f"Размер: {region.tiles[(0,0)].image.shape[1]}x{region.tiles[(0,0)].image.shape[0]} пикселей")
        print(f"Разрешение: {region.resolution:.2f} м/пиксель")
    else:
        run_simulation(
            num_frames=300,
            show_gui=True,
            synthetic=True
        )
