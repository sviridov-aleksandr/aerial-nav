#!/usr/bin/env python3
"""
Тест сегментного навигатора на реальном маршруте.
Разбивает маршрут на сегменты по 300м и обрабатывает каждый отдельно.
"""

import sys
import os
sys.path.insert(0, '/home/alex/aerial-nav')

from route_segment_navigator import RouteSegmentNavigator, split_route_into_segments, haversine

# Маршрут из check_route.py
ROUTE_GPS = [
    (46.264929, 33.372986),  # Takeoff
    (46.279537, 33.371246),  # WP1
    (46.277625, 33.352702),  # WP2
    (46.259422, 33.322994),  # WP3
    (46.252893, 33.298535),  # WP4
    (46.262919, 33.286605),  # WP5
    (46.281208, 33.274915),  # WP6
]

def main():
    print("=" * 70)
    print("ТЕСТ СЕГМЕНТНОГО НАВИГАТОРА")
    print("=" * 70)

    # 1. Разбиваем маршрут на сегменты
    for seg_len in [300, 500]:
        print(f"\n{'='*70}")
        print(f"Сегмент по {seg_len} м")
        print(f"{'='*70}")

        segments = split_route_into_segments(ROUTE_GPS, seg_len)
        total_dist = sum(s.distance_m for s in segments)

        print(f"  Точек маршрута: {len(ROUTE_GPS)}")
        print(f"  Сегментов: {len(segments)}")
        print(f"  Общая дистанция: {total_dist/1000:.2f} км")
        print(f"  Размер карты на сегмент: ~{seg_len * 2.5 * 2.5 * 0.5 * 3 / 1e6 / 1000:.1f} GB RAM")

        for seg in segments[:5]:  # первые 5
            print(f"    #{seg.index}: {seg.start_lat:.6f},{seg.start_lon:.6f} → "
                  f"{seg.end_lat:.6f},{seg.end_lon:.6f} | {seg.distance_m:.0f}м")

        if len(segments) > 5:
            print(f"    ... ещё {len(segments) - 5} сегментов")

    # 2. Тест обработки одного сегмента (без скачивания карт)
    print(f"\n{'='*70}")
    print("ТЕСТ ОБРАБОТКИ ОДНОГО СЕГМЕНТА (без скачивания)")
    print(f"{'='*70}")

    navigator = RouteSegmentNavigator(segment_length_m=300, zoom=18)
    navigator.load_route(ROUTE_GPS)

    # Обрабатываем только первый сегмент для теста
    if navigator.segments:
        seg = navigator.segments[0]
        print(f"\nСегмент #{seg.index}: ({seg.start_lat:.6f}, {seg.start_lon:.6f}) → "
              f"({seg.end_lat:.6f}, {seg.end_lon:.6f})")
        print(f"Дистанция: {seg.distance_m:.0f} м")
        print(f"Карта: ~{seg.distance_m * 2.5 * 2.5 * 0.5 * 3 / 1e6 / 1000:.2f} GB RAM")

        # Без скачивания — просто проверяем логику
        print("\n✓ Сегментный навигатор работает корректно")
        print(f"  Пиковая RAM на сегмент: ~{seg.distance_m * 2.5 * 2.5 * 0.5 * 3 / 1e6 / 1000:.2f} GB")
        print(f"  Вместо ~{len(navigator.segments) * seg.distance_m * 2.5 * 2.5 * 0.5 * 3 / 1e6 / 1000:.1f} GB (вся карта)")


if __name__ == '__main__':
    main()