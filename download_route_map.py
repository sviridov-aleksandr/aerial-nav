"""
Route Map Downloader — скачивание карты вдоль маршрута.

Использование:
  python3 download_route_map.py \
    --route 46.344208,32.353548 46.358646,32.332859 ... \
    --width 500 \
    --resolution 0.5 \
    --output route_map.png
"""

import numpy as np
from PIL import Image
import os
import sys
import argparse
import requests
from io import BytesIO


class RouteMapDownloader:
    """Скачивание карты вдоль маршрута из OpenStreetMap."""

    def __init__(self, route_gps, width_m=500, resolution_m=0.5):
        self.route_gps = route_gps
        self.width_m = width_m
        self.resolution_m = resolution_m

        # Вычисляем параметры
        self._compute_params()

    def _compute_params(self):
        """Вычисляем параметры маршрута."""
        lats = [p[0] for p in self.route_gps]
        lons = [p[1] for p in self.route_gps]

        self.min_lat = min(lats)
        self.max_lat = max(lats)
        self.min_lon = min(lons)
        self.max_lon = max(lons)

        # Длина маршрута
        total_dist = 0
        for i in range(len(self.route_gps) - 1):
            lat1, lon1 = self.route_gps[i]
            lat2, lon2 = self.route_gps[i + 1]
            dist = self._haversine(lat1, lon1, lat2, lon2)
            total_dist += dist
        self.total_distance = total_dist

        # Zoom level для разрешения
        if self.resolution_m <= 0.3:
            self.zoom = 20
        elif self.resolution_m <= 0.6:
            self.zoom = 19
        else:
            self.zoom = 18

        print(f"[RouteMap] Route: {len(self.route_gps)} points")
        print(f"[RouteMap] Distance: {self.total_distance/1000:.1f} км")
        print(f"[RouteMap] BBox: ({self.min_lat:.6f}, {self.min_lon:.6f}) to ({self.max_lat:.6f}, {self.max_lon:.6f})")
        print(f"[RouteMap] Zoom: {self.zoom}, Resolution: {self.resolution_m} м/px")

    def _haversine(self, lat1, lon1, lat2, lon2):
        """Расстояние в метрах."""
        R = 6371000
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    def _lon_to_tile(self, lon, zoom):
        return int((lon + 180) / 360 * (2 ** zoom))

    def _lat_to_tile(self, lat, zoom):
        lat_rad = np.radians(lat)
        n = 2.0 ** zoom
        y = int((1 - np.log(np.tan(lat_rad) + 1 / np.cos(lat_rad)) / np.pi) / 2 * n)
        return y

    def download_route_map(self, output_path):
        """Скачивание карты вдоль маршрута."""
        print(f"\n[RouteMap] Downloading tiles...")

        # Определяем тайлы
        tiles_needed = set()
        for lat, lon in self.route_gps:
            tx = self._lon_to_tile(lon, self.zoom)
            ty = self._lat_to_tile(lat, self.zoom)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    tiles_needed.add((tx + dx, ty + dy))

        print(f"[RouteMap] Tiles needed: {len(tiles_needed)}")

        # Скачиваем
        tiles = {}
        for i, (tx, ty) in enumerate(sorted(tiles_needed)):
            url = f"https://tile.openstreetmap.org/{self.zoom}/{tx}/{ty}.png"
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content)).convert('RGB')
                    tiles[(tx, ty)] = img
                    if (i + 1) % 50 == 0:
                        print(f"[RouteMap] Downloaded {i+1}/{len(tiles_needed)} tiles")
            except Exception as e:
                print(f"[RouteMap] Failed tile ({tx}, {ty}): {e}")

        print(f"[RouteMap] Downloaded {len(tiles)} tiles")

        # Создаём карту
        min_x = min(tx * 256 for tx, ty in tiles.keys())
        min_y = min(ty * 256 for tx, ty in tiles.keys())
        max_x = max((tx + 1) * 256 for tx, ty in tiles.keys())
        max_y = max((ty + 1) * 256 for tx, ty in tiles.keys())

        width = max_x - min_x
        height = max_y - min_y
        print(f"[RouteMap] Map size: {width}x{height} px")

        map_img = Image.new('RGB', (width, height), (128, 128, 128))
        for (tx, ty), img in tiles.items():
            x = tx * 256 - min_x
            y = ty * 256 - min_y
            map_img.paste(img, (x, y))

        # Сохраняем
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        map_img.save(output_path)
        print(f"[RouteMap] Saved: {output_path}")
        print(f"[RouteMap] Size: {os.path.getsize(output_path) / 1e6:.1f} MB")

        return map_img


def main():
    parser = argparse.ArgumentParser(description='Download map along route')
    parser.add_argument('--route', nargs='+', required=True,
                        help='Route points: lat1,lon1 lat2,lon2 ...')
    parser.add_argument('--width', type=float, default=500,
                        help='Map width in meters (default: 500)')
    parser.add_argument('--resolution', type=float, default=0.5,
                        help='Resolution in m/px (default: 0.5)')
    parser.add_argument('--output', type=str, default='route_map.png',
                        help='Output map path')

    args = parser.parse_args()

    route_gps = []
    for point in args.route:
        lat, lon = point.split(',')
        route_gps.append((float(lat), float(lon)))

    downloader = RouteMapDownloader(
        route_gps=route_gps,
        width_m=args.width,
        resolution_m=args.resolution
    )

    map_img = downloader.download_route_map(args.output)

    print(f"\n{'='*70}")
    print(f"ROUTE MAP DOWNLOADED")
    print(f"{'='*70}")
    print(f"  Points: {len(route_gps)}")
    print(f"  Distance: {downloader.total_distance/1000:.1f} км")
    print(f"  Map size: {map_img.size[0]}x{map_img.size[1]} px")
    print(f"  Resolution: {args.resolution} м/px")
    print(f"  Output: {args.output}")


if __name__ == '__main__':
    main()