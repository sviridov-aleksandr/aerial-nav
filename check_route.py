"""
Проверка маршрута на карте.

Карта: highres_46.2650_33.3732_z18.png
Размер: 17920×17920 px
Центр: (46.2650, 33.3732)
Границы:
  Lat: 46.2650 ± (17920*0.5/2)/111320 = 46.2650 ± 0.0403
  Lon: 33.3732 ± (17920*0.5/2)/(111320*cos(46.2650)) = 33.3732 ± 0.0415
"""

import numpy as np

# Параметры карты
MAP_CENTER_LAT = 46.2650
MAP_CENTER_LON = 33.3732
MAP_SIZE = 17920  # px
MAP_RESOLUTION = 0.5  # m/px

# Границы карты
lat_range = (MAP_SIZE * MAP_RESOLUTION / 2) / 111320.0
lon_m_per_deg = 111320.0 * np.cos(np.radians(MAP_CENTER_LAT))
lon_range = (MAP_SIZE * MAP_RESOLUTION / 2) / lon_m_per_deg

print("=" * 70)
print("ROUTE VALIDATION")
print("=" * 70)

print(f"\nMap bounds:")
print(f"  Lat: {MAP_CENTER_LAT - lat_range:.6f} to {MAP_CENTER_LAT + lat_range:.6f}")
print(f"  Lon: {MAP_CENTER_LON - lon_range:.6f} to {MAP_CENTER_LON + lon_range:.6f}")

# Координаты маршрута
ROUTE_GPS = [
    ("Takeoff", 46.264929, 33.372986),
    ("WP1", 46.279537, 33.371246),
    ("WP2", 46.277625, 33.352702),
    ("WP3", 46.259422, 33.322994),
    ("WP4", 46.252893, 33.298535),
    ("WP5", 46.262919, 33.286605),
    ("WP6", 46.281208, 33.274915),
]

print(f"\n{'Point':<10} {'Lat':<15} {'Lon':<15} {'On Map':<10}")
print(f"  {'-'*50}")

for name, lat, lon in ROUTE_GPS:
    on_map = (
        MAP_CENTER_LAT - lat_range <= lat <= MAP_CENTER_LAT + lat_range and
        MAP_CENTER_LON - lon_range <= lon <= MAP_CENTER_LON + lon_range
    )
    status = "✓" if on_map else "✗ OFF MAP"
    print(f"  {name:<10} {lat:<15.6f} {lon:<15.6f} {status}")

# Преобразование GPS → пиксели
def gps_to_pixels(lat, lon):
    lat_m_per_deg = 111320.0
    lon_m_per_deg = 111320.0 * np.cos(np.radians(lat))
    dy = (MAP_CENTER_LAT - lat) * lat_m_per_deg
    dx = (lon - MAP_CENTER_LON) * lon_m_per_deg
    center_px = MAP_SIZE // 2
    x = int(center_px + dx / MAP_RESOLUTION)
    y = int(center_px - dy / MAP_RESOLUTION)
    return x, y

print(f"\n{'Point':<10} {'X(px)':<10} {'Y(px)':<10} {'In Bounds':<10}")
print(f"  {'-'*40}")

for name, lat, lon in ROUTE_GPS:
    x, y = gps_to_pixels(lat, lon)
    in_bounds = 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE
    status = "✓" if in_bounds else "✗"
    print(f"  {name:<10} {x:<10} {y:<10} {status}")

# Расчёты расстояний
print(f"\nDistances between waypoints:")
for i in range(len(ROUTE_GPS) - 1):
    name1, lat1, lon1 = ROUTE_GPS[i]
    name2, lat2, lon2 = ROUTE_GPS[i + 1]
    
    x1, y1 = gps_to_pixels(lat1, lon1)
    x2, y2 = gps_to_pixels(lat2, lon2)
    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2) * MAP_RESOLUTION
    
    print(f"  {name1} → {name2}: {dist/1000:.2f} км")
