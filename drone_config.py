"""
Конфигурация для CUAV X7+ Pro + Jetson Orin Nano + Крыло.
"""

import os
import math


class DroneConfig:
    """Параметры дрона и оборудования."""
    
    # Полётный контроллер
    FC_TYPE = "CUAV_X7_PLUS_PRO"
    FC_PROTOCOL = "MAVLink2"
    FC_BAUD = 921600
    
    # Компьютер-компаньон
    COMP_COMP = "NVIDIA_JETSON_ORIN_NANO_8GB"
    USE_CUDA = True  # Jetson Orin Nano имеет GPU
    
    # Тип воздушного судна
    AIRCRAFT_TYPE = "FIXED_WING"  # Крыло (самолёт)
    
    # Полётные параметры
    MIN_ALTITUDE = 700.0  # м
    MAX_ALTITUDE = 1200.0  # м
    CRUISE_ALTITUDE = 1000.0  # м
    MIN_SPEED = 15.0  # м/с (54 км/ч)
    MAX_SPEED = 33.0  # м/с (120 км/ч)
    CRUISE_SPEED = 22.0  # м/с (79 км/ч)
    MIN_TURN_RADIUS = 100.0  # м (для крыла)
    
    # Камера
    CAMERA_TYPE = "OpenIPC_MC800S_V3"
    CAMERA_SENSOR = "Sony_IMX415"
    CAMERA_CHIPSET = "SigmaStar_SSC338Q"
    CAMERA_RESOLUTION = (3840, 2160)  # 4K (8 МП)
    CAMERA_FPS = 30
    CAMERA_LENS = "ASX-0116_KH"
    # Без оптического зума — фиксированный объектив
    CAMERA_FOV_H = 90.0  # градусов (согласно симулятору)
    CAMERA_FOV_V = 51.0  # градусов (соотношение 16:9)

    # Расчёт разрешения земли
    @staticmethod
    def get_ground_resolution(altitude_m: float) -> float:
        """
        Вычисляет разрешение земли (м/пиксель) на заданной высоте.
        При 1000 м и FOV 90°: ~0.4 м/пиксель
        """
        fov_rad = math.radians(DroneConfig.CAMERA_FOV_H)
        width = DroneConfig.CAMERA_RESOLUTION[0]
        resolution = (altitude_m * 2 * math.tan(fov_rad / 2)) / width
        return resolution
    
    # Параметры карты
    @staticmethod
    def get_required_map_resolution(altitude: float = 850.0) -> float:
        """Требуемое разрешение карты для навигации."""
        ground_res = DroneConfig.get_ground_resolution(altitude)
        # Карта должна быть в 2-3 раза детальнее кадра
        return ground_res / 2.5
    
    # Нейросеть
    @staticmethod
    def get_network_input_size() -> int:
        """Размер входа нейросети."""
        return 224  # 224x224 пикселей
    
    @staticmethod
    def get_network_embedding_dim() -> int:
        """Размер embedding."""
        return 256  # Увеличено для высокой детализации


# Пример использования
if __name__ == '__main__':
    cfg = DroneConfig()
    alt = 850.0
    print(f"Высота полёта: {alt} м")
    print(f"Разрешение земли: {cfg.get_ground_resolution(alt):.3f} м/пиксель")
    print(f"Требуемое разрешение карты: {cfg.get_required_map_resolution(alt):.3f} м/пиксель")
    print(f"Использовать CUDA: {cfg.USE_CUDA}")
    print(f"Минимальный радиус разворота: {cfg.MIN_TURN_RADIUS} м")