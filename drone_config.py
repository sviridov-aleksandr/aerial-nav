"""
Конфигурация для CUAV X7+ Pro + Jetson Orin Nano + Крыло.
"""

import os


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
    MAX_ALTITUDE = 1000.0  # м
    CRUISE_ALTITUDE = 850.0  # м
    MIN_SPEED = 15.0  # м/с (54 км/ч)
    MAX_SPEED = 30.0  # м/с (108 км/ч)
    CRUISE_SPEED = 22.0  # м/с (79 км/ч)
    MIN_TURN_RADIUS = 100.0  # м (для крыла)
    
    # Камера
    CAMERA_TYPE = "DIGITAL_OPENIPC"
    CAMERA_RESOLUTION = (1920, 1080)  # FullHD
    CAMERA_FPS = 30
    CAMERA_ZOOM = 30  # 30x оптический зум
    CAMERA_FOV_H = 15.0  # градусов (при 30x зуме)
    CAMERA_FOV_V = 8.5  # градусов
    
    # Расчёт разрешения земли
    @staticmethod
    def get_ground_resolution(altitude_m: float) -> float:
        """
        Вычисляет разрешение земли (м/пиксель) на заданной высоте.
        
        При 1000м и 30x зуме: ~0.05-0.1 м/пиксель
        """
        # Упрощённая формула: resolution = altitude * tan(fov/2) / (width/2)
        fov_rad = DroneConfig.CAMERA_FOV_H * 3.14159 / 180.0
        width = DroneConfig.CAMERA_RESOLUTION[0]
        resolution = (altitude_m * fov_rad) / width
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