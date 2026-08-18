"""
Расчёт параметров камеры и карты для CUAV X7+ Pro.
"""

import math


class CameraCalculator:
    """Калькулятор параметров камеры и разрешения земли."""

    # Параметры камеры OpenIPC с 30x зумом
    SENSOR_WIDTH_MM = 6.17  # 1/2.3" sensor
    SENSOR_HEIGHT_MM = 4.55
    PIXEL_SIZE_UM = 1.12  # размер пикселя в микронах
    BASE_FOCAL_MM = 4.24  # фокусное расстояние на широком угле

    @classmethod
    def get_focal_length(cls, zoom: int = 30) -> float:
        """Фокусное расстояние при заданном зуме (мм)."""
        return cls.BASE_FOCAL_MM * zoom

    @classmethod
    def get_fov(cls, zoom: int = 30) -> tuple:
        """Поле зрения (горизонталь, вертикаль) в градусах."""
        focal = cls.get_focal_length(zoom)
        fov_h = 2 * math.degrees(math.atan(cls.SENSOR_WIDTH_MM / (2 * focal)))
        fov_v = 2 * math.degrees(math.atan(cls.SENSOR_HEIGHT_MM / (2 * focal)))
        return fov_h, fov_v

    @classmethod
    def get_ground_resolution(cls, altitude_m: float, zoom: int = 30) -> float:
        """
        Разрешение земли (м/пиксель) на заданной высоте.
        
        При 850м и 30x зуме: ~0.05 м/пиксель (5 см/пиксель)
        """
        focal = cls.get_focal_length(zoom)
        # GSD = (altitude * pixel_size) / focal
        gsd = (altitude_m * cls.PIXEL_SIZE_UM / 1e6) / focal
        return gsd

    @classmethod
    def get_ground_coverage(cls, altitude_m: float, zoom: int = 30,
                            resolution_px: tuple = (1920, 1080)) -> tuple:
        """
        Площадь покрытия земли (ширина, высота) в метрах.
        """
        fov_h, fov_v = cls.get_fov(zoom)
        fov_h_rad = math.radians(fov_h)
        fov_v_rad = math.radians(fov_v)
        
        width = 2 * altitude_m * math.tan(fov_h_rad / 2)
        height = 2 * altitude_m * math.tan(fov_v_rad / 2)
        return width, height

    @classmethod
    def get_required_map_resolution(cls, altitude_m: float, zoom: int = 30) -> float:
        """
        Требуемое разрешение карты.
        Карта должна быть в 2-3 раза детальнее кадра для надёжного сопоставления.
        """
        gsd = cls.get_ground_resolution(altitude_m, zoom)
        return gsd / 2.5  # 2x запас


if __name__ == '__main__':
    calc = CameraCalculator()
    
    print("=" * 60)
    print("РАСЧЁТ ПАРАМЕТРОВ КАМЕРЫ (OpenIPC, 30x зум)")
    print("=" * 60)
    
    altitudes = [700, 850, 1000]
    
    for alt in altitudes:
        gsd = calc.get_ground_resolution(alt, zoom=30)
        map_res = calc.get_required_map_resolution(alt, zoom=30)
        fov_h, fov_v = calc.get_fov(zoom=30)
        coverage_w, coverage_h = calc.get_ground_coverage(alt, zoom=30)
        
        print(f"\nВысота: {alt} м")
        print(f"  FOV: {fov_h:.1f}° × {fov_v:.1f}°")
        print(f"  Покрытие земли: {coverage_w:.0f} × {coverage_h:.0f} м")
        print(f"  Разрешение земли (GSD): {gsd*100:.2f} см/пиксель")
        print(f"  Требуемое разрешение карты: {map_res*100:.2f} см/пиксель")
        print(f"  Фокусное расстояние: {calc.get_focal_length(30):.1f} мм")
    
    print("\n" + "=" * 60)
    print("ВЫВОДЫ:")
    print("=" * 60)
    print("- Карта должна быть ~0.02-0.05 м/пиксель (2-5 см/пиксель)")
    print("- Это разрешение DroneDeploy / Pix4D / Mapbox")
    print("- Обычные спутниковые карты (ESRI 0.3-2 м/пиксель) НЕ ПОДХОДЯТ")
    print("- Нужно скачивать ортофотопланы для конкретного региона")
