"""
Расчёт параметров камеры и карты для CUAV X7+ Pro + OpenIPC MC800S-V3.
"""

import math


class CameraCalculator:
    """Калькулятор параметров камеры и разрешения земли."""

    # Параметры камеры OpenIPC MC800S-V3 (Sony IMX415, объектив ASX-0116 KH)
    SENSOR_WIDTH_MM = 6.44  # 1/2.5" sensor (IMX415)
    SENSOR_HEIGHT_MM = 3.63
    PIXEL_SIZE_UM = 1.45  # размер пикселя в микронах
    CAMERA_RESOLUTION = (3840, 2160)  # 4K
    FOV_H = 90.0  # градусов (фиксированный объектив, без зума)
    FOV_V = 51.0  # градусов (соотношение 16:9)

    @classmethod
    def get_fov(cls) -> tuple:
        """Поле зрения (горизонталь, вертикаль) в градусах."""
        return cls.FOV_H, cls.FOV_V

    @classmethod
    def get_ground_resolution(cls, altitude_m: float) -> float:
        """
        Разрешение земли (м/пиксель) на заданной высоте.
        При 1000 м и FOV 90°: ~0.4 м/пиксель
        """
        fov_h_rad = math.radians(cls.FOV_H)
        width = cls.CAMERA_RESOLUTION[0]
        gsd = (altitude_m * 2 * math.tan(fov_h_rad / 2)) / width
        return gsd

    @classmethod
    def get_ground_coverage(cls, altitude_m: float) -> tuple:
        """Площадь покрытия земли (ширина, высота) в метрах."""
        fov_h_rad = math.radians(cls.FOV_H)
        fov_v_rad = math.radians(cls.FOV_V)
        width = 2 * altitude_m * math.tan(fov_h_rad / 2)
        height = 2 * altitude_m * math.tan(fov_v_rad / 2)
        return width, height

    @classmethod
    def get_required_map_resolution(cls, altitude_m: float) -> float:
        """
        Требуемое разрешение карты.
        Карта должна быть в 2-3 раза детальнее кадра для надёжного сопоставления.
        """
        gsd = cls.get_ground_resolution(altitude_m)
        return gsd / 2.5


if __name__ == '__main__':
    calc = CameraCalculator()

    print("=" * 60)
    print("РАСЧЁТ ПАРАМЕТРОВ КАМЕРЫ (OpenIPC MC800S-V3, IMX415)")
    print("=" * 60)

    altitudes = [700, 1000, 1200]

    for alt in altitudes:
        gsd = calc.get_ground_resolution(alt)
        map_res = calc.get_required_map_resolution(alt)
        fov_h, fov_v = calc.get_fov()
        coverage_w, coverage_h = calc.get_ground_coverage(alt)

        print(f"\nВысота: {alt} м")
        print(f"  FOV: {fov_h:.1f}° × {fov_v:.1f}°")
        print(f"  Покрытие земли: {coverage_w:.0f} × {coverage_h:.0f} м")
        print(f"  Разрешение земли (GSD): {gsd*100:.1f} см/пиксель")
        print(f"  Требуемое разрешение карты: {map_res*100:.1f} см/пиксель")

    print("\n" + "=" * 60)
    print("ВЫВОДЫ:")
    print("=" * 60)
    print("- GSD ~0.3-0.5 м/пиксель на рабочих высотах")
    print("- Спутниковые карты zoom 19 (0.5 м/px) подходят для навигации")
    print("- Google Satellite даёт лучшее качество, чем ESRI в ряде регионов")