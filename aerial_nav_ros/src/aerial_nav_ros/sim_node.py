"""
ROS 2 узел симуляции для тестирования навигации.
Генерирует синтетические данные камеры и IMU.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Header
import numpy as np
import cv2
from cv_bridge import CvBridge
import math
import time


class SimulationNode(Node):
    """ROS 2 узел симуляции полёта дрона."""

    def __init__(self):
        super().__init__('aerial_simulation')

        self.bridge = CvBridge()

        # Параметры симуляции
        self.declare_parameter('map_path', '')
        self.declare_parameter('resolution', 2.35)
        self.declare_parameter('drone_speed', 5.0)
        self.declare_parameter('drone_height', 50.0)

        map_path = self.get_parameter('map_path').get_parameter_value().string_value
        resolution = self.get_parameter('resolution').get_parameter_value().double_value
        self.drone_speed = self.get_parameter('drone_speed').get_parameter_value().double_value
        self.drone_height = self.get_parameter('drone_height').get_parameter_value().double_value

        # Загрузка карты
        from map_loader import MapLoader
        self.map_loader = MapLoader()
        if map_path:
            self.map_loader.load_from_image(map_path, lat=55.7550, lon=37.6173, resolution=resolution)
        else:
            self.map_loader.generate_synthetic_map(size=1024, resolution=resolution)

        # Состояние дрона
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = self.drone_height
        self.drone_vx = self.drone_speed
        self.drone_vy = self.drone_speed * 0.6
        self.start_time = time.time()

        # Публикации
        self.camera_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)

        # Таймер (30 Hz)
        self.timer = self.create_timer(1/30.0, self.simulation_step)

        self.get_logger().info('Aerial Simulation Node запущен')
        self.get_logger().info(f'Скорость: {self.drone_speed} м/с, Высота: {self.drone_height} м')

    def simulation_step(self):
        """Один шаг симуляции."""
        dt = 1/30.0
        now = time.time()
        elapsed = now - self.start_time

        # Обновление позиции (спираль)
        angle = elapsed * 0.2
        radius = 10 + elapsed * 0.1
        self.drone_x = radius * math.cos(angle)
        self.drone_y = radius * math.sin(angle)
        self.drone_z = self.drone_height + math.sin(elapsed * 0.5) * 5

        # Генерация кадра камеры
        self.publish_camera_frame(elapsed)

        # Генерация IMU
        self.publish_imu(elapsed)

    def publish_camera_frame(self, elapsed: float):
        """Генерация кадра с камеры."""
        from simulator import FlightSimulator, CameraConfig

        sim = FlightSimulator(
            map_loader=self.map_loader,
            map_region=self.map_loader.get_region(),
            camera_config=CameraConfig(fov=60, width=640, height=480)
        )
        sim.set_drone_position(self.drone_x, self.drone_y, self.drone_z)
        sim.set_drone_velocity(self.drone_vx, self.drone_vy)

        frame = sim.get_camera_frame()

        msg = self.bridge.cv2_to_imgmsg(frame, 'rgb8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'

        self.camera_pub.publish(msg)

    def publish_imu(self, elapsed: float):
        """Генерация данных IMU."""
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Угловая скорость (дрон поворачивается)
        angle = elapsed * 0.2
        omega = 0.2 * math.cos(angle)
        msg.angular_velocity.x = 0.0
        msg.angular_velocity.y = 0.0
        msg.angular_velocity.z = omega

        # Ускорение (гравитация + ускорение дрона)
        msg.linear_acceleration.x = self.drone_vx * 0.01
        msg.linear_acceleration.y = self.drone_vy * 0.01
        msg.linear_acceleration.z = -9.81  # гравитация

        self.imu_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimulationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Остановка симуляции...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
