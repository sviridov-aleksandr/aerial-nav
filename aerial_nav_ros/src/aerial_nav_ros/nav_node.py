"""
ROS 2 узел навигации дрона.
Подписывается на камеру и IMU, публикует оценку положения.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Header
import numpy as np
import cv2
from cv_bridge import CvBridge


class AerialNavigationNode(Node):
    """ROS 2 узел GPS-denied навигации по подстилающей поверхности."""

    def __init__(self):
        super().__init__('aerial_navigation')

        # Параметры
        self.declare_parameter('map_path', '')
        self.declare_parameter('resolution', 2.35)
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('use_cuda', False)

        map_path = self.get_parameter('map_path').get_parameter_value().string_value
        resolution = self.get_parameter('resolution').get_parameter_value().double_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        use_cuda = self.get_parameter('use_cuda').get_parameter_value().boolean_value

        # Мост для изображений
        self.bridge = CvBridge()

        # Загрузка карты
        self.get_logger().info('Загрузка карты...')
        from map_loader import MapLoader
        from navigator import AerialNavigator

        self.map_loader = MapLoader()
        if map_path:
            self.map_loader.load_from_image(map_path, lat=55.7550, lon=37.6173, resolution=resolution)
        else:
            self.map_loader.generate_synthetic_map(size=1024, resolution=resolution)

        self.navigator = AerialNavigator(
            map_loader=self.map_loader,
            resolution=resolution,
            use_cuda=use_cuda
        )
        self.navigator.is_initialized = True
        self.navigator.map_loader = self.map_loader

        self.get_logger().info('Карта загружена. Запуск навигации...')

        # QoS для камеры (best effort, чтобы не отставать)
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        # Подписки
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            camera_qos
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Публикации
        self.odom_pub = self.create_publisher(
            Odometry,
            '/aerial_nav/odometry',
            10
        )

        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/aerial_nav/pose',
            10
        )

        self.twist_pub = self.create_publisher(
            TwistStamped,
            '/aerial_nav/twist',
            10
        )

        # TF2 broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Состояние
        self.last_imu_time = None
        self.last_camera_time = None
        self.frame_count = 0

        # Таймер для публикации odometry
        self.timer = self.create_timer(0.033, self.publish_odometry)  # ~30 Hz

        self.get_logger().info('Aerial Navigation Node запущен')
        self.get_logger().info(f'Подписка на: /camera/image_raw, /imu/data')
        self.get_logger().info(f'Публикация: /aerial_nav/odometry, /aerial_nav/pose, /aerial_nav/twist')

    def camera_callback(self, msg: Image):
        """Обработка кадра с камеры."""
        self.last_camera_time = msg.header.stamp

        # Конвертация изображения
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        except Exception as e:
            self.get_logger().warn(f'Ошибка конвертации изображения: {e}')
            return

        # Обработка через навигатор
        nav_data = self.navigator.process_frame(cv_image, timestamp=msg.header.stamp.sec)

        if nav_data and nav_data.get('position'):
            pos = nav_data['position']
            vel = nav_data.get('velocity', {})
            heading = nav_data.get('heading', 0)
            matches = nav_data.get('num_matches', 0)

            self.frame_count += 1

            if self.frame_count % 30 == 0:
                self.get_logger().info(
                    f'Кадр {self.frame_count} | '
                    f'Позиция: ({pos.get("x_meters", 0):.1f}, {pos.get("y_meters", 0):.1f}) м | '
                    f'Совпадений: {matches}'
                )

    def imu_callback(self, msg: Imu):
        """Обработка данных IMU для интеграции с навигацией."""
        self.last_imu_time = msg.header.stamp

        # IMU данные можно использовать для:
        # 1. Предсказания движения между кадрами камеры
        # 2. Компенсации наклона камеры
        # 3. Фьюжна с визуальной одометрией (EKF)
        pass

    def publish_odometry(self):
        """Публикация odometry в ROS 2."""
        if not self.navigator.current_pose:
            return

        pose = self.navigator.current_pose
        pos = pose.get('position', {})
        vel = pose.get('velocity', {})

        # Получаем текущее время
        now = self.get_clock().now()

        # Odometry
        odom = Odometry()
        odom.header = Header(stamp=now.to_msg(), frame_id='map')
        odom.child_frame_id = self.base_frame

        # Позиция
        odom.pose.pose.position.x = pos.get('x_meters', 0) or 0
        odom.pose.pose.position.y = pos.get('y_meters', 0) or 0
        odom.pose.pose.position.z = 0.0

        # Направление (quaternion)
        heading = pose.get('heading', 0) or 0
        q = self._heading_to_quaternion(heading)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        # Скорость
        odom.twist.twist.linear.x = vel.get('vx', 0) or 0
        odom.twist.twist.linear.y = vel.get('vy', 0) or 0
        odom.twist.twist.linear.z = 0.0
        odom.twist.twist.angular.z = 0.0

        self.odom_pub.publish(odom)

        # Pose
        pose_msg = PoseStamped()
        pose_msg.header = Header(stamp=now.to_msg(), frame_id='map')
        pose_msg.pose.position.x = odom.pose.pose.position.x
        pose_msg.pose.position.y = odom.pose.pose.position.y
        pose_msg.pose.position.z = odom.pose.pose.position.z
        pose_msg.pose.orientation = odom.pose.pose.orientation
        self.pose_pub.publish(pose_msg)

        # Twist
        twist_msg = TwistStamped()
        twist_msg.header = Header(stamp=now.to_msg(), frame_id=self.base_frame)
        twist_msg.twist.linear.x = odom.twist.twist.linear.x
        twist_msg.twist.linear.y = odom.twist.twist.linear.y
        twist_msg.twist.linear.z = odom.twist.twist.linear.z
        self.twist_pub.publish(twist_msg)

        # TF
        self.tf_broadcaster.sendTransform(
            (odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z),
            (q[0], q[1], q[2], q[3]),
            now.to_msg(),
            self.base_frame,
            'map'
        )

    def _heading_to_quaternion(self, heading_deg: float) -> tuple:
        """Конвертирует heading (градусы) в quaternion."""
        import math
        heading_rad = math.radians(heading_deg)
        q = [
            0.0,  # x
            0.0,  # y
            math.sin(heading_rad / 2),  # z
            math.cos(heading_rad / 2),  # w
        ]
        return q


def main(args=None):
    rclpy.init(args=args)
    node = AerialNavigationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Остановка узла...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
