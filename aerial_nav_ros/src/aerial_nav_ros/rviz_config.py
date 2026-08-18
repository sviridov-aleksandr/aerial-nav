"""
RViz конфигурация для визуализации навигации.
"""

from ament_index_python.packages import get_package_share_directory
import os


def get_rviz_config_path():
    """Получить путь к RViz конфигурации."""
    pkg_dir = get_package_share_directory('aerial_nav_ros')
    rviz_dir = os.path.join(pkg_dir, 'rviz')
    os.makedirs(rviz_dir, exist_ok=True)
    return os.path.join(rviz_dir, 'nav.rviz')
