"""
Конфигурация навигации.
"""

from ament_index_python.packages import get_package_share_directory
import os


def get_config_dir():
    """Получить путь к директории конфигурации."""
    pkg_dir = get_package_share_directory('aerial_nav_ros')
    config_dir = os.path.join(pkg_dir, 'config')
    os.makedirs(config_dir, exist_ok=True)
    return config_dir