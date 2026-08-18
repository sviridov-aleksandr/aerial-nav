from setuptools import setup

package_name = 'aerial_nav_ros'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Alex',
    maintainer_email='alex@example.com',
    description='GPS-denied drone navigation using aerial map matching with ROS 2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'nav_node = aerial_nav_ros.nav_node:main',
            'sim_node = aerial_nav_ros.sim_node:main',
        ],
    },
)
