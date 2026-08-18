# Aerial Navigation ROS 2 Package

ROS 2 пакет для GPS-denied навигации дрона по подстилающей поверхности.

## Архитектура

```
┌─────────────────────────────────────────────────────────┐
│                    ROS 2 Workspace                       │
│                                                          │
│  ┌──────────────┐      ┌──────────────────────┐         │
│  │  sim_node    │─────▶│    nav_node          │         │
│  │  (симуляция) │      │    (навигация)       │         │
│  │              │      │                      │         │
│  │  Камера      │      │  SIFT/SuperPoint     │         │
│  │  IMU         │      │  BFMatcher/RANSAC    │         │
│  │  Позиция     │      │  Фильтр Калмана      │         │
│  └──────────────┘      └──────────┬───────────┘         │
│                                   │                      │
│                    ┌──────────────┴───────────┐         │
│                    │    RViz2 (визуализация)  │         │
│                    │  - Odometry              │         │
│                    │  - Pose                  │         │
│                    │  - Camera Feed           │         │
│                    │  - TF Frames             │         │
│                    └──────────────────────────┘         │
└─────────────────────────────────────────────────────────┘
```

## Публикуемые топики

| Топик | Тип | Описание |
|-------|-----|----------|
| `/camera/image_raw` | `sensor_msgs/Image` | Кадр с камеры дрона |
| `/imu/data` | `sensor_msgs/Imu` | Данные IMU |
| `/aerial_nav/odometry` | `nav_msgs/Odometry` | Оценка положения и скорости |
| `/aerial_nav/pose` | `geometry_msgs/PoseStamped` | Позиция в системе координат map |
| `/aerial_nav/twist` | `geometry_msgs/TwistStamped` | Скорость дрона |

## Подписываемые топики

| Топик | Тип | Описание |
|-------|-----|----------|
| `/camera/image_raw` | `sensor_msgs/Image` | Кадр с камеры (для nav_node) |
| `/imu/data` | `sensor_msgs/Imu` | Данные IMU (для nav_node) |

## Установка

### 1. Клонирование в ROS 2 workspace

```bash
# Создаём workspace
mkdir -p ~/aerial_nav_ws/src
cd ~/aerial_nav_ws/src

# Копируем пакет
cp -r /home/alex/aerial-nav/aerial_nav_ros/src/aerial_nav_ros .

# Копируем зависимости (Python)
cp -r /home/alex/aerial-nav/*.py ~/aerial_nav_ws/src/aerial_nav_ros/
cp /home/alex/aerial-nav/requirements.txt ~/aerial_nav_ws/src/aerial_nav_ros/
```

### 2. Установка Python зависимостей

```bash
cd ~/aerial_nav_ws/src/aerial_nav_ros
pip install -r requirements.txt
```

### 3. Сборка

```bash
cd ~/aerial_nav_ws
source /opt/ros/humble/setup.bash  # или rolling
colcon build --packages-select aerial_nav_ros
source install/setup.bash
```

## Запуск

### Симуляция

```bash
# Запуск симуляции + навигации
ros2 launch aerial_nav_ros full.launch.py

# Или по отдельности
ros2 run aerial_nav_ros sim_node --ros-args -p map_path:=/path/to/map.png
ros2 run aerial_nav_ros nav_node --ros-args -p map_path:=/path/to/map.png
```

### С реальным дроном

```bash
# Запуск навигации с реальной камерой
ros2 run aerial_nav_ros nav_node \
  --ros-args \
  -p map_path:=/home/alex/aerial-nav/map_cache/real_map.png \
  -p resolution:=2.35 \
  -p use_cuda:=false
```

## Параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `map_path` | string | `""` | Путь к файлу карты |
| `resolution` | double | `2.35` | Разрешение карты (м/пиксель) |
| `camera_frame` | string | `"camera_link"` | Имя фрейма камеры |
| `base_frame` | string | `"base_link"` | Имя базового фрейма дрона |
| `use_cuda` | bool | `false` | Использовать GPU |

## Интеграция с ROS 2 Navigation Stack

Пакет совместим с `nav2`:

```bash
# Публикуем odometry в стандартный топик
ros2 topic pub /odom nav_msgs/msg/Odometry "..."

# TF2 трансформации уже публикуются автоматически
# map → base_link трансформация доступна через tf2_ros
```

## Структура пакета

```
aerial_nav_ros/
├── package.xml              # Метаданные пакета
├── setup.py                 # Скрипт установки
├── setup.cfg                # Конфигурация
├── nav_node.py              # Узел навигации
├── sim_node.py              # Узел симуляции
├── config.py                # Утилиты конфигурации
├── launch/
│   └── full.launch.py       # Полный launch файл
└── rviz/
    └── nav.rviz             # Конфигурация RViz
```

## Отладка

```bash
# Проверка топов
ros2 topic list

# Просмотр odometry
ros2 topic echo /aerial_nav/odometry

# Проверка TF
ros2 run tf2_tools view_frames

# Логи
ros2 run aerial_nav_ros nav_node --ros-args --log-level debug
```

## Лицензия

MIT
