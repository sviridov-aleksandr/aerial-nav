# GPS-Denied Navigation System for UAV (Wing)

## Цель
Навигация без GPS для региона Каланчак с точностью < 15м.

## Результат
✓ **Median error: 10.1м** (цель: < 15м)
✓ **< 15м: 93.8%** (было: 4.4%)
✓ **Speedup: ×7843** (локальный поиск вместо полного)

---

## Архитектура

```
┌─────────────────────────────────────────────────────┐
│  CUAV 7+ Pro                                       │
│  ┌─────────┐  ┌──────────┐  ┌────────────────┐     │
│  │  IMU    │  │ Барометр │  │  GPS (старт)   │     │
│  └────┬────┘  └────┬────┘  └────────┬───────┘     │
│       │            │                 │              │
│       ▼            ▼                 │              │
│  ┌───────────────────────────────────┤              │
│  │    EKF FUSION                    │             │
│  │  State: [x, y, vx, vy, heading]   │              │
│  │  Predict: IMU (каждые 0.1с)       │              │
│  │  Update: Siamese match (каждые 5с)│              │
│  └────┬──────────────────────────────┘              │
│       │                                             │
│       ▼                                            │
│  ┌────────────────┐                               │
│ │  Local Matcher    │ → Siamese в окне ±50м        │
│  │  (3 тайла)        │ → ×7843 быстрее              │
│  └──────────────────┘                               │
└─────────────────────────────────────────────────────┘
```

---

## Компоненты

### 1. `local_matcher.py` — Локальный поиск
- Загружает Siamese модель и pre-extract embeddings всех тайлов карты
- Ищет match только в окне ±50м вокруг предсказанной позиции
- **3 кандидата вместо 25000** → ×7843 ускорение

### 2. `ekf_navigator.py` — EKF Fusion
- State: [x, y, vx, vy, heading]
- Predict: IMU odometry каждые 0.1с
- Update: Siamese map matching каждые 1с
- EKF correction по match с confidence > 0.5

### 3. `siamese_network.py` — Siamese модель
- AerialFeatureExtractor (ResNet-18 backbone)
- Embedding dim: 256
- Fine-tuned на Каланчаке (epoch 34, loss=0.358)

---

## Использование

```python
from local_matcher import LocalMatcher
from ekf_navigator import EKFNavigator

# 1. Инициализация
matcher = LocalMatcher(
    model_path='siamese_model_kalanchak_v2.pth',
    map_path='map_cache/highres/highres_46.2650_33.3732_z18.png',
    tile_size=224,
    resolution=0.5
)

ekf = EKFNavigator(matcher, map_resolution=0.5)

# 2. Старт (GPS)
ekf.init_from_gps(gps_x, gps_y)

# 3. Цикл полёта
while flying:
    # IMU predict
    velocity, heading = imu.read()
    ekf.predict(velocity, heading, dt=0.1)
    
    # Map matching каждые 1 сек
    if should_correct():
        tile = camera.capture()
        match = ekf.update(tile, search_radius_m=50)
        if match is not None:
            ekf.correct(match)
    
    # Текущая позиция
    x, y = ekf.get_position()
```

---

## Требования к железу

| Компонент | Минимум | Рекомендовано |
|-----------|---------|---------------|
| GPU | RTX 3060 (6GB) | RTX 5080 (16GB) |
| CPU | 4 cores | 8 cores |
| RAM | 8GB | 16GB |
| Jetson | Orin Nano 8GB | Orin NX 16GB |

## Параметры полёта

| Параметр | Значение |
|----------|---------|
| Скорость | 80-120 км/ч (22-33 м/с) |
| Высота | 1000-1200 м |
| GR | ~1 м/пиксель |
| Tile | 224×224 (224м) |
| IMU | CUAV 7+ Pro (6-осевой) |
| Камера | OpenIPC (Global Shutter) |

---

## Тестирование

```bash
# Интеграционный тест
python3 test_integration.py

# Тест локального matcher
python3 local_matcher.py

# Тест EKF навигатора
python3 ekf_navigator.py

# Resume обучения
python3 train_kalanchak_v2.py --resume-from-last
python3 train_kalanchak_v2.py --resume-from-best
```

---

## Файлы

| Файл | Описание |
|------|----------|
| `local_matcher.py` | Локальный поиск в окне ±50м |
| `ekf_navigator.py` | EKF fusion IMU + map matching |
| `test_integration.py` | Полный интеграционный тест |
| `siamese_network.py` | Siamese модель |
| `train_kalanchak_v2.py` | Обучение модели v2 |
| `siamese_model_kalanchak_v2.pth` | Лучшая модель (epoch 34) |
| `map_cache/highres/` | Карта 0.5 м/пиксель |

---

## Сравнение подходов

| Метрика | Full Search | Local (±50м) | EKF+Local |
|---------|-------------|--------------|-----------|
| Mean | 3754м | 27.5м | **10.6м** |
| Median | 3727м | 25.6м | **10.1м** |
| < 15м | 4.4% | 20.6% | **93.8%** |
| < 56м | 8.3% | 98.0% | **100%** |
| Speedup | 1x | 7843x | **7843x** |

---

## Дальнейшие улучшения

1. **Multi-scale**: добавить tile 896×896 для уникальных объектов
2. **Route planning**: маршрут вдоль дорог/линий для частых коррекций
3. **Real camera test**: тест с реальной OpenIPC камерой
4. **Jetson deployment**: оптимизация для Orin Nano
5. **Data augmentation**: random crop из 2км×2км для обучения
