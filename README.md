# aerial-nav

Визуальная навигация БПЛА без GPS (GPS-denied navigation).

## Описание

Система определения местоположения дрона по камере: кадр с бортовой камеры
сопоставляется с заранее подготовленной спутниковой картой маршрута.

Основные компоненты:

- **Siamese-сеть** (ResNet-18 backbone) — извлечение эмбеддингов из кадров камеры и тайлов карты;
- **SAFA-архитектура** (Spatial-Aware Feature Aggregation, Shi et al., NeurIPS 2019) — M=8 мягких пространственных зон для позиционной чувствительности эмбеддингов;
- **Многоуровневый индекс карты** — эмбеддинги тайлов на 13 высотных уровнях (0–1200 м);
- **Гибридный матчинг** — Siamese-ранжирование кандидатов + SuperPoint/LightGlue с RANSAC-гомографией для точного позиционирования и оценки качества матча;
- **UKF-фильтр** — слияние визуальных измерений с инерциальной навигацией;
- **MAVLink-интеграция** — передача VISION_POSITION_ESTIMATE в ArduPilot;
- **SITL-стенд** — симуляция полного контура навигации (ArduPilot SITL + генератор кадров + vision-узел).

## Результаты (SITL, маршрут ~5400 кадров)

| Конфигурация | Медианная ошибка | Покрытие |
|---|---|---|
| Чистый Siamese (глобальный эмбеддинг) | ~1793 м | 6% |
| Cell-divided + LightGlue | 0 м | 61% |
| SAFA + LightGlue | 0 м | 100% |

## Структура

```
safa_network.py        — SAFA-архитектура (M=8 зон, min-дистанция по парам)
cell_network.py        — cell-divided архитектура (4×4 ячеек) + CBAM
siamese_triplet_dataset.py — датасет триплетов (anchor/positive/negative)
train_safa.py          — обучение SAFA (BatchHard triplet loss)
train_cell.py          — обучение cell-divided
multi_level_index.py   — многоуровневый индекс карты
build_index_safa.py    — построение индекса эмбеддингов
sitl_vision_node.py    — бортовой vision-узел (Siamese + LightGlue + MAVLink)
sitl_frame_generator.py— генератор кадров для SITL
hybrid_matcher.py      — гибридный матчинг SuperPoint+LightGlue
ukf_nav.py             — UKF-фильтр слияния INS+vision
real_nav.py            — реальная навигация (RTSP-камера, RKNN/PyTorch)
web_route_planner/     — веб-планировщик маршрутов
```

## Аппаратная платформа

- Полётный контроллер: ArduPilot (BrotherHobby H743)
- Бортовой вычислитель: Jetson Orin Nano / Orange Pi 5 Plus (RKNN NPU, 69 FPS)
- Камера: OpenIPC FPV (Sony IMX415)

---

*Репозиторий очищен. Полный код доступен по запросу.*
