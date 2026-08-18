#!/bin/bash
# download_map.sh — скачивание карты вдоль маршрута (с прогресс-баром)
#
# Использование:
#   ./download_map.sh                    # маршрут по умолчанию (6 точек, Днепр)
#   ./download_map.sh --route-file r.json  # маршрут из файла route_editor.py
#
# Внимание: удаляет старый кэш тайлов map_cache/strips (тайлы для
# неправильной местности из-за бага lat/lon в route_strip_map.py).

set -e
cd "$(dirname "$0")"
source venv/bin/activate

ROUTE_FILE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --route-file)
            ROUTE_FILE="$2"
            shift 2
            ;;
        *)
            echo "Неизвестный аргумент: $1"
            exit 1
            ;;
    esac
done

# Маршрут по умолчанию (6 точек, долина Днепра)
ROUTE_POINTS=(
    "46.344208,32.353548"
    "46.358646,32.332859"
    "46.361671,32.181069"
    "46.491890,32.079251"
    "46.668248,31.952009"
    "46.848488,31.990962"
)

# Если задан файл маршрута — читаем точки из него
if [[ -n "$ROUTE_FILE" ]]; then
    echo "Загрузка маршрута из: $ROUTE_FILE"
    ROUTE_POINTS=($(python3 -c "
import json, sys
with open('$ROUTE_FILE') as f:
    data = json.load(f)
for lat, lon in data['points']:
    print(f'{lat},{lon}')
"))
fi

echo "================================================================"
echo "СКАЧИВАНИЕ КАРТЫ ВДОЛЬ МАРШРУТА"
echo "Точек: ${#ROUTE_POINTS[@]}"
echo "================================================================"

# Удаляем старый кэш (тайлы для неправильной местности)
if [[ -d map_cache/strips ]]; then
    echo "Удаление старого кэша тайлов (неправильная местность)..."
    rm -rf map_cache/strips
fi

# Запускаем скачивание (прогресс-бар встроен в route_strip_map.py)
python3 route_strip_map.py \
    --route "${ROUTE_POINTS[@]}" \
    --resolution 0.5 \
    --source esri \
    --point-size 1.0 \
    --seg-width 500 \
    --output map_cache/antiuav_route_strip.png

echo ""
echo "Готово! Карта: map_cache/antiuav_route_strip.png"
