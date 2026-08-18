"""
Tile Indexer — генерация ANN-индекса (FAISS) по эмбеддингам тайлов карты.

Использует HNSW (Hierarchical Navigable Small World) для быстрого nearest-neighbor поиска.

Пример использования:
  # Генерация индекса
  python3 tile_indexer.py --model siamese_model_kalanchak_v2.pth \\
      --tiles /home/alex/aerial-nav/map_cache/strips/ \\
      --output tile_index.faiss

  # Поиск по кадру
  python3 tile_indexer.py --model siamese_model_kalanchak_v2.pth \\
      --index tile_index.faiss \\
      --query camera_frame.jpg --top-k 5
"""

import os
import sys
import argparse
import numpy as np
import faiss
import torch
from PIL import Image
from typing import List, Tuple, Optional

# Добавляем проект в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from siamese_network import AerialFeatureExtractor


class TileIndexer:
    """
    Генерация и использование ANN-индекса FAISS (HNSW) для тайлов карты.
    
    HNSW обеспечивает:
    - Скорость поиска O(log N) вместо O(N)
    - Точность ~95-99% при k=10
    - Поддержка метрик: IP (inner product), L2 (euclidean)
    """

    def __init__(self, model_path: Optional[str] = None, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.model = self._load_model(model_path)
        self.index: Optional[faiss.Index] = None
        self.tile_ids: List[str] = []
        self.tile_paths: List[str] = []
        self.tile_coords: List[Tuple[float, float]] = []  # (lat, lon)

    def _load_model(self, model_path: Optional[str]) -> AerialFeatureExtractor:
        """Загружает модель сиамской сети."""
        if model_path and os.path.exists(model_path):
            # Сначала загружаем state_dict чтобы определить embedding_dim
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            # Определяем embedding_dim из последнего linear слоя projection
            for key in ['projection.4.weight', 'projection.0.weight']:
                if key in state_dict:
                    self.embedding_dim = state_dict[key].shape[0]
                    break
            
            model = AerialFeatureExtractor(embedding_dim=self.embedding_dim)
            model.load_state_dict(state_dict)
            print(f"[TileIndexer] Модель загружена: {model_path} (embedding_dim={self.embedding_dim})")
        else:
            print(f"[TileIndexer] Модель не указана, используется случайная инициализация")
            model = AerialFeatureExtractor(embedding_dim=self.embedding_dim)
        model.eval()
        return model

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Извлекает embedding из изображения.
        
        Args:
            image: RGB изображение (H, W, 3), uint8 или float32
        
        Returns:
            embedding (1, embedding_dim), float32
        """
        # Приводим к float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        img = image / 255.0
        img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32) / np.array([0.229, 0.224, 0.225], dtype=np.float32))
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            embedding = self.model(tensor)
        
        return embedding.cpu().numpy().astype(np.float32)

    def build_index(self, tile_dir: str, metric: str = 'IP', 
                    M: int = 16, ef_construction: int = 200) -> faiss.Index:
        """
        Строит HNSW-индекс FAISS по всем тайлам в директории (рекурсивно).
        
        Args:
            tile_dir: директория с тайлами (файлы tile_<id>.png)
            metric: 'IP' (inner product, для L2-нормализованных embedding) или 'L2'
            M: количество связей на уровень HNSW (больше = точнее, медленнее)
            ef_construction: размер окна при построении индекса
        
        Returns:
            faiss.IndexHNSWFlat
        """
        print(f"\n[TileIndexer] Сканирование тайлов: {tile_dir}")
        
        embeddings_list = []
        self.tile_ids = []
        self.tile_paths = []
        self.tile_coords = []

        # Рекурсивный поиск всех PNG файлов
        tile_files = []
        for root, dirs, files in os.walk(tile_dir):
            for fname in files:
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    tile_files.append(os.path.join(root, fname))
        
        tile_files.sort()
        print(f"[TileIndexer] Найдено файлов: {len(tile_files)}")

        # Извлекаем embedding для каждого тайла
        for i, fpath in enumerate(tile_files):
            fname = os.path.basename(fpath)
            
            try:
                img = np.array(Image.open(fpath).convert('RGB'))
                emb = self.extract_embedding(img)
                embeddings_list.append(emb)
                
                # Парсим ID из имени файла
                tile_id = fname.rsplit('.', 1)[0]
                self.tile_ids.append(tile_id)
                self.tile_paths.append(fpath)
                
                # Если есть координаты в имени (lat_lon.png), парсим
                if '_' in tile_id:
                    parts = tile_id.split('_')
                    if len(parts) >= 3:
                        try:
                            lat = float(parts[-2])
                            lon = float(parts[-1].replace('.png', ''))
                            self.tile_coords.append((lat, lon))
                        except ValueError:
                            self.tile_coords.append((0.0, 0.0))
                    else:
                        self.tile_coords.append((0.0, 0.0))
                else:
                    self.tile_coords.append((0.0, 0.0))
                
                if (i + 1) % 100 == 0:
                    print(f"[TileIndexer] Извлечено {i+1}/{len(tile_files)} embedding")
                    
            except Exception as e:
                print(f"[TileIndexer] Ошибка тайла {fname}: {e}")

        if not embeddings_list:
            raise ValueError(f"Не найдено тайлов в {tile_dir}")

        # Конкатенируем в матрицу (N, D)
        embeddings = np.concatenate(embeddings_list, axis=0)
        print(f"[TileIndexer] Embeddings shape: {embeddings.shape}")

        # Строим HNSW индекс
        if metric == 'IP':
            # Inner product для L2-нормализованных embedding
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, M)
            self.index.hnsw.efConstruction = ef_construction
        else:
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, M)
            self.index.hnsw.efConstruction = ef_construction

        # Нормализуем embedding для IP метрики
        if metric == 'IP':
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            embeddings = embeddings / norms

        # Обучаем индекс
        print(f"[TileIndexer] Построение HNSW индекса (M={M}, ef_construction={ef_construction})...")
        self.index.add(embeddings)
        print(f"[TileIndexer] Индекс построен: {self.index.ntotal} векторов")

        return self.index

    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               ef_search: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """
        ANN-поиск по индексу.
        
        Args:
            query_embedding: (1, D) или (D,) embedding запроса
            top_k: количество ближайших соседей
            ef_search: размер окна поиска (больше = точнее, медленнее)
        
        Returns:
            (distances, indices) — расстояния и индексы тайлов
        """
        if self.index is None:
            raise ValueError("Индекс не построен. Вызовите build_index() сначала.")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Нормализуем query для IP метрики
        norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-10)
        query_embedding = query_embedding / norm

        # Устанавливаем ef_search
        self.index.hnsw.efSearch = ef_search

        # Ищем
        distances, indices = self.index.search(query_embedding, top_k)
        
        return distances, indices

    def search_by_image(self, image_path: str, top_k: int = 5,
                        ef_search: int = 64) -> List[dict]:
        """
        Поиск по изображению-запросу.
        
        Args:
            image_path: путь к изображению
            top_k: количество результатов
            ef_search: размер окна поиска
        
        Returns:
            Список результатов [{'tile_id', 'path', 'distance', 'coord'}, ...]
        """
        img = np.array(Image.open(image_path).convert('RGB'))
        query_emb = self.extract_embedding(img)
        
        distances, indices = self.search(query_emb, top_k=top_k, ef_search=ef_search)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]
            results.append({
                'tile_id': self.tile_ids[idx] if idx < len(self.tile_ids) else '?',
                'path': self.tile_paths[idx] if idx < len(self.tile_paths) else '?',
                'distance': float(dist),
                'coord': self.tile_coords[idx] if idx < len(self.tile_coords) else (0.0, 0.0)
            })
        
        return results

    def save_index(self, path: str):
        """Сохраняет индекс на диск."""
        faiss.write_index(self.index, path)
        
        # Сохраняем метаданные
        meta = {
            'tile_ids': self.tile_ids,
            'tile_paths': self.tile_paths,
            'tile_coords': self.tile_coords,
            'embedding_dim': self.embedding_dim
        }
        meta_path = path.replace('.faiss', '_meta.npy')
        np.save(meta_path, {
            'tile_ids': np.array(self.tile_ids, dtype=object),
            'tile_paths': np.array(self.tile_paths, dtype=object),
            'tile_coords': np.array(self.tile_coords),
            'embedding_dim': self.embedding_dim
        })
        print(f"[TileIndexer] Индекс сохранён: {path}")
        print(f"[TileIndexer] Метаданные: {meta_path}")

    def load_index(self, path: str):
        """Загружает индекс с диска."""
        self.index = faiss.read_index(path)
        
        meta_path = path.replace('.faiss', '_meta.npy')
        if os.path.exists(meta_path):
            meta = np.load(meta_path, allow_pickle=True).item()
            self.tile_ids = meta['tile_ids'].tolist()
            self.tile_paths = meta['tile_paths'].tolist()
            self.tile_coords = meta['tile_coords'].tolist()
            self.embedding_dim = meta['embedding_dim']
            print(f"[TileIndexer] Индекс загружен: {path} ({self.index.ntotal} векторов)")
        else:
            print(f"[TileIndexer] Индекс загружен: {path} ({self.index.ntotal} векторов)")
            print(f"[TileIndexer] ⚠ Метаданные не найдены: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description='Tile Indexer — FAISS HNSW для тайлов карты')
    parser.add_argument('--model', type=str, default=None,
                        help='Путь к модели сиамской сети (.pth)')
    parser.add_argument('--tiles', type=str, default=None,
                        help='Директория с тайлами (для построения индекса)')
    parser.add_argument('--index', type=str, default=None,
                        help='Путь к существующему индексу (для поиска)')
    parser.add_argument('--query', type=str, default=None,
                        help='Изображение-запрос (для поиска)')
    parser.add_argument('--output', type=str, default='tile_index.faiss',
                        help='Путь сохранения индекса')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Количество результатов поиска')
    parser.add_argument('--M', type=int, default=16,
                        help='HNSW M параметр (связей на уровень)')
    parser.add_argument('--ef-construction', type=int, default=200,
                        help='HNSW ef_construction')
    parser.add_argument('--ef-search', type=int, default=64,
                        help='HNSW ef_search')
    parser.add_argument('--metric', choices=['IP', 'L2'], default='IP',
                        help='Метрика: IP (для нормализованных) или L2')
    parser.add_argument('--embedding-dim', type=int, default=128,
                        help='Размер embedding')

    args = parser.parse_args()

    # Создаём индексер
    indexer = TileIndexer(
        model_path=args.model,
        embedding_dim=args.embedding_dim
    )

    if args.tiles:
        # Построение индекса
        print(f"\n{'='*70}")
        print("ГЕНЕРАЦИЯ ANN-ИНДЕКСА (FAISS HNSW)")
        print(f"{'='*70}")
        
        indexer.build_index(
            tile_dir=args.tiles,
            metric=args.metric,
            M=args.M,
            ef_construction=args.ef_construction
        )
        indexer.save_index(args.output)

    elif args.index:
        # Поиск
        print(f"\n{'='*70}")
        print("ПОИСК ПО ANN-ИНДЕКСУ (FAISS HNSW)")
        print(f"{'='*70}")
        
        indexer.load_index(args.index)
        
        if args.query:
            results = indexer.search_by_image(
                args.query,
                top_k=args.top_k,
                ef_search=args.ef_search
            )
            
            print(f"\nРезультаты поиска (top-{args.top_k}):")
            print(f"{'-'*70}")
            for i, r in enumerate(results):
                print(f"  [{i+1}] {r['tile_id']}")
                print(f"      Путь: {r['path']}")
                print(f"      Расстояние: {r['distance']:.4f}")
                print(f"      Координаты: {r['coord']}")
                print()
        else:
            print(f"Индекс: {args.index}")
            print(f"Векторов: {indexer.index.ntotal}")
            print(f"Дименсия: {indexer.embedding_dim}")
            print(f"Метрика: {args.metric}")
            print(f"Метаданные: {len(indexer.tile_ids)} тайлов")


if __name__ == '__main__':
    main()
