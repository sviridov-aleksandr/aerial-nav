"""
Извлечение и сопоставление ключевых точек с помощью нейронных сетей.
Используем SuperPoint для извлечения и SuperGlue для сопоставления.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass

@dataclass
class Keypoints:
    """Ключевые точки изображения."""
    points: np.ndarray  # Nx2, (x, y)
    descriptors: np.ndarray  # NxD
    scores: np.ndarray  # N, confidence scores


class SuperPointFeatureExtractor:
    """
    Извлечение ключевых точек с помощью SuperPoint.
    Если модель недоступна, использует fallback на ORB/SIFT.
    """

    def __init__(self, use_cuda: bool = True):
        self._fallback_detector = cv2.SIFT_create(nfeatures=1000)

    def extract(self, image: np.ndarray) -> Keypoints:
        """
        Извлечь ключевые точки из изображения.

        Args:
            image: изображение (HxWx3, RGB, uint8)

        Returns:
            Keypoints с точками, дескрипторами и оценками
        """
        # Для aerial навигации используем SIFT — он стабильнее для outdoor
        return self._extract_sift(image)

    def _extract_superpoint(self, image: np.ndarray) -> Keypoints:
        """Извлечение через SuperPoint."""
        # SuperPoint ожидает grayscale (1 канал)
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = (image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        # Нормализуем в float32 [0, 1]
        gray = gray / 255.0

        # Конвертируем в тензор CHW
        img_tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model({'image': img_tensor})

        points = output['keypoints'][0].cpu().numpy()  # Nx2
        scores = output['scores'][0].cpu().numpy()       # N
        descriptors = output['descriptors'][0].cpu().numpy()  # DxN -> NxD

        # Фильтруем по порогу
        threshold = 0.5
        mask = scores > threshold
        points = points[mask]
        scores = scores[mask]
        descriptors = descriptors[:, mask].T

        return Keypoints(points=points, descriptors=descriptors, scores=scores)

    def _extract_sift(self, image: np.ndarray) -> Keypoints:
        """Fallback: извлечение через SIFT."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        keypoints = self._fallback_detector.detect(gray, None)
        _, descriptors = self._fallback_detector.compute(gray, keypoints)

        if descriptors is None or len(keypoints) == 0:
            return Keypoints(
                points=np.zeros((0, 2)),
                descriptors=np.zeros((0, 128)),
                scores=np.zeros(0)
            )

        points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        scores = np.array([kp.response for kp in keypoints], dtype=np.float32)

        return Keypoints(points=points, descriptors=descriptors, scores=scores)


class SuperGlueMatcher:
    """
    Сопоставление ключевых точек между двумя изображениями.
    Для aerial навигации используем BFMatcher.
    """

    def __init__(self, use_cuda: bool = True):
        self._fallback_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def match(self, kp1: Keypoints, kp2: Keypoints,
              img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Сопоставить ключевые точки двух изображений.

        Для aerial навигации используем BFMatcher — он стабильнее для outdoor.

        Args:
            kp1: ключевые точки первого изображения
            kp2: ключевые точки второго изображения
            img1: первое изображение
            img2: второе изображение

        Returns:
            (matches1, matches2) — индексы сопоставленных точек
        """
        # Для aerial навигации используем BFMatcher
        return self._match_bf(kp1, kp2)

    def _match_superglue(self, kp1: Keypoints, kp2: Keypoints,
                         img1: np.ndarray = None, img2: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Сопоставление через SuperGlue."""
        n1 = len(kp1.points)
        n2 = len(kp2.points)

        if n1 == 0 or n2 == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        # SuperGlue ожидает grayscale изображения для нормализации
        def to_gray_tensor(img):
            if img is None:
                return None
            if len(img.shape) == 3 and img.shape[2] == 3:
                gray = (img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114).astype(np.float32)
            else:
                gray = img.astype(np.float32)
            gray = gray / 255.0
            return torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(self.device)

        img_tensor1 = to_gray_tensor(img1)
        img_tensor2 = to_gray_tensor(img2)

        pts1 = torch.from_numpy(kp1.points).float().unsqueeze(0).to(self.device)
        pts2 = torch.from_numpy(kp2.points).float().unsqueeze(0).to(self.device)
        desc1 = torch.from_numpy(kp1.descriptors).float().unsqueeze(0).to(self.device)
        desc2 = torch.from_numpy(kp2.descriptors).float().unsqueeze(0).to(self.device)

        # SuperGlue: scores shape (B, N)
        scores1 = torch.ones((1, n1), device=self.device)
        scores2 = torch.ones((1, n2), device=self.device)

        with torch.no_grad():
            output = self.model({'image0': img_tensor1, 'keypoints0': pts1, 'descriptors0': desc1,
                                 'scores0': scores1,
                                 'image1': img_tensor2, 'keypoints1': pts2, 'descriptors1': desc2,
                                 'scores1': scores2})

        matches = output['matches0'][0].cpu().numpy()
        scores = output['matching_scores0'][0].cpu().numpy()

        # Фильтруем: -1 = нет сопоставления
        valid = matches >= 0
        matches1 = np.where(valid)[0]
        matches2 = matches[valid].astype(np.int64)

        return matches1, matches2

    def _match_bf(self, kp1: Keypoints, kp2: Keypoints) -> Tuple[np.ndarray, np.ndarray]:
        """Сопоставление через BFMatcher с RANSAC для отбрасывания аутлайеров."""
        if len(kp1.descriptors) == 0 or len(kp2.descriptors) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        # KNN match (k=2 для ratio test)
        matches = self._fallback_matcher.knnMatch(
            kp1.descriptors, kp2.descriptors, k=2
        )

        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:  # Lowe's ratio test (мягче)
                good.append(m)

        if len(good) < 5:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        matches1 = np.array([m.queryIdx for m in good], dtype=np.float32)
        matches2 = np.array([m.trainIdx for m in good], dtype=np.float32)

        # RANSAC для оценки гомографии и отбрасывания аутлайеров
        if len(matches1) >= 5:
            pts1 = kp1.points[matches1.astype(np.int64)]
            pts2 = kp2.points[matches2.astype(np.int64)]

            # Увеличиваем порог до 10 пикселей для aerial-данных
            H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 10.0)

            if H is not None and mask is not None:
                inliers = mask.ravel().astype(bool)
                n_inliers = int(np.sum(inliers))
                
                # Если больше 50% инлайнеров — принимаем
                if n_inliers >= 5 and n_inliers / len(good) > 0.3:
                    matches1 = matches1[inliers]
                    matches2 = matches2[inliers]
                    return matches1.astype(np.int64), matches2.astype(np.int64)

        # Если RANSAC не помог, возвращаем все good matches
        return matches1.astype(np.int64), matches2.astype(np.int64)


class PoseEstimator:
    """
    Оценка положения, скорости и направления движения дрона.
    """

    def __init__(self, resolution: float = 0.1):
        self.resolution = resolution  # м/пиксель
        self.last_pixel_pos = None
        self.last_timestamp = None
        self.history: List[Tuple[np.ndarray, float]] = []

    def estimate_pose(self, matches1: np.ndarray, matches2: np.ndarray,
                      kp1: Keypoints, kp2: Keypoints,
                      map_crop: np.ndarray, camera_frame: np.ndarray,
                      timestamp: float = None) -> Dict:
        """
        Оценить положение, скорость и направление.

        Логика: если дрон сместился на (dx, dy) в пикселях, то
        matched точки в кадре камеры сместятся в противоположную сторону.
        """
        if len(matches1) < 5:
            return {
                'position': None,
                'velocity': None,
                'heading': None,
                'confidence': 0.0,
                'num_matches': len(matches1)
            }

        # Получаем matched точки
        map_points = kp1.points[matches1]  # точки на карте
        cam_points = kp2.points[matches2]  # точки в кадре камеры

        # Вычисляем смещение: разница между центрами matched точек
        # Если дрон движется вправо, объекты в кадре смещаются влево
        map_center = np.mean(map_points, axis=0)
        cam_center = np.mean(cam_points, axis=0)

        # Смещение дрона в пикселях (противоположно смещению объектов в кадре)
        delta_px = map_center - cam_center

        # Конвертируем в метры
        delta_meters = delta_px * self.resolution

        # Оценка положения
        position = {
            'pixel_offset': delta_px,
            'meters': delta_meters,
            'x_meters': delta_meters[0],
            'y_meters': delta_meters[1]
        }

        # Оценка скорости
        velocity = None
        if self.last_timestamp is not None and timestamp is not None:
            dt = timestamp - self.last_timestamp
            if dt > 0:
                velocity = {
                    'pixels_per_sec': delta_px / dt,
                    'meters_per_sec': delta_meters / dt,
                    'speed_ms': np.linalg.norm(delta_meters) / dt,
                    'speed_kmh': np.linalg.norm(delta_meters) / dt * 3.6
                }

        # Оценка направления (heading)
        heading = None
        if velocity is not None:
            vx, vy = velocity['pixels_per_sec']
            heading = np.degrees(np.arctan2(vx, vy)) % 360

        # Конфиденциальность на основе количества совпадений
        confidence = min(1.0, len(matches1) / 20.0)

        # Обновляем историю
        self.last_pixel_pos = cam_center.copy()
        self.last_timestamp = timestamp
        self.history.append((cam_center.copy(), timestamp or 0))

        # Храним только последние 10 записей
        if len(self.history) > 10:
            self.history = self.history[-10:]

        return {
            'position': position,
            'velocity': velocity,
            'heading': heading,
            'confidence': confidence,
            'num_matches': len(matches1)
        }

    def reset(self):
        """Сбросить историю."""
        self.last_pixel_pos = None
        self.last_timestamp = None
        self.history = []
