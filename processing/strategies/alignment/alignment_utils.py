"""
Общие утилиты для стратегий совмещения изображений
"""

from typing import Any, Dict, List, Optional, Tuple

# pylint: disable=no-member
import cv2
import numpy as np
from scipy.spatial import cKDTree  # type: ignore

from infrastructure.debug_formatter import DebugFormatter


class AlignmentUtils:
    """Утилиты для стратегий совмещения изображений"""

    def __init__(self, debug_mode: bool = False):
        """
        Инициализация утилит с форматтером для отладочного вывода.

        Args:
            config_service: Сервис конфигурации
        """
        self.debug_mode = debug_mode
        self.debug = DebugFormatter(debug_mode, __name__)

    def safe_pearsonr(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Безопасное вычисление корреляции Пирсона для изображений.
        Возвращает 0.0, если один из массивов константный или размеры не совпадают.
        """
        try:
            if a.shape != b.shape:
                return 0.0

            # Явное преобразование к правильным типам
            a_flat = np.asarray(a, dtype=np.float64).flatten()
            b_flat = np.asarray(b, dtype=np.float64).flatten()

            # Проверка на константные массивы и пустые массивы
            if (len(a_flat) == 0 or
                np.all(a_flat == a_flat[0]) or
                    np.all(b_flat == b_flat[0])):
                return 0.0

            # Используем numpy corrcoef который возвращает матрицу корреляции
            corr_matrix = np.corrcoef(a_flat, b_flat)

            if corr_matrix.shape != (2, 2):
                return 0.0

            corr = corr_matrix[0, 1]

            if np.isnan(corr) or np.isinf(corr):
                return 0.0

            return float(corr)

        except Exception as e:
            self.debug.debug(f"Ошибка в safe_pearsonr: {e}")
            return 0.0

    def safe_find_contours(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Универсальное извлечение контуров для всех версий OpenCV.
        """
        result = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Явное преобразование к List[np.ndarray]
        contours = result[0] if len(result) == 2 else result[1]
        return list(contours)

    def extract_contours_and_centroids(
        self,
        image: np.ndarray,
        min_area: int = 10,
        name: str = ""
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Извлечение контуров и центроидов из бинарного изображения."""
        try:
            contours = self.safe_find_contours(image)

            # Фильтрация по площади
            height, width = image.shape
            max_area = height * width * 0.1

            valid_contours = [
                cnt for cnt in contours
                if min_area < cv2.contourArea(cnt) < max_area
            ]

            centroids = self._get_centroids(valid_contours)

            if self.debug_mode:
                self.debug.debug(
                    f"{name}: контуров {len(contours)}->{len(valid_contours)}, центроидов {len(centroids)}"
                )

            return valid_contours, centroids
        except Exception as e:
            self.debug.error(f"extract_contours_and_centroids ошибка: {e}")
            return [], np.array([], dtype=np.float32)

    def _get_centroids(self, contours: List[np.ndarray]) -> np.ndarray:
        """Вычисление центроидов контуров."""
        centers = []
        for contour in contours:
            moments = cv2.moments(contour)
            if moments['m00'] > 0:
                center_x = moments['m10'] / moments['m00']
                center_y = moments['m01'] / moments['m00']
                centers.append([center_x, center_y])

        return np.array(centers, dtype=np.float32) if centers else np.array([], dtype=np.float32)

    def match_and_estimate(
        self,
        scan_centroids: np.ndarray,
        ref_centroids: np.ndarray,
        ransac_params: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], int, float, str]:
        """
        Улучшенное сопоставление точек с детальной диагностикой ошибок.

        Args:
            scan_centroids: Центроиды контуров скана
            ref_centroids: Центроиды контуров эталона
            ransac_params: Параметры RANSAC

        Returns:
            Tuple: (affine_matrix, inliers_count, mean_error, error_message)
        """
        try:
            if self.debug_mode:
                self.debug.section("Сопоставление точек RANSAC", phase="DEBUG")

            # Проверка входных данных
            if len(scan_centroids) < 3 or len(ref_centroids) < 3:
                error_msg = f"Недостаточно точек: scan={len(scan_centroids)}, ref={len(ref_centroids)}"
                self.debug.warn(error_msg)
                return None, 0, float('inf'), error_msg

            # Гарантируем float64 для точности
            scan_centroids = scan_centroids.astype(np.float64)
            ref_centroids = ref_centroids.astype(np.float64)

            # KD-дерево для сопоставления
            tree = cKDTree(ref_centroids)
            distances, indices = tree.query(scan_centroids, k=1)

            # Анализ расстояний
            mean_distance = np.mean(distances)
            if self.debug_mode:
                self.debug.debug(
                    f"Среднее расстояние KDTree: {mean_distance:.2f}")

            if mean_distance > 1000:  # Слишком большие расстояния
                error_msg = f"Точки слишком далеко: mean_distance={mean_distance:.2f}"
                self.debug.warn(error_msg)
                return None, 0, float('inf'), error_msg

            matched_ref = ref_centroids[indices]
            matched_scan = scan_centroids

            # Пробуем разные методы RANSAC
            methods = [
                ('RANSAC', cv2.RANSAC),
                ('LMEDS', cv2.LMEDS)
            ]

            best_matrix = None
            best_inliers = 0
            best_error = float('inf')
            best_method = None

            for method_name, method in methods:
                try:
                    matrix, inliers_mask = cv2.estimateAffinePartial2D(
                        matched_scan, matched_ref,
                        method=method,
                        ransacReprojThreshold=ransac_params.get(
                            'threshold', 3.0),
                        maxIters=ransac_params.get('max_iterations', 2000),
                        confidence=ransac_params.get('confidence', 0.99)
                    )

                    if matrix is None:
                        if self.debug_mode:
                            self.debug.debug(
                                f"{method_name}: невалидная матрица")
                        continue

                    # Подсчет inliers
                    inliers_count = int(np.sum(inliers_mask))

                    # Вычисление ошибки
                    inlier_scan = matched_scan[inliers_mask.ravel() == 1]
                    inlier_ref = matched_ref[inliers_mask.ravel() == 1]

                    if len(inlier_scan) > 0:
                        transformed = cv2.transform(
                            inlier_scan.reshape(-1, 1, 2), matrix
                        ).reshape(-1, 2)
                        errors = np.linalg.norm(
                            transformed - inlier_ref, axis=1)
                        mean_error = float(np.mean(errors))
                    else:
                        mean_error = float('inf')

                    if self.debug_mode:
                        self.debug.debug(
                            f"{method_name}: inliers={inliers_count}, error={mean_error:.2f}"
                        )

                    # Обновляем лучший результат
                    if inliers_count > best_inliers or (
                        inliers_count == best_inliers and mean_error < best_error
                    ):
                        best_matrix = matrix
                        best_inliers = inliers_count
                        best_error = mean_error
                        best_method = method_name

                except Exception as e:
                    if self.debug_mode:
                        self.debug.debug(f"{method_name} failed: {e}")

            if best_matrix is None:
                error_msg = "Все методы RANSAC завершились неудачно"
                self.debug.error(error_msg)
                return None, 0, float('inf'), error_msg

            if self.debug_mode:
                self.debug.success(
                    f"Лучший метод: {best_method}, inliers={best_inliers}, error={best_error:.2f}"
                )

            return best_matrix, best_inliers, best_error, ""

        except Exception as e:
            error_msg = f"Ошибка match_and_estimate: {str(e)}"
            self.debug.error(error_msg)
            return None, 0, float('inf'), error_msg

    def _is_valid_affine_matrix(self, matrix: np.ndarray) -> bool:
        """Тщательная проверка валидности аффинной матрицы."""
        if matrix is None or np.allclose(matrix, 0):
            return False

        # Проверка вращательной части
        rot_matrix = matrix[:2, :2]
        det = np.linalg.det(rot_matrix)

        # 🔧 ИСПРАВЛЕНО: Более гибкая проверка детерминанта
        if abs(det - 1.0) > 0.5:  # Увеличили допуск
            self.debug.debug(f"Невалидный детерминант: {det} (ожидается ~1.0)")
            return False

        # Проверка что матрица не вырождена в точку
        if np.allclose(rot_matrix, 0, atol=1e-6):
            self.debug.debug("Нулевая матрица вращения")
            return False

        # 🔧 ИСПРАВЛЕНО: Более гибкая проверка трансляции
        translation = matrix[:, 2]
        max_translation = 5000  # Увеличили максимальный допустимый сдвиг
        if np.any(np.abs(translation) > max_translation):
            self.debug.debug(
                f"Слишком большая трансляция: {translation} (max: {max_translation})")
            return False

        # 🔧 ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: матрица не должна быть идентичной для всех точек
        test_points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
        transformed = cv2.transform(
            test_points.reshape(-1, 1, 2), matrix).reshape(-1, 2)

        # Все точки трансформированы в одну
        if np.allclose(transformed, transformed[0]):
            self.debug.debug("Матрица преобразует все точки в одну локацию")
            return False

        return True

    def _fallback_translation(
        self,
        scan_centroids: np.ndarray,
        ref_centroids: np.ndarray
    ) -> Tuple[Optional[np.ndarray], int, float, str]:
        """Fallback метод: простая трансляция на основе центров масс."""
        try:
            # Центры масс
            scan_center = np.mean(scan_centroids, axis=0)
            ref_center = np.mean(ref_centroids, axis=0)

            # Вектор трансляции
            translation = ref_center - scan_center

            # Создаем матрицу трансляции
            matrix = np.array([
                [1, 0, translation[0]],
                [0, 1, translation[1]]
            ], dtype=np.float64)

            self.debug.info(f"Используется fallback трансляция: {translation}")
            translation_norm = float(np.linalg.norm(translation))
            return matrix, len(scan_centroids), translation_norm, "Использована fallback трансляция"

        except Exception as e:
            error_msg = f"Fallback не удался: {str(e)}"
            self.debug.error(error_msg)
            return None, 0, float('inf'), error_msg

    def transform_image_simple(
        self,
        image: np.ndarray,
        rotate: int = 0,
        flip: Optional[int] = None
    ) -> np.ndarray:
        """Простое преобразование изображения."""
        result = image.copy()

        # Поворот
        if rotate == 90:
            result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotate == -90:
            result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == 180:
            result = cv2.rotate(result, cv2.ROTATE_180)

        # Отражение (если указано)
        if flip is not None:
            result = cv2.flip(result, flip)

        return result

    def transform_image_matrix(
        self,
        image: np.ndarray,
        rotate: int = 0,
        flip: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Преобразование с возвратом матрицы."""
        height, width = image.shape
        flip_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        if flip is not None:
            if flip == 1:  # горизонтальное отражение
                flip_matrix = np.array(
                    [[-1, 0, width-1], [0, 1, 0]], dtype=np.float32)
            elif flip == 0:  # вертикальное отражение
                flip_matrix = np.array(
                    [[1, 0, 0], [0, -1, height-1]], dtype=np.float32)

        rotate_matrix = cv2.getRotationMatrix2D(
            (width / 2, height / 2), -rotate, 1.0)

        # Объединяем матрицы
        total_matrix = rotate_matrix @ np.vstack([flip_matrix, [0, 0, 1]])
        total_matrix = total_matrix[:2, :]

        result = cv2.warpAffine(image, total_matrix, (width, height))

        return result, total_matrix

    def calculate_alignment_metrics(
        self,
        ref_binary: np.ndarray,
        aligned_image: np.ndarray,
        correlation: float,
        mean_contour_distance: float = 0.0
    ) -> Dict[str, Any]:
        """
        Вычисляет метрики совмещения (IoU, Dice, количество контуров) 
        с проверкой входных данных.
        """
        try:
            # 🔍 ПРОВЕРКА ВХОДНЫХ ДАННЫХ
            if ref_binary is None or aligned_image is None:
                self.debug.error("Пустые входные данные для расчета метрик")
                return {'alignment_metrics': {}}

            if ref_binary.size == 0 or aligned_image.size == 0:
                self.debug.error(
                    "Изображения нулевого размера для расчета метрик")
                return {'alignment_metrics': {}}

            if self.debug_mode:
                self.debug.section("Расчет метрик совмещения", phase="DEBUG")
                self.debug.debug(
                    f"ref_binary: shape={ref_binary.shape}, dtype={ref_binary.dtype}")
                self.debug.debug(
                    f"aligned_image: shape={aligned_image.shape}, dtype={aligned_image.dtype}")

            # 🔧 ПРОВЕРКА И КОРРЕКЦИЯ БИНАРИЗАЦИИ
            # Проверяем ref_binary
            ref_unique = np.unique(ref_binary)
            if (len(ref_unique) > 2 or
                (len(ref_unique) == 2 and
                 not (0 in ref_unique and (1 in ref_unique or 255 in ref_unique)))):
                self.debug.warn(
                    "ref_binary не бинаризован правильно, применяю бинаризацию")
                _, ref_binary = cv2.threshold(
                    ref_binary, 128, 1, cv2.THRESH_BINARY)

            # Бинаризуем aligned_image
            aligned_unique = np.unique(aligned_image)
            if (len(aligned_unique) <= 2 and
                    (0 in aligned_unique and (1 in aligned_unique or 255 in aligned_unique))):
                # Уже бинаризовано, нормализуем к 0-1
                aligned_binary = (aligned_image > 0).astype(np.uint8)
            else:
                if self.debug_mode:
                    self.debug.debug("aligned_image требует бинаризации")
                _, aligned_binary = cv2.threshold(
                    aligned_image, 128, 1, cv2.THRESH_BINARY)

            # 🔍 ПРОВЕРКА ПОСЛЕ БИНАРИЗАЦИИ
            ref_sum = np.sum(ref_binary)
            aligned_sum = np.sum(aligned_binary)

            if self.debug_mode:
                self.debug.debug(
                    f"После бинаризации - ref_sum={ref_sum}, aligned_sum={aligned_sum}")

            if ref_sum == 0:
                self.debug.warn(
                    "ref_binary полностью черный (нет белых пикселей)")
            if aligned_sum == 0:
                self.debug.warn(
                    "aligned_binary полностью черный (нет белых пикселей)")

            # 📊 ВЫЧИСЛЕНИЕ МЕТРИК
            intersection = np.logical_and(ref_binary, aligned_binary)
            union = np.logical_or(ref_binary, aligned_binary)

            intersection_pixels = np.sum(intersection)
            union_pixels = np.sum(union)

            # IoU с защитой от деления на ноль
            iou = intersection_pixels / union_pixels if union_pixels > 0 else 0.0

            # Dice coefficient с защитой от деления на ноль
            total_pixels = ref_sum + aligned_sum
            dice = (2.0 * intersection_pixels) / \
                total_pixels if total_pixels > 0 else 0.0

            # 🔍 КОНТУРЫ
            ref_contours = self.safe_find_contours(ref_binary)
            aligned_contours = self.safe_find_contours(aligned_binary)

            if self.debug_mode:
                metrics_data = {
                    'Пересечение пикселей': intersection_pixels,
                    'Объединение пикселей': union_pixels,
                    'IoU': f"{iou:.6f}",
                    'Коэффициент Dice': f"{dice:.6f}",
                    'Контуры ref': len(ref_contours),
                    'Контуры aligned': len(aligned_contours),
                    'Корреляция': f"{correlation:.6f}",
                    'Ср. расстояние контуров': f"{mean_contour_distance:.6f}"
                }
                self.debug.metrics_table("Результаты метрик", metrics_data)

            return {
                'alignment_metrics': {
                    'correlation': float(correlation),
                    'iou': float(iou),
                    'dice_coefficient': float(dice),
                    'intersection_pixels': int(intersection_pixels),
                    'union_pixels': int(union_pixels),
                    'mean_contour_distance': float(mean_contour_distance),
                    'ref_contours_count': len(ref_contours),
                    'aligned_contours_count': len(aligned_contours),
                    'ref_nonzero_pixels': int(ref_sum),
                    'aligned_nonzero_pixels': int(aligned_sum)
                }
            }

        except Exception as e:
            self.debug.error(f"Ошибка calculate_alignment_metrics: {e}")
            return {'alignment_metrics': {}}
