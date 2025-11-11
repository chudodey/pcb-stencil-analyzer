"""
Стратегия совмещения через трансформацию эталона (подход 2 из AlignmentEngine)
"""

import time
from typing import Dict, Any, Optional
import numpy as np

# pylint: disable=no-member
import cv2

from .base_alignment import AlignmentStrategy
from ..base_strategies import StrategyResult
from .alignment_utils import AlignmentUtils


class ReferenceTransformAlignmentStrategy(AlignmentStrategy):
    """
    Стратегия совмещения через трансформацию эталона.
    Трансформирует эталон и использует RANSAC для совмещения по центроидам контуров.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__("ReferenceTransformAlignment", config or {})
        self.correlation_threshold = self.config.get(
            'correlation_threshold', 0.85)

    def execute(self, input_data: Any, context: Dict[str, Any]) -> StrategyResult:
        """
        Выполняет совмещение через трансформацию эталона и RANSAC.
        """
        start_time = time.time()

        try:
            if self.debug_fmt:
                self.debug_fmt.info(
                    "Начало стратегии трансформации эталона", indent=1)

            # 🔧 ИЗВЛЕЧЕНИЕ ИЗОБРАЖЕНИЯ ИЗ ВХОДНЫХ ДАННЫХ
            reference = self._extract_input_image(input_data, 'reference')
            scan = self._extract_input_image(input_data, "scan")

            if self.debug_fmt:
                self.debug_fmt.debug("Исходные размеры изображений:", indent=2)
                self.debug_fmt.debug(f"Эталон: {reference.shape}", indent=3)
                self.debug_fmt.debug(f"Скан: {scan.shape}", indent=3)

            # ✅ ВЫРАВНИВАЕМ РАЗМЕРЫ напрямую
            ref_binary, scan_binary, _ = self._align_image_sizes(
                reference, scan
            )

            if self.debug_fmt:
                self.debug_fmt.debug("Выровненные размеры:", indent=2)
                self.debug_fmt.debug(f"Эталон: {ref_binary.shape}", indent=3)
                self.debug_fmt.debug(f"Скан: {scan_binary.shape}", indent=3)

            # Извлечение контуров скана
            scan_contours_result = AlignmentUtils.extract_contours_and_centroids(
                scan_binary, self.config.get('min_contour_area', 10), "Скан"
            )
            scan_contours, scan_centroids = scan_contours_result

            if self.debug_fmt:
                self.debug_fmt.debug("Контуры скана:", indent=2)
                self.debug_fmt.debug(
                    f"Контуров: {len(scan_contours)}", indent=3)
                self.debug_fmt.debug(
                    f"Центроидов: {len(scan_centroids)}", indent=3)

            if len(scan_centroids) < 3:
                error_msg = f"Скан содержит недостаточно контуров: {len(scan_centroids)}"
                if self.debug_fmt:
                    self.debug_fmt.error(error_msg, indent=2)
                return StrategyResult(
                    strategy_name=self.name,
                    success=False,
                    result_data=None,
                    metrics={},
                    processing_time=time.time() - start_time,
                    error_message=error_msg
                )

            # Пробуем различные трансформации эталона
            best_result = self._try_reference_transformations(
                ref_binary, scan_binary, scan_centroids, context)

            if best_result is None:
                error_msg = "Не удалось найти подходящую трансформацию эталона"
                if self.debug_fmt:
                    self.debug_fmt.error(error_msg, indent=2)
                return StrategyResult(
                    strategy_name=self.name,
                    success=False,
                    result_data=None,
                    metrics={},
                    processing_time=time.time() - start_time,
                    error_message=error_msg
                )

            # Создаем финальный результат
            aligned_image = best_result['result_image']
            correlation = best_result['correlation']

            if self.debug_fmt:
                self.debug_fmt.success(
                    "Найдена лучшая трансформация:", indent=2)
                self.debug_fmt.debug(
                    f"Ориентация: {best_result['orientation']}", indent=3)
                self.debug_fmt.debug(
                    f"Корреляция: {correlation:.3f}", indent=3)
                self.debug_fmt.debug(
                    f"Инлайнеров: {best_result['inliers']}", indent=3)
                self.debug_fmt.debug(
                    f"Ошибка: {best_result['error']:.3f}", indent=3)

            # Вычисляем метрики
            metrics = AlignmentUtils.calculate_alignment_metrics(
                ref_binary, aligned_image, correlation, best_result['error']
            )

            # Дополнительные метрики из матрицы преобразования
            final_metrics = self._extract_transformation_metrics(
                best_result['final_matrix'], context
            )

            # Безопасное обновление метрик
            if 'alignment_metrics' in metrics:
                metrics['alignment_metrics'].update(final_metrics)

            if self.debug_fmt:
                self.debug_fmt.metrics_table(
                    "Метрики совмещения", metrics.get('alignment_metrics', {}))

            return StrategyResult(
                strategy_name=self.name,
                success=True,
                result_data={
                    'aligned_image': aligned_image,
                    'transform': best_result['final_matrix'],
                    'orientation': best_result['orientation'],
                    'correlation': correlation
                },
                metrics=metrics,
                processing_time=time.time() - start_time
            )

        except Exception as e:
            error_msg = f"Ошибка в стратегии трансформации эталона: {str(e)}"
            if self.debug_fmt:
                self.debug_fmt.error(error_msg, indent=1)
            return StrategyResult(
                strategy_name=self.name,
                success=False,
                result_data=None,
                metrics={},
                processing_time=time.time() - start_time,
                error_message=error_msg
            )

    def _try_reference_transformations(self, ref_binary: np.ndarray, scan_binary: np.ndarray,
                                       scan_centroids: np.ndarray, context: Dict[str, Any]) -> Optional[Dict]:
        """
        Пробует различные трансформации эталона и выбирает лучшую.
        """
        transformations = [
            (0, None, "0°"), (90, None, "90°"), (180,
                                                 None, "180°"), (-90, None, "270°"),
            (0, 1, "0°+flip_h"), (90, 1, "90°+flip_h"), (180,
                                                         1, "180°+flip_h"), (-90, 1, "270°+flip_h"),
        ]

        best_result = {
            'matrix': None, 'orientation': None, 'inliers': -1, 'error': float('inf'),
            'correlation': -1, 'ref_transformed': None, 'ref_transform_matrix': None,
            'final_matrix': None, 'result_image': None
        }

        ransac_params = {
            'ransac_threshold': self.config.get('ransac_threshold', 3.0),
            'max_iterations': self.config.get('max_iterations', 2000),
            'confidence': self.config.get('confidence', 0.99),
            'refine_iterations': self.config.get('refine_iterations', 10)
        }

        debug_mode = context.get('debug_mode', False)

        if self.debug_fmt:
            self.debug_fmt.debug(
                "Тестирование трансформаций эталона:", indent=2)

        for rotate, flip, orientation_name in transformations:
            try:
                if self.debug_fmt:
                    self.debug_fmt.debug(
                        f"Трансформация: {orientation_name}", indent=3)

                # Трансформация эталона с получением матрицы
                ref_transformed, ref_transform_matrix = AlignmentUtils.transform_image_matrix(
                    ref_binary, rotate=rotate, flip=flip
                )

                # Извлечение контуров из трансформированного эталона
                ref_contours_result = AlignmentUtils.extract_contours_and_centroids(
                    ref_transformed, self.config.get('min_contour_area', 10),
                    f"Эталон ({orientation_name})"
                )
                ref_contours, ref_centroids = ref_contours_result

                if len(ref_centroids) < 3:
                    if self.debug_fmt:
                        self.debug_fmt.debug(
                            "Недостаточно контуров, пропускаем", indent=4)
                    continue

                if self.debug_fmt:
                    self.debug_fmt.debug(
                        f"Контуров эталона: {len(ref_contours)}", indent=4)

                # RANSAC: скан → трансформированный эталон
                result = AlignmentUtils.match_and_estimate(
                    scan_centroids, ref_centroids, ransac_params, debug_mode=debug_mode)
                affine_matrix, inliers_count, mean_error, error_msg = result

                if affine_matrix is None:
                    if self.debug_fmt:
                        self.debug_fmt.debug(
                            f"RANSAC не удался: {error_msg}", indent=4)
                    continue

                # Применение к скану для проверки корреляции
                height, width = ref_transformed.shape
                scan_aligned = cv2.warpAffine(
                    scan_binary, affine_matrix, (width, height))

                # Корреляция
                correlation = AlignmentUtils.safe_pearsonr(
                    ref_transformed, scan_aligned)

                # Обновление лучшего результата
                is_better = (
                    inliers_count > best_result['inliers'] or
                    (inliers_count == best_result['inliers'] and mean_error < best_result['error']) or
                    (inliers_count == best_result['inliers'] and mean_error ==
                     best_result['error'] and correlation > best_result['correlation'])
                )

                if is_better:
                    best_result.update({
                        'matrix': affine_matrix, 'orientation': orientation_name, 'inliers': inliers_count,
                        'error': mean_error, 'correlation': correlation, 'ref_transformed': ref_transformed,
                        'ref_transform_matrix': ref_transform_matrix, 'scan_aligned': scan_aligned
                    })

                    if self.debug_fmt:
                        self.debug_fmt.debug(
                            "Обновлен лучший результат:", indent=4)
                        self.debug_fmt.debug(
                            f"Инлайнеров: {inliers_count}", indent=5)
                        self.debug_fmt.debug(
                            f"Ошибка: {mean_error:.3f}", indent=5)
                        self.debug_fmt.debug(
                            f"Корреляция: {correlation:.4f}", indent=5)

            except Exception as e:
                if self.debug_fmt:
                    self.debug_fmt.debug(
                        f"Ошибка трансформации: {str(e)}", indent=4)
                continue

        # Если найдено решение, вычисляем финальную матрицу
        if best_result['matrix'] is not None:
            try:
                if self.debug_fmt:
                    self.debug_fmt.debug(
                        "Вычисление финальной матрицы:", indent=2)

                # Композиция матриц
                affine_matrix = best_result['matrix']
                ref_transform_matrix = best_result['ref_transform_matrix']

                # Вычисляем обратную матрицу для трансформации эталона
                ref_inverse_matrix = cv2.invertAffineTransform(
                    ref_transform_matrix)

                # Комбинируем матрицы
                final_matrix = ref_inverse_matrix @ np.vstack(
                    [affine_matrix, [0, 0, 1]])
                final_matrix = final_matrix[:2, :]

                # Применяем к исходному скану
                height, width = ref_binary.shape
                result_image = cv2.warpAffine(
                    scan_binary, final_matrix, (width, height))

                best_result['result_image'] = result_image
                best_result['final_matrix'] = final_matrix

                if self.debug_fmt:
                    self.debug_fmt.debug(
                        "Финальная матрица вычислена", indent=3)
                    self.debug_fmt.debug(
                        f"Корреляция: {best_result['correlation']:.4f}", indent=4)

            except Exception as e:
                error_msg = f"Ошибка композиции матриц: {str(e)}"
                if self.debug_fmt:
                    self.debug_fmt.error(error_msg, indent=2)
                return None

        if self.debug_fmt:
            if best_result['matrix'] is not None:
                self.debug_fmt.debug(
                    f"Лучшая трансформация: {best_result['orientation']}", indent=2)
            else:
                self.debug_fmt.debug(
                    "Подходящая трансформация не найдена", indent=2)

        return best_result if best_result['matrix'] is not None else None

    def _extract_transformation_metrics(self, transform_matrix: np.ndarray,
                                        context: Dict[str, Any]) -> Dict[str, float]:
        """
        Извлекает метрики из матрицы преобразования.
        """
        try:
            rotation_rad = np.arctan2(
                transform_matrix[1, 0], transform_matrix[0, 0])
            rotation_deg = np.degrees(rotation_rad)

            shift_x_px = transform_matrix[0, 2]
            shift_y_px = transform_matrix[1, 2]

            dpi = context.get('dpi', 600)
            mm_per_pixel = 25.4 / dpi
            shift_x_mm = shift_x_px * mm_per_pixel
            shift_y_mm = shift_y_px * mm_per_pixel

            metrics = {
                'rotation_degrees': rotation_deg,
                'shift_x_px': shift_x_px,
                'shift_y_px': shift_y_px,
                'shift_x_mm': shift_x_mm,
                'shift_y_mm': shift_y_mm
            }

            if self.debug_fmt:
                self.debug_fmt.debug("Метрики трансформации:", indent=2)
                self.debug_fmt.debug(
                    f"Поворот: {rotation_deg:+.2f}°", indent=3)
                self.debug_fmt.debug(
                    f"Сдвиг X: {shift_x_px:+.1f} px", indent=3)
                self.debug_fmt.debug(
                    f"Сдвиг Y: {shift_y_px:+.1f} px", indent=3)
                self.debug_fmt.debug(
                    f"Сдвиг X: {shift_x_mm:+.2f} мм", indent=3)
                self.debug_fmt.debug(
                    f"Сдвиг Y: {shift_y_mm:+.2f} мм", indent=3)

            return metrics

        except Exception as e:
            error_msg = f"Ошибка извлечения метрик трансформации: {str(e)}"
            if self.debug_fmt:
                self.debug_fmt.warn(error_msg, indent=2)
            return {
                'rotation_degrees': 0.0, 'shift_x_px': 0.0, 'shift_y_px': 0.0,
                'shift_x_mm': 0.0, 'shift_y_mm': 0.0
            }
