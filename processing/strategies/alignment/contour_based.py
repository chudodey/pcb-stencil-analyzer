"""
Стратегия совмещения на основе контуров (подход 1 из AlignmentEngine)
"""

import time
from typing import Dict, Any, Optional
import numpy as np

# pylint: disable=no-member
import cv2

from .base_alignment import AlignmentStrategy
from ..base_strategies import StrategyResult
from .alignment_utils import AlignmentUtils


class ContourBasedAlignmentStrategy(AlignmentStrategy):
    """
    Стратегия совмещения на основе контуров.
    Трансформирует скан и использует RANSAC для совмещения по центроидам контуров.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        # 🔧 ИСПРАВЛЕНИЕ: передаем пустой dict если config None
        super().__init__("ContourBasedAlignment", config or {})
        self.correlation_threshold = self.config.get(
            'correlation_threshold', 0.85)

    def execute(self, input_data: Any, context: Dict[str, Any]) -> StrategyResult:
        """
        Выполняет совмещение через трансформацию скана и RANSAC.

        Args:
            input_data: Словарь с ключами 'reference' и 'scan' содержащими изображения
            context: Контекст выполнения
        """
        start_time = time.time()

        try:
            debug_mode = context.get('debug_mode', False)
            if self.debug_fmt:
                self.debug_fmt.info(
                    "Начало обработки контурной стратегией", indent=1)

            # 🔧 ИЗВЛЕЧЕНИЕ ИЗОБРАЖЕНИЙ ИЗ ВХОДНЫХ ДАННЫХ
            reference = self._extract_input_image(input_data, 'reference')
            scan = self._extract_input_image(input_data, 'scan')

            if self.debug_fmt:
                self.debug_fmt.debug("Извлечены изображения:", indent=2)
                self.debug_fmt.debug(f"Эталон: {reference.shape}", indent=3)
                self.debug_fmt.debug(f"Скан: {scan.shape}", indent=3)

            # ✅ ВЫРАВНИВАЕМ РАЗМЕРЫ напрямую
            ref_binary, scan_binary, was_rotated = self._align_image_sizes(
                reference, scan
            )

            if self.debug_fmt:
                self.debug_fmt.debug(
                    "Выровнены размеры изображений:", indent=2)
                self.debug_fmt.debug(
                    f"Эталон бинарный: {ref_binary.shape}", indent=3)
                self.debug_fmt.debug(
                    f"Скан бинарный: {scan_binary.shape}", indent=3)
                if was_rotated:
                    self.debug_fmt.debug("Скан был повернут на 90°", indent=3)

            # Извлечение контуров эталона
            ref_contours, ref_centroids = self.alignment_utils.extract_contours_and_centroids(
                ref_binary, self.config.get('min_contour_area', 10), "Эталон")

            if self.debug_fmt:
                self.debug_fmt.debug(
                    f"Найдено контуров эталона: {len(ref_contours)}", indent=2)
                self.debug_fmt.debug(
                    f"Центроидов: {len(ref_centroids)}", indent=3)

            if len(ref_centroids) < 3:
                error_msg = f"Эталон содержит недостаточно контуров: {len(ref_centroids)}"
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

            # Пробуем различные трансформации скана
            best_result = self._try_transformations(
                scan_binary, ref_binary, ref_centroids, context)

            if best_result is None:
                error_msg = "Не удалось найти подходящую трансформацию"
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
                    f"Найдена лучшая трансформация: {best_result['orientation']}", indent=2)
                self.debug_fmt.debug(
                    f"Корреляция: {correlation:.3f}", indent=3)
                self.debug_fmt.debug(
                    f"Инлайнеров: {best_result['inliers']}", indent=3)
                self.debug_fmt.debug(
                    f"Ошибка: {best_result['error']:.3f}", indent=3)

            # Вычисляем метрики
            metrics = self.alignment_utils.calculate_alignment_metrics(
                ref_binary, aligned_image, correlation, best_result['error']
            )

            # Дополнительные метрики из матрицы преобразования
            final_metrics = self._extract_transformation_metrics(
                best_result['matrix'], context
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
                    'transform': best_result['matrix'],
                    'orientation': best_result['orientation'],
                    'correlation': correlation
                },
                metrics=metrics,
                processing_time=time.time() - start_time
            )

        except Exception as e:
            error_msg = f"Ошибка в контурной стратегии: {str(e)}"
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

    def _try_transformations(self, scan_binary: np.ndarray, ref_binary: np.ndarray,
                             ref_centroids: np.ndarray, context: Dict[str, Any]) -> Optional[Dict]:
        """
        Пробует различные трансформации скана и выбирает лучшую.
        """
        if len(ref_centroids) < 3:
            return None

        transformations = [
            (0, None, "0°"), (90, None, "90°"), (180,
                                                 None, "180°"), (-90, None, "270°"),
            (0, 1, "0°+flip_h"), (90, 1, "90°+flip_h"), (180,
                                                         1, "180°+flip_h"), (-90, 1, "270°+flip_h"),
        ]

        best_result = {
            'matrix': None, 'orientation': None, 'inliers': -1, 'error': float('inf'),
            'correlation': -1, 'result_image': None
        }

        ransac_params = {
            'ransac_threshold': self.config.get('ransac_threshold', 3.0),
            'max_iterations': self.config.get('max_iterations', 2000),
            'confidence': self.config.get('confidence', 0.99),
            'refine_iterations': self.config.get('refine_iterations', 10)
        }

        debug_mode = context.get('debug_mode', False)

        if self.debug_fmt:
            self.debug_fmt.debug("Тестирование трансформаций:", indent=2)

        for rotate, flip, orientation_name in transformations:
            try:
                if self.debug_fmt:
                    self.debug_fmt.debug(
                        f"Трансформация: {orientation_name}", indent=3)

                # Трансформация скана
                scan_transformed = AlignmentUtils.transform_image_simple(
                    scan_binary, rotate=rotate, flip=flip
                )

                # Извлечение контуров из трансформированного скана
                scan_contours, scan_centroids = AlignmentUtils.extract_contours_and_centroids(
                    scan_transformed, self.config.get('min_contour_area', 10),
                    f"Скан ({orientation_name})"
                )

                if len(scan_centroids) < 3:
                    if self.debug_fmt:
                        self.debug_fmt.debug(
                            "Недостаточно контуров, пропускаем", indent=4)
                    continue

                if self.debug_fmt:
                    self.debug_fmt.debug(
                        f"Контуров в скане: {len(scan_contours)}", indent=4)

                # RANSAC
                result = AlignmentUtils.match_and_estimate(
                    scan_centroids, ref_centroids, ransac_params)
                affine_matrix, inliers_count, mean_error, error_msg = result

                if affine_matrix is None:
                    if self.debug_fmt:
                        self.debug_fmt.debug(
                            f"RANSAC не удался: {error_msg}", indent=4)
                    continue

                # Финальное выравнивание
                height, width = ref_binary.shape
                aligned = cv2.warpAffine(
                    scan_transformed, affine_matrix, (width, height))

                # Корреляция
                correlation = AlignmentUtils.safe_pearsonr(ref_binary, aligned)

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
                        'error': mean_error, 'correlation': correlation, 'result_image': aligned
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
