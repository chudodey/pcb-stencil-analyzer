"""
Стратегия совмещения с учетом масштаба и поворота на основе эллипсов и фазовой корреляции.
"""

import time
from typing import Dict, Any, Tuple, List, Optional

# pylint: disable=no-member
import cv2
import numpy as np

from .base_alignment import AlignmentStrategy
from ..base_strategies import StrategyResult


class MomentScaleAlignmentStrategy(AlignmentStrategy):
    """Стратегия совмещения с учетом масштаба и поворота на основе эллипсов и фазовой корреляции."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__("MomentScaleAlignment", config or {})
        self.min_contours_for_ellipse = self.config.get(
            'min_contours_for_ellipse', 5)
        self.correlation_threshold = self.config.get(
            'correlation_threshold', 0.1)

    def execute(self, input_data: Any, context: Dict[str, Any]) -> StrategyResult:
        """
        Выполняет совмещение на основе эллипсов центроидов и фазовой корреляции.
        """
        start_time = time.time()

        try:
            debug_mode = context.get('debug_mode', False)
            if self.debug_fmt:
                self.debug_fmt.info(
                    "Начало стратегии совмещения с учетом масштаба и поворота", indent=1)

            # 🔧 ИЗВЛЕЧЕНИЕ ИЗОБРАЖЕНИЯ ИЗ ВХОДНЫХ ДАННЫХ
            reference = self._extract_input_image(input_data, 'reference')
            scan = self._extract_input_image(input_data, "scan")

            if self.debug_fmt:
                self.debug_fmt.debug("Исходные размеры изображений:", indent=2)
                self.debug_fmt.debug(f"Эталон: {reference.shape}", indent=3)
                self.debug_fmt.debug(f"Скан: {scan.shape}", indent=3)

            # ✅ ВЫРАВНИВАЕМ РАЗМЕРЫ
            ref_aligned, scan_aligned, was_rotated = self._align_image_sizes(
                reference, scan
            )

            if self.debug_fmt:
                self.debug_fmt.debug("Выровненные размеры:", indent=2)
                self.debug_fmt.debug(f"Эталон: {ref_aligned.shape}", indent=3)
                self.debug_fmt.debug(f"Скан: {scan_aligned.shape}", indent=3)
                if was_rotated:
                    self.debug_fmt.debug("Скан был повернут на 90°", indent=3)

            # 🔧 ПОДГОТОВКА БИНАРНЫХ ИЗОБРАЖЕНИЙ
            ref_binary = self._prepare_binary_image(ref_aligned)
            scan_binary = self._prepare_binary_image(scan_aligned)

            # 🔍 ПОИСК КОНТУРОВ
            ref_contours = cv2.findContours(
                ref_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            scan_contours = cv2.findContours(
                scan_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

            # 🔧 ПРЕОБРАЗОВАНИЕ КОНТУРОВ В СПИСОК numpy.ndarray
            ref_contours_list = list(ref_contours)
            scan_contours_list = list(scan_contours)

            if self.debug_fmt:
                self.debug_fmt.debug("Найдено контуров:", indent=2)
                self.debug_fmt.debug(
                    f"Эталон: {len(ref_contours_list)}", indent=3)
                self.debug_fmt.debug(
                    f"Скан: {len(scan_contours_list)}", indent=3)

            # 🚨 ПРОВЕРКА ДОСТАТОЧНОСТИ КОНТУРОВ
            if len(ref_contours_list) < self.min_contours_for_ellipse or len(scan_contours_list) < self.min_contours_for_ellipse:
                error_msg = (
                    f"Недостаточно контуров для вычисления эллипса "
                    f"(требуется > {self.min_contours_for_ellipse}, "
                    f"получено ref={len(ref_contours_list)}, scan={len(scan_contours_list)})"
                )
                if self.debug_fmt:
                    self.debug_fmt.warn(error_msg, indent=2)
                return StrategyResult(
                    strategy_name=self.name,
                    success=False,
                    result_data=None,
                    metrics={},
                    processing_time=time.time() - start_time,
                    error_message=error_msg
                )

            # 🔧 ВЫЧИСЛЕНИЕ ТРАНСФОРМАЦИИ
            transform, angle_diff, scale_factor, shift_dx, shift_dy, correlation = self._compute_ellipse_transform(
                ref_contours_list, scan_contours_list, ref_binary, scan_binary, debug_mode
            )

            if self.debug_fmt:
                self.debug_fmt.success("Вычислена трансформация:", indent=2)
                self.debug_fmt.debug(
                    f"Разница углов: {angle_diff:.2f}°", indent=3)
                self.debug_fmt.debug(
                    f"Коэффициент масштаба: {scale_factor:.4f}", indent=3)
                self.debug_fmt.debug(f"Сдвиг X: {shift_dx:.2f}", indent=3)
                self.debug_fmt.debug(f"Сдвиг Y: {shift_dy:.2f}", indent=3)
                self.debug_fmt.debug(
                    f"Корреляция: {correlation:.6f}", indent=3)

            # 🔧 ПРИМЕНЕНИЕ ТРАНСФОРМАЦИИ
            aligned_image = cv2.warpAffine(
                scan_aligned,  # ✅ ИСПОЛЬЗУЕМ scan_aligned вместо scan
                transform,
                (ref_aligned.shape[1], ref_aligned.shape[0])
            )

            if self.debug_fmt:
                self.debug_fmt.debug("После трансформации:", indent=2)
                self.debug_fmt.debug(
                    f"Размер выровненного изображения: {aligned_image.shape}", indent=3)

            # 📊 ВЫЧИСЛЕНИЕ МЕТРИК
            metrics = self._calculate_metrics(
                ref_aligned, aligned_image, correlation)
            metrics['alignment_metrics'].update({
                'angle_difference': float(angle_diff),
                'calculated_scale': float(scale_factor),
                'shift_dx': float(shift_dx),
                'shift_dy': float(shift_dy)
            })

            if self.debug_fmt:
                self.debug_fmt.metrics_table(
                    "Метрики совмещения", metrics.get('alignment_metrics', {}))

            return StrategyResult(
                strategy_name=self.name,
                success=True,
                result_data={
                    'aligned_image': aligned_image,
                    'transform': transform,
                    'angle_difference': angle_diff,
                    'scale_factor': scale_factor,
                    'shift': (shift_dx, shift_dy)
                },
                metrics=metrics,
                processing_time=time.time() - start_time
            )

        except Exception as e:
            error_msg = f"Ошибка в стратегии совмещения с учетом масштаба: {str(e)}"
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

    def _prepare_binary_image(self, image: np.ndarray) -> np.ndarray:
        """Подготавливает бинарное изображение для анализа."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Адаптивная бинаризация
        if np.unique(image).size <= 2:
            binary = (image > 0).astype(np.uint8)
        else:
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            binary = (binary > 0).astype(np.uint8)

        return binary

    def _get_centroids(self, contours: List[np.ndarray]) -> np.ndarray:
        """
        Вычисляет центроиды контуров.

        Args:
            contours: Список контуров (np.ndarray).

        Returns:
            Массив центроидов (x, y).
        """
        centroids = []
        for contour in contours:
            if len(contour) < 3:
                continue
            moments = cv2.moments(contour)
            if moments['m00'] > 0:
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
                centroids.append([cx, cy])
        return np.array(centroids, dtype=np.float32) if centroids else np.array([])

    def _compute_ellipse_transform(
        self,
        ref_contours: List[np.ndarray],
        scan_contours: List[np.ndarray],
        ref_binary: np.ndarray,
        scan_binary: np.ndarray,
        debug_mode: bool
    ) -> Tuple[np.ndarray, float, float, float, float, float]:
        """
        Вычисляет трансформацию на основе эллипсов и фазовой корреляции.
        """
        ref_points = self._get_centroids(ref_contours)
        scan_points = self._get_centroids(scan_contours)

        if self.debug_fmt:
            self.debug_fmt.debug("Центроиды контуров:", indent=2)
            self.debug_fmt.debug(f"Эталон: {len(ref_points)} точек", indent=3)
            self.debug_fmt.debug(f"Скан: {len(scan_points)} точек", indent=3)

        if len(ref_points) < self.min_contours_for_ellipse or len(scan_points) < self.min_contours_for_ellipse:
            raise ValueError(
                f"Недостаточно точек для эллипса: ref={len(ref_points)}, "
                f"scan={len(scan_points)}"
            )

        # 🔧 ВЫЧИСЛЕНИЕ ЭЛЛИПСОВ
        ellipse_ref = cv2.fitEllipse(ref_points)
        ellipse_scan = cv2.fitEllipse(scan_points)

        angle_ref = ellipse_ref[2]
        angle_scan = ellipse_scan[2]
        angle_diff = angle_ref - angle_scan

        size_ref = max(ellipse_ref[1])
        size_scan = max(ellipse_scan[1])
        scale_factor = size_ref / size_scan if size_scan > 0 else 1.0

        if self.debug_fmt:
            self.debug_fmt.debug("Анализ эллипсов:", indent=2)
            self.debug_fmt.debug(f"Угол эталона: {angle_ref:.2f}°", indent=3)
            self.debug_fmt.debug(f"Угол скана: {angle_scan:.2f}°", indent=3)
            self.debug_fmt.debug(f"Разница углов: {angle_diff:.2f}°", indent=3)
            self.debug_fmt.debug(f"Размер эталона: {size_ref:.2f}", indent=3)
            self.debug_fmt.debug(f"Размер скана: {size_scan:.2f}", indent=3)
            self.debug_fmt.debug(
                f"Коэффициент масштаба: {scale_factor:.4f}", indent=3)

        # 🔧 ПОВОРОТ И МАСШТАБИРОВАНИЕ
        h_scan, w_scan = scan_binary.shape
        rotation_center = (w_scan / 2, h_scan / 2)
        rotation_matrix = cv2.getRotationMatrix2D(
            rotation_center, angle_diff, scale_factor
        )

        height, width = ref_binary.shape
        scan_rotated = cv2.warpAffine(
            scan_binary, rotation_matrix, (width, height)
        )

        # 🔧 ФАЗОВАЯ КОРРЕЛЯЦИЯ ДЛЯ СДВИГА
        shift, correlation_phase = cv2.phaseCorrelate(
            ref_binary.astype(np.float32), scan_rotated.astype(np.float32)
        )
        dx, dy = shift

        if self.debug_fmt:
            self.debug_fmt.debug("Фазовая корреляция:", indent=2)
            self.debug_fmt.debug(f"Сдвиг X: {dx:.2f}", indent=3)
            self.debug_fmt.debug(f"Сдвиг Y: {dy:.2f}", indent=3)
            self.debug_fmt.debug(
                f"Корреляция фазы: {correlation_phase:.6f}", indent=3)

        # 🔧 ПРАВИЛЬНОЕ КОМБИНИРОВАНИЕ ТРАНСФОРМАЦИЙ С getAffineTransform
        # Создаем матрицу трансляции
        translation_matrix = np.array([
            [1.0, 0.0, -dx],
            [0.0, 1.0, -dy]
        ], dtype=np.float32)

        # 🔧 СПОСОБ 1: Прямое комбинирование матриц (правильный способ)
        # Преобразуем в однородные координаты для умножения матриц
        rotation_homogeneous = np.vstack([rotation_matrix, [0.0, 0.0, 1.0]])
        translation_homogeneous = np.vstack(
            [translation_matrix, [0.0, 0.0, 1.0]])

        # Умножаем матрицы: сначала поворот+масштаб, потом трансляция
        final_homogeneous = translation_homogeneous @ rotation_homogeneous

        # Возвращаем к аффинному формату (убираем последнюю строку)
        final_transform = final_homogeneous[:2, :]

        if self.debug_fmt:
            self.debug_fmt.debug("Матрицы трансформации:", indent=2)
            self.debug_fmt.debug("Матрица поворота:", indent=3)
            self.debug_fmt.debug(f"{rotation_matrix}", indent=4)
            self.debug_fmt.debug("Матрица трансляции:", indent=3)
            self.debug_fmt.debug(f"{translation_matrix}", indent=4)
            self.debug_fmt.debug("Финальная матрица трансформации:", indent=3)
            self.debug_fmt.debug(f"{final_transform}", indent=4)

        # 🔧 ВЫЧИСЛЕНИЕ ФИНАЛЬНОЙ КОРРЕЛЯЦИИ
        aligned_final = cv2.warpAffine(
            scan_binary, final_transform, (width, height)
        )

        # Корреляция между эталоном и выровненным изображением
        if ref_binary.size > 0 and aligned_final.size > 0:
            # 🔧 ИСПРАВЛЕНИЕ: Правильное использование np.corrcoef
            ref_flat = ref_binary.flatten().astype(np.float64)
            aligned_flat = aligned_final.flatten().astype(np.float64)

            # Проверяем, что массивы не пустые и имеют одинаковую длину
            if len(ref_flat) == len(aligned_flat) and len(ref_flat) > 1:
                correlation_matrix = np.corrcoef(ref_flat, aligned_flat)
                correlation = correlation_matrix[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0
        else:
            correlation = 0.0

        if self.debug_fmt:
            self.debug_fmt.debug("Финальная трансформация:", indent=2)
            self.debug_fmt.debug(f"Разница углов: {angle_diff:.2f}°", indent=3)
            self.debug_fmt.debug(
                f"Коэффициент масштаба: {scale_factor:.4f}", indent=3)
            self.debug_fmt.debug(f"Сдвиг X: {dx:.2f}", indent=3)
            self.debug_fmt.debug(f"Сдвиг Y: {dy:.2f}", indent=3)
            self.debug_fmt.debug(f"Корреляция: {correlation:.6f}", indent=3)

        return final_transform, angle_diff, scale_factor, dx, dy, correlation
