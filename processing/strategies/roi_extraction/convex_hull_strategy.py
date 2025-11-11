# processing/strategies/roi_extraction/convex_hull_strategy.py
"""
Стратегия выделения ROI на основе выпуклой оболочки контуров.

СООТВЕТСТВИЕ ИНСТРУКЦИИ:
- Уровень 4: ProcessingStrategy (Исполнитель)
- Гарантированный composite_score в корне метрик
- Логирование через execute_with_logging из родительского класса
- Детальное описание процесса выделения ROI convex hull
"""

import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .base_roi_extraction import BaseROIExtractionStrategy, StrategyResult


class ConvexHullROIStrategy(BaseROIExtractionStrategy):
    """Стратегия выделения ROI на основе выпуклой оболочки контуров."""

    def __init__(self, name: str = "ConvexHullROI", config: Dict[str, Any] = None):
        super().__init__(name, config or {})

    def _extract_roi(self, image: np.ndarray, contours: List[np.ndarray],
                     context: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """
        Выделение ROI на основе выпуклой оболочки всех контуров.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Реализация абстрактного метода из BaseROIExtractionStrategy
        - Детальное логирование процесса convex hull выделения
        - Возвращает координаты ROI (x1, y1, x2, y2)
        """
        debug_mode = context.get('debug_mode', False)

        if debug_mode:
            self.debug_fmt.debug("Расчет выпуклой оболочки...", indent=2)

        # 1. ОБЪЕДИНЕНИЕ ВСЕХ ТОЧЕК КОНТУРОВ
        if not contours:
            raise ValueError("Нет контуров для построения выпуклой оболочки")

        all_points = np.vstack([contour.reshape(-1, 2)
                               for contour in contours])

        if debug_mode:
            self.debug_fmt.debug(
                f"Объединено точек: {len(all_points)}",
                indent=3
            )

        # 2. ПОСТРОЕНИЕ ВЫПУКЛОЙ ОБОЛОЧКИ
        hull = cv2.convexHull(all_points)
        x, y, w, h = cv2.boundingRect(hull)

        if debug_mode:
            self.debug_fmt.debug(
                f"Выпуклая оболочка: ({x}, {y}, {w}, {h})",
                indent=3
            )
            self.debug_fmt.debug(
                f"Точек в оболочке: {len(hull)}",
                indent=3
            )

        # 3. ПРИМЕНЕНИЕ MARGIN С ПРОВЕРКОЙ ГРАНИЦ
        x_start, y_start, x_end, y_end = self._apply_margin(
            (x, y, w, h), image.shape)

        if debug_mode:
            self.debug_fmt.debug(
                f"ROI с margin {self.margin_pixels}px: "
                f"({x_start}, {y_start}) - ({x_end}, {y_end})",
                indent=3
            )
            self.debug_fmt.debug(
                f"Размер ROI: {x_end - x_start} x {y_end - y_start}",
                indent=3
            )

        # 4. ПРОВЕРКА ВАЛИДНОСТИ ROI
        if x_end <= x_start or y_end <= y_start:
            raise ValueError(
                f"Некорректные координаты ROI: ({x_start}, {y_start}, {x_end}, {y_end})")

        roi_width = x_end - x_start
        roi_height = y_end - y_start

        if debug_mode:
            self.debug_fmt.debug(
                f"Валидация: width={roi_width}, height={roi_height}",
                indent=3
            )

        # 5. ПРОВЕРКА МИНИМАЛЬНОГО РАЗМЕРА
        min_roi_size = context.get('min_roi_size', 100)
        if roi_width < min_roi_size or roi_height < min_roi_size:
            if debug_mode:
                self.debug_fmt.warn(
                    f"ROI слишком мал: {roi_width}x{roi_height} "
                    f"(min: {min_roi_size}x{min_roi_size})",
                    indent=3
                )

        return (x_start, y_start, x_end, y_end)

    def _calculate_strategy_specific_metrics(self,
                                             roi_coords: Tuple[int, int, int, int],
                                             contours: List[np.ndarray],
                                             context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Расчет специфичных метрик для стратегии convex hull.

        Returns:
            Dict[str, Any]: Дополнительные метрики для convex hull стратегии
        """
        debug_mode = context.get('debug_mode', False)
        x1, y1, x2, y2 = roi_coords
        roi_width, roi_height = x2 - x1, y2 - y1

        # 1. РАСЧЕТ ЭФФЕКТИВНОСТИ ВЫПУКЛОЙ ОБОЛОЧКИ
        all_points = np.vstack([contour.reshape(-1, 2)
                               for contour in contours])
        hull = cv2.convexHull(all_points)
        hull_area = cv2.contourArea(hull)
        roi_area = roi_width * roi_height

        hull_efficiency = hull_area / roi_area if roi_area > 0 else 0

        # 2. РАСЧЕТ КОМПАКТНОСТИ ОБОЛОЧКИ
        hull_perimeter = cv2.arcLength(hull, True)
        hull_compactness = (4 * np.pi * hull_area) / \
            (hull_perimeter ** 2) if hull_perimeter > 0 else 0

        # 3. РАСЧЕТ РАСПРЕДЕЛЕНИЯ КОНТУРОВ ВНУТРИ ОБОЛОЧКИ
        contours_in_roi = [
            c for c in contours if self._is_contour_in_roi(c, roi_coords)]

        if contours_in_roi:
            contour_areas = [cv2.contourArea(c) for c in contours_in_roi]
            area_coverage = sum(contour_areas) / roi_area
        else:
            area_coverage = 0

        strategy_metrics = {
            'hull_efficiency': min(1.0, hull_efficiency),
            'hull_compactness': min(1.0, hull_compactness),
            'area_coverage': min(1.0, area_coverage),
            'hull_area': hull_area,
            'hull_perimeter': hull_perimeter,
            'contours_in_roi': len(contours_in_roi),
            'total_contours': len(contours),
            'roi_area': roi_area,
            'hull_points_count': len(hull)
        }

        if debug_mode:
            self.debug_fmt.debug(
                f"Специфичные метрики convex hull: "
                f"efficiency={strategy_metrics['hull_efficiency']:.3f}, "
                f"compactness={strategy_metrics['hull_compactness']:.3f}, "
                f"coverage={strategy_metrics['area_coverage']:.3f}",
                indent=3
            )

        return strategy_metrics

    def _calculate_roi_metrics(self, roi_coords: Tuple[int, ...],
                               original_shape: Tuple[int, ...],
                               contours: List[np.ndarray],
                               context: Dict[str, Any]) -> Dict[str, float]:
        """
        Переопределение расчета метрик с учетом специфики convex hull.

        Returns:
            Dict[str, float]: Метрики с гарантированным composite_score
        """
        # 1. БАЗОВЫЕ МЕТРИКИ ИЗ РОДИТЕЛЬСКОГО КЛАССА
        base_metrics = super()._calculate_roi_metrics(
            roi_coords, original_shape, contours, context
        )

        # 2. СПЕЦИФИЧНЫЕ МЕТРИКИ CONVEX HULL
        strategy_metrics = self._calculate_strategy_specific_metrics(
            roi_coords, contours, context
        )

        # 3. КОРРЕКТИРОВКА COMPOSITE_SCORE С УЧЕТОМ ЭФФЕКТИВНОСТИ HULL
        hull_efficiency = strategy_metrics.get('hull_efficiency', 0.5)
        hull_compactness = strategy_metrics.get('hull_compactness', 0.5)

        # Улучшаем composite_score если выпуклая оболочка эффективна
        efficiency_boost = 0.1 * (hull_efficiency + hull_compactness)
        adjusted_composite = min(
            1.0, base_metrics['composite_score'] + efficiency_boost)

        # 4. ОБЪЕДИНЕНИЕ МЕТРИК
        return {
            **base_metrics,
            'composite_score': adjusted_composite,  # ← СКОРРЕКТИРОВАННЫЙ
            **strategy_metrics
        }

    def is_applicable(self, input_data: Any, context: Dict[str, Any]) -> bool:
        """
        Проверка применимости convex hull стратегии.

        Returns:
            bool: True если есть достаточно контуров для построения оболочки
        """
        try:
            contours = self._get_contours_from_input(input_data)
            # Convex hull требует минимум 3 точки
            if len(contours) < 1:
                return False

            all_points = np.vstack([contour.reshape(-1, 2)
                                   for contour in contours])
            return len(all_points) >= 3
        except:
            return False
