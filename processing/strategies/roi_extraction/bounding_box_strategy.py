# processing/strategies/roi_extraction/bounding_box_strategy.py
"""
Стратегия выделения ROI на основе bounding box контуров.

СООТВЕТСТВИЕ ИНСТРУКЦИИ:
- Уровень 4: ProcessingStrategy (Исполнитель)  
- Гарантированный composite_score в корне метрик
- Логирование через execute_with_logging из родительского класса
- Детальное описание процесса выделения ROI bounding box
"""

import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .base_roi_extraction import BaseROIExtractionStrategy, StrategyResult


class BoundingBoxROIStrategy(BaseROIExtractionStrategy):
    """Стратегия выделения ROI на основе bounding box контуров."""

    def __init__(self, name: str = "BoundingBoxROI", config: Dict[str, Any] = None):
        super().__init__(name, config or {})

    def _extract_roi(self, image: np.ndarray, contours: List[np.ndarray],
                     context: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """
        Выделение ROI на основе bounding box всех контуров.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Реализация абстрактного метода из BaseROIExtractionStrategy
        - Детальное логирование процесса выделения
        - Возвращает координаты ROI (x1, y1, x2, y2)
        """
        debug_mode = context.get('debug_mode', False)

        if debug_mode:
            self.debug_fmt.debug("Расчет общего bounding box...", indent=2)

        # 1. ВЫЧИСЛЕНИЕ BOUNDING BOX ДЛЯ ВСЕХ КОНТУРОВ
        all_rects = [cv2.boundingRect(c) for c in contours]

        if not all_rects:
            raise ValueError("Не удалось вычислить bounding box для контуров")

        # 2. НАХОЖДЕНИЕ МИНИМАЛЬНЫХ И МАКСИМАЛЬНЫХ КООРДИНАТ
        min_x = min(r[0] for r in all_rects)
        min_y = min(r[1] for r in all_rects)
        max_x = max(r[0] + r[2] for r in all_rects)
        max_y = max(r[1] + r[3] for r in all_rects)

        if debug_mode:
            self.debug_fmt.debug(
                f"Исходный bbox: ({min_x}, {min_y}) - ({max_x}, {max_y})",
                indent=3
            )

        # 3. ПРИМЕНЕНИЕ MARGIN С ПРОВЕРКОЙ ГРАНИЦ
        x_start = max(0, min_x - self.margin_pixels)
        y_start = max(0, min_y - self.margin_pixels)
        x_end = min(image.shape[1], max_x + self.margin_pixels)
        y_end = min(image.shape[0], max_y + self.margin_pixels)

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
        Расчет специфичных метрик для стратегии bounding box.

        Returns:
            Dict[str, Any]: Дополнительные метрики для bounding box стратегии
        """
        debug_mode = context.get('debug_mode', False)
        x1, y1, x2, y2 = roi_coords
        roi_width, roi_height = x2 - x1, y2 - y1

        # 1. РАСЧЕТ ПЛОТНОСТИ КОНТУРОВ В ROI
        contours_in_roi = [
            c for c in contours if self._is_contour_in_roi(c, roi_coords)]
        contour_density = len(contours_in_roi) / (roi_width *
                                                  roi_height) if roi_width * roi_height > 0 else 0

        # 2. РАСЧЕТ РАСПРЕДЕЛЕНИЯ КОНТУРОВ ПО ПЛОЩАДИ ROI
        if contours_in_roi:
            contour_areas = [cv2.contourArea(c) for c in contours_in_roi]
            area_coverage = sum(contour_areas) / (roi_width * roi_height)
        else:
            area_coverage = 0

        # 3. РАСЧЕТ ЭФФЕКТИВНОСТИ BOUNDING BOX
        total_bbox_area = sum(r[2] * r[3]
                              for r in [cv2.boundingRect(c) for c in contours])
        bbox_efficiency = total_bbox_area / \
            (roi_width * roi_height) if roi_width * roi_height > 0 else 0

        strategy_metrics = {
            # Нормализация
            'contour_density': min(1.0, contour_density * 10000),
            'area_coverage': min(1.0, area_coverage),
            'bbox_efficiency': min(1.0, bbox_efficiency),
            'contours_in_roi': len(contours_in_roi),
            'total_contours': len(contours),
            'roi_area': roi_width * roi_height
        }

        if debug_mode:
            self.debug_fmt.debug(
                f"Специфичные метрики: density={strategy_metrics['contour_density']:.3f}, "
                f"coverage={strategy_metrics['area_coverage']:.3f}, "
                f"efficiency={strategy_metrics['bbox_efficiency']:.3f}",
                indent=3
            )

        return strategy_metrics

    def is_applicable(self, input_data: Any, context: Dict[str, Any]) -> bool:
        """
        Проверка применимости bounding box стратегии.

        Returns:
            bool: True если есть контуры для обработки
        """
        try:
            contours = self._get_contours_from_input(input_data)
            return len(contours) > 0
        except:
            return False
