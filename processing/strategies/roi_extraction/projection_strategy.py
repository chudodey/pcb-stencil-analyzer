# processing/strategies/roi_extraction/projection_strategy.py
"""
Стратегия выделения ROI на основе сравнения проекций (улучшенная версия из ноутбука).

СООТВЕТСТВИЕ ИНСТРУКЦИИ:
- Уровень 4: ProcessingStrategy (Исполнитель)
- Гарантированный composite_score в корне метрик
- Логирование через execute_with_logging из родительского класса
- Детальное описание процесса выделения ROI проекционным методом
"""

import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .base_roi_extraction import BaseROIExtractionStrategy, StrategyResult


class ProjectionROIStrategy(BaseROIExtractionStrategy):
    """Стратегия выделения ROI на основе сравнения проекций."""

    def __init__(self, name: str = "ProjectionROI", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        self.correlation_threshold = self.config.get(
            'correlation_threshold', 0.7)
        self.crop_padding = self.config.get('crop_padding', 50)

    def _extract_roi(self, image: np.ndarray, contours: List[np.ndarray],
                     context: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """
        Выделение ROI на основе сравнения проекций эталона и скана.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Реализация абстрактного метода из BaseROIExtractionStrategy
        - Детальное логирование процесса проекционного выделения
        - Возвращает координаты ROI (x1, y1, x2, y2)
        """
        debug_mode = context.get('debug_mode', False)

        if debug_mode:
            self.debug_fmt.debug("Начало проекционного анализа...", indent=2)

        # 1. ПОДГОТОВКА ЭТАЛОННОГО ИЗОБРАЖЕНИЯ
        ref_gray = context.get('reference_image')
        if ref_gray is None:
            raise ValueError(
                "Не предоставлено эталонное изображение для проекционного сравнения")

        if debug_mode:
            self.debug_fmt.debug(
                f"Эталон: {ref_gray.shape}, Скан: {image.shape}",
                indent=3
            )

        # 2. ВЫПОЛНЕНИЕ ПРОЕКЦИОННОГО АНАЛИЗА
        roi_coords, projection_metrics = self._find_roi_by_projection_comparison(
            image, ref_gray, debug_mode
        )

        # 3. ПРОВЕРКА КАЧЕСТВА КОРРЕЛЯЦИИ
        mean_correlation = projection_metrics.get('mean_correlation', 0)
        if mean_correlation < self.correlation_threshold:
            if debug_mode:
                self.debug_fmt.warn(
                    f"Корреляция ниже порога: {mean_correlation:.3f} < {self.correlation_threshold}",
                    indent=3
                )
            raise ValueError(
                f"Корреляция проекций ниже порога: {mean_correlation:.3f}")

        if debug_mode:
            self.debug_fmt.debug(
                f"Корреляция приемлема: {mean_correlation:.3f} ≥ {self.correlation_threshold}",
                indent=3
            )

        # 4. ПРИМЕНЕНИЕ PADDING К ROI
        x1, y1, x2, y2 = self._apply_padding(roi_coords, image.shape)

        if debug_mode:
            self.debug_fmt.debug(
                f"ROI с padding {self.crop_padding}px: "
                f"({x1}, {y1}) - ({x2}, {y2})",
                indent=3
            )
            self.debug_fmt.debug(
                f"Размер ROI: {x2 - x1} x {y2 - y1}",
                indent=3
            )

        return (x1, y1, x2, y2)

    def _find_roi_by_projection_comparison(self, scan_binary: np.ndarray,
                                           ref_gray: np.ndarray, debug: bool = False) -> Tuple[Tuple[int, int, int, int], Dict[str, Any]]:
        """
        Находит ROI путем сравнения проекций эталона и скана.
        Возвращает координаты ROI и метрики качества проекций.
        """
        # 1. ПОДГОТОВКА ЭТАЛОННЫХ ПРОЕКЦИЙ
        _, ref_binary = cv2.threshold(ref_gray, 128, 255, cv2.THRESH_BINARY)
        ref_h_proj = np.sum(ref_binary, axis=1)  # Вертикальная проекция (по Y)
        # Горизонтальная проекция (по X)
        ref_v_proj = np.sum(ref_binary, axis=0)

        # 2. ПОДГОТОВКА СКАН-ПРОЕКЦИЙ
        scan_h_proj = np.sum(scan_binary, axis=1)
        scan_v_proj = np.sum(scan_binary, axis=0)

        if debug:
            self.debug_fmt.debug(
                f"Проекции эталона: вертикальная={len(ref_h_proj)}, горизонтальная={len(ref_v_proj)}",
                indent=3
            )
            self.debug_fmt.debug(
                f"Проекции скана: вертикальная={len(scan_h_proj)}, горизонтальная={len(scan_v_proj)}",
                indent=3
            )

        # 3. НОРМАЛИЗАЦИЯ ДЛЯ СРАВНЕНИЯ
        ref_h_norm = ref_h_proj / \
            np.max(ref_h_proj) if np.max(ref_h_proj) > 0 else ref_h_proj
        ref_v_norm = ref_v_proj / \
            np.max(ref_v_proj) if np.max(ref_v_proj) > 0 else ref_v_proj
        scan_h_norm = scan_h_proj / \
            np.max(scan_h_proj) if np.max(scan_h_proj) > 0 else scan_h_proj
        scan_v_norm = scan_v_proj / \
            np.max(scan_v_proj) if np.max(scan_v_proj) > 0 else scan_v_proj

        # 4. ПОИСК ОПТИМАЛЬНОГО СМЕЩЕНИЯ ЧЕРЕЗ КОРРЕЛЯЦИЮ
        def find_best_shift(ref_proj, scan_proj, axis_name):
            """Находит лучшее смещение между проекциями."""
            if len(scan_proj) < len(ref_proj):
                padding = len(ref_proj) - len(scan_proj)
                scan_proj = np.pad(scan_proj, (0, padding), 'constant')

            correlation = np.correlate(scan_proj, ref_proj, mode='valid')
            best_shift = int(np.argmax(correlation))
            max_correlation = float(np.max(correlation))

            if debug:
                self.debug_fmt.debug(
                    f"Корреляция по {axis_name}: смещение={best_shift}, корреляция={max_correlation:.3f}",
                    indent=4
                )

            return best_shift, max_correlation

        y_shift, y_correlation = find_best_shift(ref_h_norm, scan_h_norm, 'Y')
        x_shift, x_correlation = find_best_shift(ref_v_norm, scan_v_norm, 'X')

        # 5. ОПРЕДЕЛЕНИЕ ROI
        y1 = max(0, y_shift)
        y2 = min(scan_binary.shape[0], y_shift + len(ref_h_proj))
        x1 = max(0, x_shift)
        x2 = min(scan_binary.shape[1], x_shift + len(ref_v_proj))

        # 6. РАСЧЕТ МЕТРИК КАЧЕСТВА ПРОЕКЦИЙ
        roi_area = (x2 - x1) * (y2 - y1)
        total_scan_area = scan_binary.shape[0] * scan_binary.shape[1]
        coverage_ratio = roi_area / total_scan_area if total_scan_area > 0 else 0

        # Количество апертур в ROI
        roi_binary = scan_binary[y1:y2, x1:x2]
        apertures_in_roi = np.sum(roi_binary > 0)
        total_apertures = np.sum(scan_binary > 0)
        aperture_coverage = apertures_in_roi / \
            total_apertures if total_apertures > 0 else 0

        projection_metrics = {
            'y_shift': y_shift,
            'x_shift': x_shift,
            'y_correlation': y_correlation,
            'x_correlation': x_correlation,
            'roi_width': x2 - x1,
            'roi_height': y2 - y1,
            'roi_area': roi_area,
            'coverage_ratio': coverage_ratio,
            'apertures_in_roi': apertures_in_roi,
            'aperture_coverage': aperture_coverage,
            'mean_correlation': (x_correlation + y_correlation) / 2,
            'ref_width': ref_gray.shape[1],
            'ref_height': ref_gray.shape[0]
        }

        return (x1, y1, x2, y2), projection_metrics

    def _calculate_strategy_specific_metrics(self,
                                             roi_coords: Tuple[int, int, int, int],
                                             contours: List[np.ndarray],
                                             context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Расчет специфичных метрик для проекционной стратегии.

        Returns:
            Dict[str, Any]: Дополнительные метрики для проекционной стратегии
        """
        debug_mode = context.get('debug_mode', False)

        # Получаем эталонное изображение для повторного расчета проекций
        ref_gray = context.get('reference_image')
        if ref_gray is None:
            return {}

        # Повторяем проекционный анализ для получения метрик
        scan_binary = self._get_original_image_from_context(context)
        _, projection_metrics = self._find_roi_by_projection_comparison(
            scan_binary, ref_gray, debug_mode
        )

        strategy_metrics = {
            'projection_correlation_x': projection_metrics.get('x_correlation', 0),
            'projection_correlation_y': projection_metrics.get('y_correlation', 0),
            'projection_mean_correlation': projection_metrics.get('mean_correlation', 0),
            'projection_shift_x': projection_metrics.get('x_shift', 0),
            'projection_shift_y': projection_metrics.get('y_shift', 0),
            'projection_coverage_ratio': projection_metrics.get('coverage_ratio', 0),
            'projection_aperture_coverage': projection_metrics.get('aperture_coverage', 0)
        }

        if debug_mode:
            self.debug_fmt.debug(
                f"Специфичные метрики проекции: "
                f"correlation={strategy_metrics['projection_mean_correlation']:.3f}, "
                f"shift=({strategy_metrics['projection_shift_x']}, {strategy_metrics['projection_shift_y']})",
                indent=3
            )

        return strategy_metrics

    def _calculate_roi_metrics(self, roi_coords: Tuple[int, ...],
                               original_shape: Tuple[int, ...],
                               contours: List[np.ndarray],
                               context: Dict[str, Any]) -> Dict[str, float]:
        """
        Переопределение расчета метрик с учетом специфики проекционного метода.

        Returns:
            Dict[str, float]: Метрики с гарантированным composite_score
        """
        # 1. БАЗОВЫЕ МЕТРИКИ ИЗ РОДИТЕЛЬСКОГО КЛАССА
        base_metrics = super()._calculate_roi_metrics(
            roi_coords, original_shape, contours, context
        )

        # 2. СПЕЦИФИЧНЫЕ МЕТРИКИ ПРОЕКЦИОННОГО МЕТОДА
        strategy_metrics = self._calculate_strategy_specific_metrics(
            roi_coords, contours, context
        )

        # 3. КОРРЕКТИРОВКА COMPOSITE_SCORE С УЧЕТОМ КАЧЕСТВА ПРОЕКЦИЙ
        projection_correlation = strategy_metrics.get(
            'projection_mean_correlation', 0.5)

        # Улучшаем composite_score если проекционная корреляция высокая
        # ±0.1 за отклонение от 0.5
        correlation_boost = 0.2 * (projection_correlation - 0.5)
        adjusted_composite = max(
            0.0, min(1.0, base_metrics['composite_score'] + correlation_boost))

        # 4. ОБЪЕДИНЕНИЕ МЕТРИК
        return {
            **base_metrics,
            'composite_score': adjusted_composite,  # ← СКОРРЕКТИРОВАННЫЙ
            **strategy_metrics
        }

    def _apply_padding(self, roi_coords: Tuple[int, int, int, int],
                       image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Применяет padding к координатам ROI."""
        x1, y1, x2, y2 = roi_coords
        padding = self.crop_padding

        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(image_shape[1], x2 + padding)
        y2_pad = min(image_shape[0], y2 + padding)

        return x1_pad, y1_pad, x2_pad, y2_pad

    def _get_original_image_from_context(self, context: Dict[str, Any]) -> np.ndarray:
        """Извлекает оригинальное изображение из контекста."""
        # В проекционной стратегии изображение может быть в контексте
        # или передаваться через input_data в execute
        return context.get('scan_image')

    def is_applicable(self, input_data: Any, context: Dict[str, Any]) -> bool:
        """
        Проверка применимости проекционной стратегии.

        Returns:
            bool: True если есть эталонное изображение для сравнения
        """
        ref_gray = context.get('reference_image')
        if ref_gray is None:
            return False

        # Проверяем что изображения имеют сравнимые размеры
        try:
            scan_image = self._get_original_image(input_data)
            # Проекционный метод требует что скан не меньше эталона
            return (scan_image.shape[0] >= ref_gray.shape[0] * 0.5 and
                    scan_image.shape[1] >= ref_gray.shape[1] * 0.5)
        except:
            return False
