# processing/strategies/roi_extraction/base_roi_extraction.py
"""
Базовый класс для стратегий выделения области интереса (ROI)

СООТВЕТСТВИЕ ИНСТРУКЦИИ:
- Уровень 4: ProcessingStrategy (Исполнитель)
- Гарантированный расчет composite_score в метриках
- Централизованное логирование через execute_with_logging
- Детальное описание процесса выполнения ROI extraction
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np

from ..base_strategies import ROIExtractionStrategy, StrategyResult


class BaseROIExtractionStrategy(ROIExtractionStrategy):
    """Базовый класс для стратегий выделения ROI с гарантированным composite_score."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config or {})
        self.margin_pixels = self.config.get('margin_pixels', 50)

    def execute(self, input_data: Any, context: Dict[str, Any]) -> StrategyResult:
        """
        Основная логика выполнения стратегии ROI extraction.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Вызывается через execute_with_logging из родительского класса
        - Возвращает StrategyResult с гарантированным composite_score
        - Логирование выполняется автоматически в родительском классе
        """
        start_time = time.time()
        debug_mode = context.get('debug_mode', False)

        try:
            # 1. ИЗВЛЕЧЕНИЕ ВХОДНЫХ ДАННЫХ
            if debug_mode:
                self.debug_fmt.debug("Извлечение входных данных...", indent=2)

            original_image = self._get_original_image(input_data)
            contours = self._get_contours_from_input(input_data)
            contours = [c for c in contours if len(c) > 0 and c.size > 0]

            if debug_mode:
                self.debug_fmt.debug(
                    f"Изображение: {original_image.shape}, Контуры: {len(contours)}",
                    indent=3
                )

            # 2. ВЫПОЛНЕНИЕ ОСНОВНОЙ ЛОГИКИ ROI EXTRACTION
            if debug_mode:
                self.debug_fmt.debug("Выполнение выделения ROI...", indent=2)

            roi_coords = self._extract_roi(original_image, contours, context)

            if debug_mode:
                self.debug_fmt.debug(
                    f"ROI координаты: {roi_coords}", indent=3
                )

            # 3. ВЫРЕЗАНИЕ ROI ИЗ ИЗОБРАЖЕНИЯ
            x1, y1, x2, y2 = roi_coords
            roi_image = original_image[y1:y2, x1:x2]

            # 4. РАСЧЕТ МЕТРИК КАЧЕСТВА С ГАРАНТИРОВАННЫМ COMPOSITE_SCORE
            if debug_mode:
                self.debug_fmt.debug("Расчет метрик качества ROI...", indent=2)

            metrics = self._calculate_roi_metrics(
                roi_coords, original_image.shape, contours, context
            )

            if debug_mode:
                self.debug_fmt.debug(
                    f"composite_score гарантирован: {metrics.get('composite_score', 0):.3f}",
                    indent=3
                )

            # 5. ПОДГОТОВКА РЕЗУЛЬТАТА
            result_data = {
                'roi_coordinates': roi_coords,
                'roi_image': roi_image,
                'original_shape': original_image.shape,
                'contours_count': len(contours)
            }

            # 6. СОХРАНЕНИЕ ОТЛАДОЧНЫХ ИЗОБРАЖЕНИЙ
            if debug_mode and context.get('save_debug_images', False):
                self._save_debug_images(
                    original_image, contours, roi_coords, roi_image, self.name
                )

            processing_time = time.time() - start_time

            return StrategyResult(
                strategy_name=self.name,
                success=True,
                result_data=result_data,
                metrics=metrics,
                processing_time=processing_time,
                error_message=None
            )

        except Exception as error:
            processing_time = time.time() - start_time
            return self._create_error_result(processing_time, error)

    def _calculate_roi_metrics(self, roi_coords: Tuple[int, ...],
                               original_shape: Tuple[int, ...],
                               contours: List[np.ndarray],
                               context: Dict[str, Any]) -> Dict[str, float]:
        """
        Вычисление метрик качества ROI с ГАРАНТИРОВАННЫМ composite_score.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - composite_score всегда в корне метрик
        - Стратегия сама знает как оценить свое качество
        """
        x1, y1, x2, y2 = roi_coords
        roi_width, roi_height = x2 - x1, y2 - y1

        # 1. ПОКРЫТИЕ АПЕРТУР
        contours_in_roi = sum(
            1 for c in contours if self._is_contour_in_roi(c, roi_coords))
        aperture_coverage = contours_in_roi / \
            len(contours) if contours else 1.0

        # 2. СООТВЕТСТВИЕ ГРАНИЦАМ (с учетом поворота на 90 градусов)
        expected_size = context.get('expected_size_px')
        boundary_match = 0.5

        if expected_size:
            expected_width, expected_height = expected_size

            # Оригинальная ориентация
            w_ratio_orig = min(roi_width, expected_width) / \
                max(roi_width, expected_width)
            h_ratio_orig = min(roi_height, expected_height) / \
                max(roi_height, expected_height)
            score_orig = (w_ratio_orig + h_ratio_orig) / 2

            # Повернутая ориентация
            w_ratio_rot = min(roi_width, expected_height) / \
                max(roi_width, expected_height)
            h_ratio_rot = min(roi_height, expected_width) / \
                max(roi_height, expected_width)
            score_rot = (w_ratio_rot + h_ratio_rot) / 2

            boundary_match = max(score_orig, score_rot)

        # 3. СООТВЕТСТВИЕ СООТНОШЕНИЮ СТОРОН
        expected_aspect_ratio = context.get('expected_aspect_ratio', 1.0)
        roi_aspect_ratio = roi_width / roi_height if roi_height > 0 else 1.0
        aspect_ratio_match = min(roi_aspect_ratio, expected_aspect_ratio) / max(
            roi_aspect_ratio, expected_aspect_ratio
        ) if max(roi_aspect_ratio, expected_aspect_ratio) > 0 else 1.0

        # 4. ГАРАНТИРОВАННЫЙ COMPOSITE_SCORE
        composite_score = (
            0.5 * aperture_coverage +      # Важнее всего покрытие апертур
            0.3 * boundary_match +         # Соответствие границам
            0.2 * aspect_ratio_match       # Соответствие соотношению сторон
        )

        return {
            # ← ГАРАНТИРОВАННО
            'composite_score': max(0.0, min(1.0, composite_score)),
            'aperture_coverage': aperture_coverage,
            'boundary_match': boundary_match,
            'aspect_ratio_match': aspect_ratio_match,
            'roi_width': roi_width,
            'roi_height': roi_height,
            'contours_in_roi': contours_in_roi,
            'total_contours': len(contours)
        }

    def _extract_roi(self, image: np.ndarray, contours: List[np.ndarray],
                     context: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """
        Абстрактный метод для реализации конкретной логики выделения ROI.

        Должен быть реализован в дочерних классах.

        Returns:
            Tuple[int, int, int, int]: Координаты ROI (x1, y1, x2, y2)
        """
        raise NotImplementedError(
            "Дочерние классы должны реализовать этот метод")

    def _get_contours_from_input(self, image_data: Any) -> List[np.ndarray]:
        """Извлекает контуры из входных данных бинаризации."""
        debug_mode = self.config.get('debug_mode', False)

        # ✅ Основной случай: image_data - словарь с контурами от бинаризации
        if isinstance(image_data, dict):
            contours = image_data.get('contours', [])
            if debug_mode:
                self.debug_fmt.debug(
                    f"Найдено контуров в словаре: {len(contours)}", indent=3)
            return contours

        # ❌ Fallback
        if debug_mode:
            self.debug_fmt.warn(
                "Контуры не найдены во входных данных", indent=3)
        return []

    def _get_original_image(self, image_data: Any) -> np.ndarray:
        """Извлекает оригинальное изображение из входных данных."""
        debug_mode = self.config.get('debug_mode', False)

        # ✅ Если пришел словарь от бинаризации
        if isinstance(image_data, dict):
            image = image_data.get('binary_image')
            if image is not None:
                if debug_mode:
                    self.debug_fmt.debug(
                        f"Извлечено изображение: {image.shape}", indent=3)
                return image
            else:
                raise ValueError("No 'binary_image' key in input dict")

        # ✅ Если пришло прямо изображение
        elif isinstance(image_data, np.ndarray):
            if debug_mode:
                self.debug_fmt.debug(
                    f"Прямое изображение: {image_data.shape}", indent=3)
            return image_data

        # ❌ Неподдерживаемый тип
        raise ValueError(f"Unsupported input type: {type(image_data)}")

    def _is_contour_in_roi(self, contour: np.ndarray, roi_coords: Tuple[int, ...]) -> bool:
        """Проверяет, находится ли контур полностью внутри ROI."""
        if len(contour) == 0 or contour.size == 0:  # Проверка на пустой контур
            return False

        try:
            cx, cy, cw, ch = cv2.boundingRect(contour)
            x1, y1, x2, y2 = roi_coords
            return (cx >= x1 and cy >= y1 and cx + cw <= x2 and cy + ch <= y2)
        except Exception as e:  # Fallback на случай других OpenCV ошибок
            if self.config.get('debug_mode', False):
                self.debug_fmt.warn(f"Некорректный контур: {str(e)}", indent=4)
            return False

    def _save_debug_images(self, original_image: np.ndarray, contours: List[np.ndarray],
                           roi_coords: Tuple[int, ...], roi_image: np.ndarray, prefix: str):
        """Сохраняет отладочные изображения."""
        debug_dir = Path("debug_output")
        debug_dir.mkdir(exist_ok=True)

        if len(original_image.shape) == 2:
            original_with_contours = cv2.cvtColor(
                original_image.copy(), cv2.COLOR_GRAY2BGR)
        else:
            original_with_contours = original_image.copy()

        cv2.drawContours(original_with_contours, contours, -1, (0, 255, 0), 2)
        x1, y1, x2, y2 = roi_coords
        cv2.rectangle(original_with_contours, (x1, y1),
                      (x2, y2), (255, 0, 0), 3)

        cv2.imwrite(
            str(debug_dir / f"{prefix}_roi_debug.png"), original_with_contours)
        cv2.imwrite(str(debug_dir / f"{prefix}_roi_result.png"), roi_image)

        if hasattr(self, 'debug_fmt'):
            self.debug_fmt.info(f"💾 Сохранены debug изображения для {prefix}")

    def _create_error_result(self, proc_time: float, error: Exception) -> StrategyResult:
        """Создает стандартизированный результат ошибки."""
        return StrategyResult(
            strategy_name=self.name,
            success=False,
            result_data=None,
            metrics={},
            processing_time=proc_time,
            error_message=str(error)
        )

    def _apply_margin(self, coords: Tuple[int, int, int, int], image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Применяет margin к координатам с проверкой границ."""
        x, y, w, h = coords
        margin = self.margin_pixels

        x_start = max(0, x - margin)
        y_start = max(0, y - margin)
        x_end = min(image_shape[1], x + w + margin)
        y_end = min(image_shape[0], y + h + margin)

        return x_start, y_start, x_end, y_end

    def _draw_bbox_overlay(self, base_image: np.ndarray, roi_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """Возвращает копию изображения с нарисованной рамкой ROI."""
        if base_image is None:
            return None
        if len(base_image.shape) == 2:
            overlay = cv2.cvtColor(base_image.copy(), cv2.COLOR_GRAY2BGR)
        else:
            overlay = base_image.copy()
        x1, y1, x2, y2 = roi_coords
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return overlay
