"""
Базовый класс для стратегий совмещения изображений.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# pylint: disable=no-member
import cv2

import numpy as np

from ..base_strategies import AlignmentStrategy as BaseAlignmentStrategy
from .alignment_utils import AlignmentUtils


class AlignmentStrategy(BaseAlignmentStrategy):
    """Базовый класс для стратегий совмещения изображений."""

    def __init__(self, strategy_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(strategy_name, config)
        self.correlation_threshold = self.config.get(
            'correlation_threshold', 0.85)

        # Предполагаем, что config содержит нужные параметры

        # ConfigService имеет все нужные методы
        debug_mode = self.config.get('debug_mode', False)
        self.alignment_utils = AlignmentUtils(debug_mode=debug_mode)

    def _extract_image_from_input(self, input_data: Any, input_name: str = "input") -> np.ndarray:
        """Извлекает изображение из входных данных."""
        debug_mode = self.config.get('debug_mode', False)
        if debug_mode and self.debug_fmt:
            self.debug_fmt.debug(
                f"Извлечение изображения из входных данных: {input_name}", indent=1)
            self.debug_fmt.debug(
                f"Тип данных: {type(input_data)}", indent=2)

        if isinstance(input_data, dict):
            if debug_mode and self.debug_fmt:
                self.debug_fmt.debug(
                    f"Ключи словаря: {list(input_data.keys())}", indent=2)

            # Пробуем разные ключи для извлечения изображения
            image_keys = ['image', 'roi_image', 'binary_image',
                          'aligned_image', 'processed_image',]
            for key in image_keys:
                if key in input_data and input_data[key] is not None:
                    if debug_mode and self.debug_fmt:
                        self.debug_fmt.debug(
                            f"Найдено изображение по ключу: {key}", indent=3)
                    return input_data[key]

            # Если не нашли изображение
            if debug_mode and self.debug_fmt:
                self.debug_fmt.warn(
                    f"Изображение не найдено в словаре. Доступные ключи: {list(input_data.keys())}", indent=2)
            raise ValueError(
                f"Изображение не найдено во входных данных. Доступные ключи: {list(input_data.keys())}")

        elif isinstance(input_data, np.ndarray):
            if debug_mode and self.debug_fmt:
                self.debug_fmt.debug(
                    f"Прямое изображение, размер: {input_data.shape}", indent=2)
            return input_data
        else:
            if debug_mode and self.debug_fmt:
                self.debug_fmt.error(
                    f"Неподдерживаемый тип данных: {type(input_data)}", indent=2)
            raise ValueError(
                f"Неподдерживаемый тип входных данных: {type(input_data)}")

    def _align_image_sizes(self, reference: np.ndarray, scan: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Выравнивает размеры изображений, учитывая возможный поворот на 90 градусов.

        Args:
            reference: Эталонное изображение.
            scan: Сканированное изображение.

        Returns:
            Кортеж: (выровненное эталонное, выровненное сканированное, был_ли_поворот)
        """
        ref_h, ref_w = reference.shape[:2]
        scan_h, scan_w = scan.shape[:2]

        # Проверяем, нужно ли вращать сканированное изображение на 90 градусов
        should_rotate = self._should_rotate_90(reference, scan)

        if should_rotate:
            # Поворачиваем сканированное изображение на 90 градусов по часовой стрелке
            scan = cv2.rotate(scan, cv2.ROTATE_90_CLOCKWISE)
            scan_h, scan_w = scan.shape[:2]
            if self.debug_fmt:
                self.debug_fmt.debug(
                    "Изображение скана повернуто на 90° по часовой стрелке", indent=1)

        # Теперь выравниваем размеры
        max_height = max(ref_h, scan_h)
        max_width = max(ref_w, scan_w)

        # Создаем новые изображения с черным фоном
        ref_aligned = np.zeros((max_height, max_width), dtype=reference.dtype)
        scan_aligned = np.zeros((max_height, max_width), dtype=scan.dtype)

        # Размещаем оригинальные изображения в центре новых
        ref_aligned[:ref_h, :ref_w] = reference
        scan_aligned[:scan_h, :scan_w] = scan

        return ref_aligned, scan_aligned, should_rotate

    def _should_rotate_90(self, reference: np.ndarray, scan: np.ndarray) -> bool:
        """
        Определяет, нужно ли вращать сканированное изображение на 90 градусов.

        Args:
            reference: Эталонное изображение.
            scan: Сканированное изображение.

        Returns:
            True если нужно вращать, False если нет.
        """
        ref_h, ref_w = reference.shape[:2]
        scan_h, scan_w = scan.shape[:2]

        # Соотношения сторон
        ref_ratio = ref_w / ref_h if ref_h > 0 else 1.0
        scan_ratio = scan_w / scan_h if scan_h > 0 else 1.0

        # Если соотношения сторон сильно отличаются, проверяем возможность поворота
        if abs(ref_ratio - scan_ratio) > 0.3:
            # Проверяем, будет ли соотношение лучше после поворота
            rotated_scan_ratio = scan_h / scan_w if scan_w > 0 else 1.0
            if abs(ref_ratio - rotated_scan_ratio) < abs(ref_ratio - scan_ratio):
                return True

        # Дополнительная проверка: если размеры примерно одинаковы, но ориентация разная
        size_diff_original = abs(ref_h - scan_h) + abs(ref_w - scan_w)
        size_diff_rotated = abs(ref_h - scan_w) + abs(ref_w - scan_h)

        if size_diff_rotated < size_diff_original * 0.8:
            return True

        return False

    def _calculate_metrics(
        self,
        ref_binary: np.ndarray,
        aligned_image: np.ndarray,
        correlation: float,
        mean_contour_distance: float = 0.0
    ) -> Dict[str, Any]:
        """
        Вычисляет метрики совмещения используя готовый метод из utils.
        """
        from .alignment_utils import AlignmentUtils  # если нужно
        # TODO это конечно очень не правильно, а если его вынести вверх - то образуется цикличесая ссылка.

        debug_mode = self.config.get('debug_mode', False)
        if debug_mode and self.debug_fmt:
            self.debug_fmt.debug("Вычисление метрик совмещения", indent=1)

        try:
            # Используем готовый метод из AlignmentUtils который уже защищен от ошибок
            metrics = AlignmentUtils.calculate_alignment_metrics(
                ref_binary, aligned_image, correlation, mean_contour_distance
            )

            if debug_mode and self.debug_fmt:
                self.debug_fmt.metrics_table("Метрики совмещения", metrics)
            return metrics

        except Exception as e:
            if debug_mode and self.debug_fmt:
                self.debug_fmt.error(
                    f"Ошибка при вычислении метрик: {e}", indent=1)

            return {
                'alignment_metrics': {
                    'correlation': correlation,
                    'iou': 0.0,
                    'dice_coefficient': 0.0,
                    'intersection_pixels': 0,
                    'union_pixels': 0,
                    'mean_contour_distance': mean_contour_distance,
                    'ref_contours_count': 0,
                    'aligned_contours_count': 0
                }
            }

    def _save_debug_images(
        self,
        reference: np.ndarray,
        scan: np.ndarray,
        aligned_image: np.ndarray,
        rotation_or_angle: float,
        strategy_name: str
    ):
        """
        Сохраняет отладочные изображения.

        Args:
            reference: Эталонное изображение.
            scan: Исходный скан.
            aligned_image: Совмещенное изображение.
            rotation_or_angle: Угол поворота или угол различия.
            strategy_name: Название стратегии для имени файла.
        """
        debug_dir = Path("debug_output")
        debug_dir.mkdir(exist_ok=True)

        ref_color = (
            cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
            if len(reference.shape) == 2 else reference.copy()
        )
        scan_color = (
            cv2.cvtColor(scan, cv2.COLOR_GRAY2BGR)
            if len(scan.shape) == 2 else scan.copy()
        )
        aligned_color = (
            cv2.cvtColor(aligned_image, cv2.COLOR_GRAY2BGR)
            if len(aligned_image.shape) == 2 else aligned_image.copy()
        )

        overlay = cv2.addWeighted(ref_color, 0.5, aligned_color, 0.5, 0.0)

        overlay_filename = f"{strategy_name}_angle_{rotation_or_angle:.2f}_overlay.png"
        aligned_filename = f"{strategy_name}_angle_{rotation_or_angle:.2f}_aligned.png"

        cv2.imwrite(str(debug_dir / overlay_filename), overlay)
        cv2.imwrite(str(debug_dir / aligned_filename), aligned_color)

        if self.debug_fmt:
            self.debug_fmt.debug(
                f"Сохранены отладочные изображения:", indent=1)
            self.debug_fmt.debug(f"  - {overlay_filename}", indent=2)
            self.debug_fmt.debug(f"  - {aligned_filename}", indent=2)

    def _extract_input_image(self, input_data: Any, image_key: str = 'image') -> np.ndarray:
        """
        Универсальный метод извлечения изображения из входных данных.
        """
        debug_mode = self.config.get('debug_mode', False)

        if debug_mode and self.debug_fmt:
            self.debug_fmt.debug(
                f"Извлечение изображения по ключу: {image_key}", indent=1)
            self.debug_fmt.debug(
                f"Тип входных данных: {type(input_data)}", indent=2)

        if isinstance(input_data, dict):
            if debug_mode and self.debug_fmt:
                self.debug_fmt.debug(
                    f"Входные данные - словарь, извлекаем '{image_key}'...", indent=2)
                self.debug_fmt.debug(
                    f"Доступные ключи: {list(input_data.keys())}", indent=3)

            image = input_data.get(image_key)
            if image is None:
                raise ValueError(
                    f"Ключ '{image_key}' не найден во входных данных")

            # 🔧 ДОПОЛНИТЕЛЬНАЯ ОБРАБОТКА: если image тоже словарь, извлекаем изображение
            if isinstance(image, dict):
                if debug_mode and self.debug_fmt:
                    self.debug_fmt.debug(
                        f"Изображение - словарь, извлекаем фактическое изображение...", indent=3)
                    self.debug_fmt.debug(
                        f"Ключи словаря изображения: {list(image.keys())}", indent=4)

                # Пробуем разные ключи для извлечения изображения
                possible_keys = ['image', 'roi_image',
                                 'binary_image', 'aligned_image']
                for key in possible_keys:
                    if key in image and image[key] is not None:
                        actual_image = image[key]
                        if isinstance(actual_image, np.ndarray):
                            if debug_mode and self.debug_fmt:
                                self.debug_fmt.debug(
                                    f"Извлечено изображение по ключу '{key}', размер: {actual_image.shape}", indent=4)
                            return actual_image

                # Если не нашли изображение, но есть другие данные
                raise ValueError(
                    f"Изображение не найдено в словаре {image_key}")

            if debug_mode and self.debug_fmt:
                self.debug_fmt.debug(
                    f"Тип извлеченного изображения: {type(image)}", indent=3)

            if isinstance(image, np.ndarray):
                if debug_mode and self.debug_fmt:
                    self.debug_fmt.debug(
                        f"Размер изображения: {image.shape}", indent=3)
                return image

            raise ValueError(
                f"Изображение по ключу '{image_key}' не является numpy массивом")

        elif isinstance(input_data, np.ndarray):
            if debug_mode and self.debug_fmt:
                self.debug_fmt.debug(
                    f"Прямой numpy массив, размер: {input_data.shape}", indent=2)
            return input_data

        else:
            if debug_mode and self.debug_fmt:
                self.debug_fmt.error(
                    f"Неподдерживаемый тип входных данных: {type(input_data)}", indent=2)
            raise ValueError(
                f"Неподдерживаемый тип входных данных: {type(input_data)}")
