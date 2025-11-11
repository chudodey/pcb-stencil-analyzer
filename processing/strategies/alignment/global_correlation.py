"""
Стратегия совмещения на основе глобальной корреляции с использованием шаблонного соответствия.
"""

import time
from typing import Dict, Any, Tuple, Optional

# pylint: disable=no-member
import cv2

import numpy as np

from .base_alignment import AlignmentStrategy
from ..base_strategies import StrategyResult


class GlobalCorrelationAlignmentStrategy(AlignmentStrategy):
    """Стратегия совмещения на основе сравнения шаблонов с учетом поворотов."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__("GlobalCorrelationAlignment", config or {})
        self._rotation_angles = [0, 90, 180, 270]
        self._template_method = cv2.TM_CCOEFF_NORMED

    def execute(self, input_data: Any, context: Dict[str, Any]) -> StrategyResult:
        """Выполняет совмещение изображений методом глобальной корреляции."""
        start_time = time.time()

        try:
            debug_mode = context.get('debug_mode', False)
            if self.debug_fmt:
                self.debug_fmt.info(
                    "Начало глобальной корреляционной стратегии", indent=1)

            # Извлечение изображения
            reference = self._extract_input_image(input_data, 'reference')
            scan_image = self._extract_input_image(input_data, 'scan')

            if self.debug_fmt:
                self.debug_fmt.debug("Исходные размеры изображений:", indent=2)
                self.debug_fmt.debug(f"Эталон: {reference.shape}", indent=3)
                self.debug_fmt.debug(f"Скан: {scan_image.shape}", indent=3)

            # Выравнивание размеров
            ref_aligned, scan_aligned, was_rotated = self._align_image_sizes(
                reference, scan_image
            )

            if self.debug_fmt:
                self.debug_fmt.debug("Выровненные размеры:", indent=2)
                self.debug_fmt.debug(f"Эталон: {ref_aligned.shape}", indent=3)
                self.debug_fmt.debug(f"Скан: {scan_aligned.shape}", indent=3)
                if was_rotated:
                    self.debug_fmt.debug("Скан был повернут на 90°", indent=3)

            # Детальный анализ изображений перед совмещением
            if debug_mode and self.debug_fmt:
                self._debug_image_analysis(
                    ref_aligned, scan_aligned, "ДО совмещения")

            # Поиск совмещения на ВЫРОВНЕННЫХ изображениях
            transform, correlation, rotation = self._find_best_template_match(
                ref_aligned, scan_aligned, debug_mode
            )

            if self.debug_fmt:
                self.debug_fmt.success(
                    "Найдена лучшая трансформация:", indent=2)
                self.debug_fmt.debug(
                    f"Корреляция: {correlation:.6f}", indent=3)
                self.debug_fmt.debug(f"Поворот: {rotation}°", indent=3)

            # 🔧 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Правильное применение трансформации
            aligned_image = self._apply_transformation(
                scan_aligned, transform, ref_aligned.shape, rotation, debug_mode)

            # Обрезаем черные границы
            aligned_cropped = self._crop_to_content(aligned_image)

            if self.debug_fmt:
                self.debug_fmt.debug("Обрезанное изображение:", indent=2)
                self.debug_fmt.debug(
                    f"Размер: {aligned_cropped.shape}", indent=3)

            # Детальный анализ после совмещения
            if debug_mode and self.debug_fmt:
                self._debug_image_analysis(
                    ref_aligned, aligned_cropped, "ПОСЛЕ совмещения")

            # Вычисляем метрики на ВЫРОВНЕННЫХ изображениях
            metrics = self._calculate_metrics(
                ref_aligned, aligned_image, correlation
            )

            if self.debug_fmt:
                self.debug_fmt.metrics_table(
                    "Метрики совмещения", metrics.get('alignment_metrics', {}))

            result = StrategyResult(
                strategy_name=self.name,
                success=True,
                result_data={
                    'aligned_image': aligned_cropped,
                    'transform': transform,
                    'correlation': correlation,
                    'rotation_angle': rotation
                },
                metrics=metrics,
                processing_time=time.time() - start_time
            )

            if self.debug_fmt:
                self.debug_fmt.success(
                    f"Глобальная корреляционная стратегия завершена за {result.processing_time:.3f} сек",
                    indent=1
                )

            return result

        except Exception as e:
            error_msg = f"Ошибка в глобальной корреляционной стратегии: {str(e)}"
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

    def _debug_image_analysis(self, image1: np.ndarray, image2: np.ndarray, stage: str):
        """Детальный анализ изображений для отладки."""
        if not self.debug_fmt:
            return

        self.debug_fmt.debug(f"АНАЛИЗ ИЗОБРАЖЕНИЙ {stage}:", indent=2)

        # Анализ первого изображения
        self.debug_fmt.debug("Изображение 1:", indent=3)
        self.debug_fmt.debug(f"Размер: {image1.shape}", indent=4)
        self.debug_fmt.debug(f"Тип данных: {image1.dtype}", indent=4)
        self.debug_fmt.debug(
            f"Диапазон: [{image1.min()}, {image1.max()}]", indent=4)
        self.debug_fmt.debug(f"Среднее: {image1.mean():.3f}", indent=4)
        non_zero_count = np.count_nonzero(image1)
        self.debug_fmt.debug(
            f"Ненулевых пикселей: {non_zero_count}/{image1.size} ({100 * non_zero_count / image1.size:.1f}%)", indent=4)

        # Анализ второго изображения
        self.debug_fmt.debug("Изображение 2:", indent=3)
        self.debug_fmt.debug(f"Размер: {image2.shape}", indent=4)
        self.debug_fmt.debug(f"Тип данных: {image2.dtype}", indent=4)
        self.debug_fmt.debug(
            f"Диапазон: [{image2.min()}, {image2.max()}]", indent=4)
        self.debug_fmt.debug(f"Среднее: {image2.mean():.3f}", indent=4)
        non_zero_count = np.count_nonzero(image2)
        self.debug_fmt.debug(
            f"Ненулевых пикселей: {non_zero_count}/{image2.size} ({100 * non_zero_count / image2.size:.1f}%)", indent=4)

    def _find_best_template_match(
        self,
        ref_binary: np.ndarray,
        scan_binary: np.ndarray,
        debug_mode: bool
    ) -> Tuple[np.ndarray, float, int]:
        """
        Находит лучшую трансформацию с использованием cv2.matchTemplate для разных поворотов.
        """
        best_score = -float('inf')
        best_transform = None
        best_correlation = 0.0
        best_rotation = 0

        if self.debug_fmt:
            self.debug_fmt.debug("Поиск шаблонного соответствия:", indent=2)
            self.debug_fmt.debug(
                f"Тестируемые углы: {self._rotation_angles}", indent=3)

        for angle in self._rotation_angles:
            if self.debug_fmt:
                self.debug_fmt.debug(
                    f"Тестирование поворота: {angle}°", indent=3)

            # Поворачиваем эталонное изображение
            rot_ref = self._rotate_image(ref_binary, angle)

            if self.debug_fmt:
                self.debug_fmt.debug(
                    f"Повернутый эталон: {rot_ref.shape}", indent=4)

            # Масштабируем сканированное изображение если нужно
            scan_resized = self._resize_scan_to_fit_template(
                rot_ref, scan_binary
            )

            if self.debug_fmt and scan_resized.shape != scan_binary.shape:
                self.debug_fmt.debug(
                    f"Масштабированный скан: {scan_resized.shape}", indent=4)

            # Выполняем сравнение шаблонов
            correlation, max_loc = self._perform_template_match(
                rot_ref, scan_resized
            )

            if self.debug_fmt:
                self.debug_fmt.debug(
                    f"Корреляция: {correlation:.4f}", indent=4)
                self.debug_fmt.debug(f"Позиция максимума: {max_loc}", indent=4)

            if correlation > best_score:
                best_score = correlation
                best_correlation = correlation
                best_rotation = angle

                # Вычисляем трансформацию с учетом поворота
                best_transform = self._calculate_transformation_matrix(
                    angle, max_loc, scan_resized.shape, rot_ref.shape
                )

                if self.debug_fmt:
                    self.debug_fmt.debug(
                        "Обновлен лучший результат:", indent=4)
                    self.debug_fmt.debug(f"Угол: {angle}°", indent=5)
                    self.debug_fmt.debug(
                        f"Корреляция: {correlation:.4f}", indent=5)

        if best_transform is None:
            error_msg = "Не найдено подходящей трансформации для любого угла поворота"
            if self.debug_fmt:
                self.debug_fmt.error(error_msg, indent=2)
            raise ValueError(error_msg)

        if self.debug_fmt:
            self.debug_fmt.debug("Лучшая трансформация:", indent=2)
            self.debug_fmt.debug(f"Угол: {best_rotation}°", indent=3)
            self.debug_fmt.debug(
                f"Корреляция: {best_correlation:.6f}", indent=3)

        return best_transform, best_correlation, best_rotation

    def _rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """Поворачивает изображение на заданный угол."""
        if angle == 0:
            return image
        elif angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            if self.debug_fmt:
                self.debug_fmt.warn(
                    f"Неподдерживаемый угол поворота: {angle}, используется 0°", indent=2)
            return image

    def _resize_scan_to_fit_template(
        self,
        template: np.ndarray,
        scan: np.ndarray
    ) -> np.ndarray:
        """
        Масштабирует сканированное изображение чтобы оно помещалось в шаблон.
        """
        if scan.shape[0] > template.shape[0] or scan.shape[1] > template.shape[1]:
            scale = min(
                template.shape[0] / scan.shape[0],
                template.shape[1] / scan.shape[1]
            )
            scan_resized = cv2.resize(
                scan,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA
            )
            if self.debug_fmt:
                self.debug_fmt.debug(
                    f"Масштабирование скана: коэффициент {scale:.4f}", indent=4)
            return scan_resized
        else:
            return scan

    def _perform_template_match(
        self,
        template: np.ndarray,
        scan: np.ndarray
    ) -> Tuple[float, Tuple[int, int]]:
        """Выполняет сравнение шаблонов и возвращает лучшую корреляцию и позицию."""
        result = cv2.matchTemplate(template, scan, self._template_method)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # 🔧 Простое и надежное преобразование в Tuple[int, int]
        max_loc_tuple = (int(max_loc[0]), int(max_loc[1]))

        return max_val, max_loc_tuple

    def _calculate_transformation_matrix(
        self,
        angle: int,
        max_loc: Tuple[int, int],
        scan_shape: Tuple[int, int],
        ref_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Вычисляет матрицу трансформации с учетом поворота и смещения."""
        x_shift, y_shift = max_loc
        scan_height, scan_width = scan_shape
        ref_height, ref_width = ref_shape

        if self.debug_fmt:
            self.debug_fmt.debug("Параметры трансформации:", indent=4)
            self.debug_fmt.debug(f"Угол: {angle}°", indent=5)
            self.debug_fmt.debug(f"Позиция: {max_loc}", indent=5)
            self.debug_fmt.debug(f"Размер скана: {scan_shape}", indent=5)
            self.debug_fmt.debug(f"Размер эталона: {ref_shape}", indent=5)

        # Корректируем смещение для разных поворотов
        if angle == 90:
            x_shift, y_shift = (
                y_shift, ref_height - x_shift - scan_width
            )
        elif angle == 180:
            # 🔧 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Для поворота на 180° используем специальную логику
            x_shift, y_shift = (
                ref_width - scan_width,  # Центрируем по ширине
                ref_height - scan_height  # Центрируем по высоте
            )
        elif angle == 270:
            x_shift, y_shift = (
                ref_width - y_shift - scan_height,
                x_shift
            )

        if self.debug_fmt:
            self.debug_fmt.debug(
                f"Скорректированное смещение: X={x_shift}, Y={y_shift}", indent=4)

        # Создаем матрицу трансформации
        angle_rad = np.radians(angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        transform = np.array([
            [cos_angle, -sin_angle, float(x_shift)],
            [sin_angle, cos_angle, float(y_shift)]
        ], dtype=np.float32)

        return transform

    def _apply_transformation(
        self,
        image: np.ndarray,
        transform: np.ndarray,
        output_shape: Tuple[int, int],
        rotation: int,
        debug_mode: bool = False
    ) -> np.ndarray:
        """
        Применяет трансформацию к изображению с правильной обработкой поворотов.
        """
        if self.debug_fmt:
            self.debug_fmt.debug(
                f"Применение трансформации с поворотом: {rotation}°", indent=3)

        result: np.ndarray = image.copy()  # Значение по умолчанию

        # 🔧 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Для простых поворотов используем встроенные функции OpenCV
        if rotation in [0, 90, 180, 270]:
            # Для стандартных поворотов используем оптимизированные функции OpenCV
            if rotation == 0:
                result = image.copy()
            elif rotation == 90:
                result = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                result = cv2.rotate(image, cv2.ROTATE_180)
            elif rotation == 270:
                result = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Если размеры не совпадают, делаем resize
            if result.shape[:2] != output_shape[:2]:
                result = cv2.resize(result, (output_shape[1], output_shape[0]))
        else:
            # Для сложных трансформаций используем warpAffine
            result = cv2.warpAffine(
                image, transform, (output_shape[1], output_shape[0]))

        if debug_mode and self.debug_fmt:
            self.debug_fmt.debug("РЕЗУЛЬТАТ ТРАНСФОРМАЦИИ:", indent=3)
            self.debug_fmt.debug("Входное изображение:", indent=4)
            self.debug_fmt.debug(f"Размер: {image.shape}", indent=5)
            self.debug_fmt.debug(f"Тип данных: {image.dtype}", indent=5)
            self.debug_fmt.debug(
                f"Диапазон: [{image.min()}, {image.max()}]", indent=5)
            self.debug_fmt.debug("Выходное изображение:", indent=4)
            self.debug_fmt.debug(f"Размер: {result.shape}", indent=5)
            self.debug_fmt.debug(f"Тип данных: {result.dtype}", indent=5)
            self.debug_fmt.debug(
                f"Диапазон: [{result.min()}, {result.max()}]", indent=5)
            non_zero_count = np.count_nonzero(result)
            self.debug_fmt.debug(
                f"Ненулевых пикселей: {non_zero_count}/{result.size}", indent=5)

        return result

    def _crop_to_content(self, image: np.ndarray) -> np.ndarray:
        """Обрезает изображение до ненулевой области."""
        # Используем более надежный метод поиска ненулевых областей
        if len(image.shape) == 3:
            # Для цветных изображений проверяем все каналы
            non_zero_mask = np.any(image > 0, axis=2)
        else:
            # Для grayscale
            non_zero_mask = image > 0

        non_zero_coords = np.column_stack(np.where(non_zero_mask))

        if len(non_zero_coords) > 0:
            y_coords = non_zero_coords[:, 0]
            x_coords = non_zero_coords[:, 1]
            x, y = np.min(x_coords), np.min(y_coords)
            w, h = np.max(x_coords) - x + 1, np.max(y_coords) - y + 1

            # Добавляем небольшой отступ
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)

            cropped = image[y:y+h, x:x+w]

            if self.debug_fmt:
                self.debug_fmt.debug("Обрезка изображения:", indent=3)
                self.debug_fmt.debug(
                    f"Исходный размер: {image.shape}", indent=4)
                self.debug_fmt.debug(
                    f"Обрезанный размер: {cropped.shape}", indent=4)
                self.debug_fmt.debug(
                    f"Ненулевых пикселей: {len(non_zero_coords)}", indent=4)

            return cropped
        else:
            if self.debug_fmt:
                self.debug_fmt.warn(
                    "Не найдено ненулевых областей, используется полное изображение", indent=3)
            return image
