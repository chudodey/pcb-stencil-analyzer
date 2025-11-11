"""
Стратегия совмещения на основе сравнения проекций, инвариантная к повороту.
"""

import time
from typing import Dict, Any, Tuple, Optional

# pylint: disable=no-member
import cv2
import numpy as np

from .base_alignment import AlignmentStrategy
from ..base_strategies import StrategyResult


class ProjectionBasedAlignmentStrategy(AlignmentStrategy):
    """Стратегия совмещения на основе сравнения проекций, инвариантная к повороту."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__("ProjectionBasedAlignment", config or {})
        self.correlation_threshold = config.get(
            'correlation_threshold', 0.1) if config else 0.1

    def execute(self, input_data: Any, context: Dict[str, Any]) -> StrategyResult:
        """
        Выполняет совмещение на основе проекций с учетом поворотов (0, 90, 180, 270).
        """
        start_time = time.time()

        try:
            debug_mode = context.get('debug_mode', False)
            if self.debug_fmt:
                self.debug_fmt.info(
                    "Начало проекционной стратегии совмещения", indent=1)

            # 🔧 ИЗВЛЕЧЕНИЕ ИЗОБРАЖЕНИЯ ИЗ ВХОДНЫХ ДАННЫХ
            reference = self._extract_input_image(input_data, 'reference')
            scan = self._extract_input_image(input_data, "scan")

            if self.debug_fmt:
                self.debug_fmt.debug("Исходные размеры изображений:", indent=2)
                self.debug_fmt.debug(f"Эталон: {reference.shape}", indent=3)
                self.debug_fmt.debug(f"Скан: {scan.shape}", indent=3)

            # ✅ ВЫРАВНИВАЕМ РАЗМЕРЫ
            ref_aligned, scan_aligned, _ = self._align_image_sizes(
                reference, scan
            )

            if self.debug_fmt:
                self.debug_fmt.debug("Выровненные размеры:", indent=2)
                self.debug_fmt.debug(f"Эталон: {ref_aligned.shape}", indent=3)
                self.debug_fmt.debug(f"Скан: {scan_aligned.shape}", indent=3)

            # 🔧 ПОДГОТОВКА БИНАРНЫХ ИЗОБРАЖЕНИЙ
            ref_binary = self._prepare_binary_image(ref_aligned)
            scan_binary = self._prepare_binary_image(scan_aligned)

            transform, corr_h, corr_v, rotation = self._find_best_transform(
                ref_binary, scan_binary, debug_mode
            )

            if self.debug_fmt:
                self.debug_fmt.success(
                    "Найдена лучшая трансформация:", indent=2)
                self.debug_fmt.debug(f"Поворот: {rotation}°", indent=3)
                self.debug_fmt.debug(
                    f"Корреляция горизонтальная: {corr_h:.4f}", indent=3)
                self.debug_fmt.debug(
                    f"Корреляция вертикальная: {corr_v:.4f}", indent=3)

            # 🔧 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Используем scan_aligned вместо scan!
            aligned_image = cv2.warpAffine(
                scan_aligned,  # ✅ ПРАВИЛЬНОЕ ИЗОБРАЖЕНИЕ
                transform,
                (ref_aligned.shape[1], ref_aligned.shape[0])
            )

            # Проверка качества результата
            if max(corr_h, corr_v) < self.correlation_threshold:
                if self.debug_fmt:
                    self.debug_fmt.warn(
                        f"Корреляция ниже порога ({max(corr_h, corr_v):.6f} < {self.correlation_threshold:.3f})",
                        indent=2
                    )

            metrics = self._calculate_metrics(
                ref_aligned, aligned_image, max(corr_h, corr_v))

            if self.debug_fmt:
                self.debug_fmt.metrics_table(
                    "Метрики совмещения", metrics.get('alignment_metrics', {}))

            return StrategyResult(
                strategy_name=self.name,
                success=True,
                result_data={
                    'aligned_image': aligned_image,
                    'transform': transform,
                    'correlation_horizontal': corr_h,
                    'correlation_vertical': corr_v,
                    'rotation_angle': rotation
                },
                metrics=metrics,
                processing_time=time.time() - start_time
            )

        except Exception as e:
            error_msg = f"Ошибка в проекционной стратегии совмещения: {str(e)}"
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
        """Подготавливает бинарное изображение для анализа проекций."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Адаптивная бинаризация для лучших результатов
        if np.unique(image).size <= 2:
            # Уже бинарное
            binary = (image > 0).astype(np.uint8)
        else:
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            binary = (binary > 0).astype(np.uint8)

        return binary

    def _find_best_transform(
        self,
        ref_binary: np.ndarray,
        scan_binary: np.ndarray,
        debug_mode: bool
    ) -> Tuple[np.ndarray, float, float, int]:
        """
        Находит лучшую трансформацию, сравнивая проекции для поворотов 0, 90, 180, 270 градусов.
        """
        min_proj_intensity = 1e-6
        rotations = [0, 90, 180, 270]
        best_score = -float('inf')
        best_transform = None
        best_corr_h = 0.0
        best_corr_v = 0.0
        best_rotation = 0

        if self.debug_fmt:
            self.debug_fmt.debug("Поиск лучшей трансформации:", indent=2)
            self.debug_fmt.debug(f"Тестируемые углы: {rotations}", indent=3)

        for angle in rotations:
            if self.debug_fmt:
                self.debug_fmt.debug(
                    f"Тестирование поворота: {angle}°", indent=3)

            # Поворачиваем эталонное изображение
            rot_ref = self._rotate_image(ref_binary, angle)

            # Вычисляем проекции
            ref_h_proj = np.sum(rot_ref, axis=1).astype(float)
            ref_v_proj = np.sum(rot_ref, axis=0).astype(float)
            scan_h_proj = np.sum(scan_binary, axis=1).astype(float)
            scan_v_proj = np.sum(scan_binary, axis=0).astype(float)

            if self.debug_fmt:
                self.debug_fmt.debug("Максимумы проекций:", indent=4)
                self.debug_fmt.debug(
                    f"Эталон горизонталь: {np.max(ref_h_proj):.1f}", indent=5)
                self.debug_fmt.debug(
                    f"Эталон вертикаль: {np.max(ref_v_proj):.1f}", indent=5)
                self.debug_fmt.debug(
                    f"Скан горизонталь: {np.max(scan_h_proj):.1f}", indent=5)
                self.debug_fmt.debug(
                    f"Скан вертикаль: {np.max(scan_v_proj):.1f}", indent=5)

            # Пропускаем если проекции слишком слабые
            if (np.max(ref_h_proj) < min_proj_intensity or
                np.max(ref_v_proj) < min_proj_intensity or
                np.max(scan_h_proj) < min_proj_intensity or
                    np.max(scan_v_proj) < min_proj_intensity):
                if self.debug_fmt:
                    self.debug_fmt.debug(
                        "Слабые проекции, пропускаем угол", indent=4)
                continue

            # 🔧 УЛУЧШЕННАЯ НОРМАЛИЗАЦИЯ
            ref_h_proj = self._normalize_projection(ref_h_proj)
            ref_v_proj = self._normalize_projection(ref_v_proj)
            scan_h_proj = self._normalize_projection(scan_h_proj)
            scan_v_proj = self._normalize_projection(scan_v_proj)

            # Сглаживание проекций
            ref_h_proj = cv2.GaussianBlur(
                ref_h_proj.reshape(-1, 1), (5, 1), 0).flatten()
            ref_v_proj = cv2.GaussianBlur(
                ref_v_proj.reshape(-1, 1), (5, 1), 0).flatten()
            scan_h_proj = cv2.GaussianBlur(
                scan_h_proj.reshape(-1, 1), (5, 1), 0).flatten()
            scan_v_proj = cv2.GaussianBlur(
                scan_v_proj.reshape(-1, 1), (5, 1), 0).flatten()

            # Поиск лучшего смещения
            y_shift, corr_h = self._find_best_shift(
                ref_h_proj, scan_h_proj, debug_mode)
            x_shift, corr_v = self._find_best_shift(
                ref_v_proj, scan_v_proj, debug_mode)

            score = corr_h + corr_v

            if score > best_score and score > 0:  # Только положительные корреляции
                best_score = score
                best_corr_h = corr_h
                best_corr_v = corr_v
                best_rotation = angle

                # 🔧 ПРАВИЛЬНЫЙ РАСЧЕТ ТРАНСФОРМАЦИИ
                best_transform = self._calculate_transformation_matrix(
                    angle, x_shift, y_shift
                )

                if self.debug_fmt:
                    self.debug_fmt.debug(
                        "Обновлен лучший результат:", indent=4)
                    self.debug_fmt.debug(f"Угол: {angle}°", indent=5)
                    self.debug_fmt.debug(f"Общий счет: {score:.6f}", indent=5)
                    self.debug_fmt.debug(f"Сдвиг X: {x_shift}", indent=5)
                    self.debug_fmt.debug(f"Сдвиг Y: {y_shift}", indent=5)

        if best_transform is None:
            error_msg = "Не найдено подходящей трансформации для любого угла поворота"
            if self.debug_fmt:
                self.debug_fmt.error(error_msg, indent=2)
            raise ValueError(error_msg)

        if self.debug_fmt:
            self.debug_fmt.debug("Лучшая трансформация:", indent=2)
            self.debug_fmt.debug(f"Угол: {best_rotation}°", indent=3)
            self.debug_fmt.debug(f"Общий счет: {best_score:.6f}", indent=3)
            self.debug_fmt.debug(
                f"Корреляция горизонтальная: {best_corr_h:.6f}", indent=3)
            self.debug_fmt.debug(
                f"Корреляция вертикальная: {best_corr_v:.6f}", indent=3)

        return best_transform, best_corr_h, best_corr_v, best_rotation

    def _normalize_projection(self, projection: np.ndarray) -> np.ndarray:
        """Нормализует проекцию к диапазону [0, 1]."""
        proj = projection.copy()
        proj_min = np.min(proj)
        proj_max = np.max(proj)

        if proj_max - proj_min > 1e-6:
            proj = (proj - proj_min) / (proj_max - proj_min)
        else:
            proj = np.zeros_like(proj)

        return proj

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
            return image

    def _calculate_transformation_matrix(
        self,
        angle: int,
        x_shift: int,
        y_shift: int
    ) -> np.ndarray:
        """Вычисляет матрицу трансформации с учетом поворота и смещения."""
        angle_rad = np.radians(angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        # 🔧 ПРАВИЛЬНАЯ МАТРИЦА ПРЕОБРАЗОВАНИЯ
        transform = np.array([
            [cos_angle, -sin_angle, float(x_shift)],
            [sin_angle, cos_angle, float(y_shift)]
        ], dtype=np.float32)

        return transform

    def _find_best_shift(
        self,
        ref_proj: np.ndarray,
        scan_proj: np.ndarray,
        debug_mode: bool = False
    ) -> Tuple[int, float]:
        """
        Находит лучшее смещение и максимальную корреляцию с улучшенной логикой.
        """
        # Приводим к одинаковой длине
        max_len = max(len(ref_proj), len(scan_proj))
        ref_padded = np.pad(ref_proj, (0, max_len - len(ref_proj)), 'constant')
        scan_padded = np.pad(
            scan_proj, (0, max_len - len(scan_proj)), 'constant')

        # Используем кросс-корреляцию
        correlation = np.correlate(scan_padded, ref_padded, mode='full')

        # Нормализуем корреляцию
        if np.max(correlation) > 0:
            correlation = correlation / np.max(correlation)

        # Находим лучший сдвиг (центрируем)
        best_shift = int(np.argmax(correlation) - len(ref_padded) + 1)
        max_corr = float(np.max(correlation))

        if debug_mode and self.debug_fmt:
            self.debug_fmt.debug("Результат корреляции:", indent=4)
            self.debug_fmt.debug(
                f"Максимум корреляции: {max_corr:.6f}", indent=5)
            self.debug_fmt.debug(f"Лучший сдвиг: {best_shift}", indent=5)

        return best_shift, max_corr
