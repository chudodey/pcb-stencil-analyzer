# processing/metrics_calculator.py
"""
Модуль для вычисления геометрических и пространственных метрик контуров.
Инкапсулирует всю математическую логику, связанную с анализом апертур.

РЕФАКТОРИНГ: Единообразный отладочный вывод через DebugFormatter
"""

from typing import List, Dict, Optional
import numpy as np

# pylint: disable=no-member
import cv2
from scipy.spatial.distance import pdist

from domain.data_models import ApertureMetrics, SpatialMetrics
from infrastructure.logging_service import LoggingService
from infrastructure.debug_formatter import DebugFormatter
from infrastructure import ConfigService


class MetricsCalculator:
    """
    Вычисляет детальные метрики апертур и пространственные метрики
    на основе списка контуров. Поддерживает оба формата:
    - List[np.ndarray] 
    - List[List[Point]]
    """

    def __init__(self, contours: List, scale_factor: float = 1.0,
                 config_service: Optional[ConfigService] = None):
        """
        Инициализация калькулятора.

        Args:
            contours: Список контуров (np.ndarray или List[Point]).
            scale_factor: Коэффициент для перевода пикселей в мм.
            config_service: Сервис конфигурации для отладочного вывода.
        """
        self.logger = LoggingService.get_logger(__name__)

        # Инициализация DebugFormatter
        self.debug_fmt = DebugFormatter(
            config_service.debug_mode, __name__) if config_service else None

        self.debug_mode = config_service.debug_mode if config_service else False

        with self._timed_section("ИНИЦИАЛИЗАЦИЯ METRICS CALCULATOR"):
            self.contours = self._normalize_contours(contours)
            self.scale_factor = scale_factor
            self._calculated_properties = {}

            if self.debug_mode:
                self.debug_fmt.debug(f"Контуров: {len(self.contours)}")
                self.debug_fmt.debug(f"Коэффициент масштаба: {scale_factor}")

    def _timed_section(self, title: str):
        """Контекстный менеджер для секций с таймингом."""
        if self.debug_mode and self.debug_fmt:
            return self.debug_fmt.timed_section(title)
        else:
            # Заглушка для не-debug режима
            class DummyContext:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return DummyContext()

    def _normalize_contours(self, contours: List) -> List[np.ndarray]:
        """
        Нормализует контуры в единый формат np.ndarray.

        Returns:
            Список нормализованных контуров.
        """
        if self.debug_mode:
            self.debug_fmt.section("НОРМАЛИЗАЦИЯ КОНТУРОВ", width=50)
            self.debug_fmt.debug(
                f"Исходное количество контуров: {len(contours)}")

        normalized_contours = []
        conversion_stats = {
            'numpy_arrays': 0,
            'point_objects': 0,
            'lists_tuples': 0,
            'invalid_skipped': 0,
            'too_small_skipped': 0
        }

        # УПРОЩЕНО: убрана детализация по каждому контуру
        for contour in contours:
            try:
                # Уже numpy массив
                if isinstance(contour, np.ndarray):
                    conversion_stats['numpy_arrays'] += 1
                    normalized_contours.append(contour)
                    continue

                # Конвертация из Point объектов
                if hasattr(contour, '__iter__') and len(contour) > 0:
                    first_point = contour[0]
                    if hasattr(first_point, 'x') and hasattr(first_point, 'y'):
                        conversion_stats['point_objects'] += 1
                        points_array = np.array(
                            [[p.x, p.y] for p in contour], dtype=np.int32)
                        normalized_contours.append(points_array)
                        continue

                # Конвертация из списка/кортежа
                if isinstance(contour, (list, tuple)) and len(contour) > 0:
                    conversion_stats['lists_tuples'] += 1
                    try:
                        normalized_contour = np.array(contour, dtype=np.int32)
                        if normalized_contour.ndim == 1:
                            normalized_contour = normalized_contour.reshape(
                                -1, 1, 2)
                        normalized_contours.append(normalized_contour)
                        continue
                    except (ValueError, TypeError):
                        conversion_stats['invalid_skipped'] += 1
                        continue

                # Неизвестный тип
                conversion_stats['invalid_skipped'] += 1

            except Exception:
                conversion_stats['invalid_skipped'] += 1

        # Валидация нормализованных контуров
        valid_contours = []
        for contour in normalized_contours:
            try:
                # Проверка минимального количества точек
                if len(contour) < 3:
                    conversion_stats['too_small_skipped'] += 1
                    continue

                # Проверка площади
                area = cv2.contourArea(contour)
                if area <= 0:
                    conversion_stats['too_small_skipped'] += 1
                    continue

                valid_contours.append(contour)

            except Exception:
                conversion_stats['invalid_skipped'] += 1

        # УПРОЩЕНО: выводим только итоговую статистику
        if self.debug_mode:
            stats_info = {
                # Простой показатель качества
                'composite_score': len(valid_contours) / max(len(contours), 1),
                'Исходные контуры': len(contours),
                'Валидные контуры': len(valid_contours),
                'Эффективность': f"{(len(valid_contours) / max(len(contours), 1)) * 100:.1f}%",
                'Numpy массивы': conversion_stats['numpy_arrays'],
                'Point объекты': conversion_stats['point_objects'],
                'Списки/кортежи': conversion_stats['lists_tuples'],
                'Пропущено (невалидные)': conversion_stats['invalid_skipped'],
                'Пропущено (<3 точек)': conversion_stats['too_small_skipped']
            }
            self.debug_fmt.metrics_table(
                "СТАТИСТИКА НОРМАЛИЗАЦИИ", stats_info, indent=1)

            # Дополнительная информация о причинах пропуска
            if conversion_stats['too_small_skipped'] > 0:
                self.debug_fmt.warn(
                    f"Пропущено {conversion_stats['too_small_skipped']} контуров с <3 точками",
                    indent=1
                )

        return valid_contours

    def _get_properties(self) -> Dict:
        """
        Лениво вычисляет и кеширует базовые свойства контуров,
        чтобы избежать повторных вычислений.
        """
        if self._calculated_properties:
            return self._calculated_properties

        with self._timed_section("ВЫЧИСЛЕНИЕ СВОЙСТВ КОНТУРОВ"):
            areas_px = []
            perimeters_px = []
            aspect_ratios = []
            circularities = []

            if self.debug_mode:
                self.debug_fmt.debug(
                    f"Обработка {len(self.contours)} контуров...")

            for i, contour in enumerate(self.contours):
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)

                areas_px.append(area)
                perimeters_px.append(perimeter)

                # Расчет соотношения сторон
                _, (width, height), _ = cv2.minAreaRect(contour)
                aspect_ratio = max(width, height) / min(width,
                                                        height) if min(width, height) > 0 else 1.0
                aspect_ratios.append(aspect_ratio)

                # Расчет округлости
                circularity = (4 * np.pi * area) / \
                    (perimeter ** 2) if perimeter > 0 else 0
                circularities.append(circularity)

                if self.debug_mode and i < 5:
                    self.debug_fmt.debug(
                        f"Контур {i}: площадь={area:.4f}, периметр={perimeter:.4f}, "
                        f"соотношение={aspect_ratio:.2f}, округлость={circularity:.3f}",
                        indent=1
                    )

            self._calculated_properties = {
                "areas_px": np.array(areas_px, dtype=float),
                "perimeters_px": np.array(perimeters_px, dtype=float),
                "aspect_ratios": np.array(aspect_ratios, dtype=float),
                "circularities": np.array(circularities, dtype=float),
            }

            if self.debug_mode:
                props = self._calculated_properties
                summary = {
                    'Общее количество': len(areas_px),
                    'Площадь (min)': f"{np.min(props['areas_px']):.4f}",
                    'Площадь (max)': f"{np.max(props['areas_px']):.4f}",
                    'Площадь (mean)': f"{np.mean(props['areas_px']):.4f}",
                    'Соотношение сторон (min)': f"{np.min(props['aspect_ratios']):.2f}",
                    'Соотношение сторон (max)': f"{np.max(props['aspect_ratios']):.2f}",
                    'Округлость (min)': f"{np.min(props['circularities']):.3f}",
                    'Округлость (max)': f"{np.max(props['circularities']):.3f}"
                }
                self.debug_fmt.metrics_table(
                    "СВОДКА СВОЙСТВ", summary, indent=1)

            return self._calculated_properties

    def calculate_aperture_metrics(self) -> ApertureMetrics:
        """
        Вычисляет и возвращает полностью заполненный объект ApertureMetrics.
        """
        with self._timed_section("ВЫЧИСЛЕНИЕ МЕТРИК АПЕРТУР"):
            props = self._get_properties()
            areas_sq_mm = props["areas_px"] * (self.scale_factor ** 2)

            if len(areas_sq_mm) == 0:
                if self.debug_mode:
                    self.debug_fmt.error(
                        "Нет валидных контуров для вычисления метрик")
                else:
                    self.logger.warning(
                        "Нет валидных контуров для вычисления метрик")
                return ApertureMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, [], 0)

            # Вычисляем все метрики
            count = len(self.contours)
            mean_area = float(np.mean(areas_sq_mm))
            std_area = float(np.std(areas_sq_mm))
            min_area = float(np.min(areas_sq_mm))
            max_area = float(np.max(areas_sq_mm))
            median_area = float(np.median(areas_sq_mm))
            min_circularity = float(np.min(props["circularities"]))
            mean_circularity = float(np.mean(props["circularities"]))
            aspect_ratios_list = props["aspect_ratios"].tolist()
            mean_aspect_ratio = float(np.mean(props["aspect_ratios"]))

            if self.debug_mode:
                aperture_stats = {
                    'Количество апертур': count,
                    'Площадь min': f"{min_area:.4f} мм²",
                    'Площадь max': f"{max_area:.4f} мм²",
                    'Площадь mean': f"{mean_area:.4f} мм²",
                    'Площадь median': f"{median_area:.4f} мм²",
                    'Площадь std': f"{std_area:.4f}",
                    'Округлость min': f"{min_circularity:.3f}",
                    'Округлость mean': f"{mean_circularity:.3f}",
                    'Соотношение mean': f"{mean_aspect_ratio:.2f}"
                }
                self.debug_fmt.box("МЕТРИКИ АПЕРТУР", aperture_stats, width=50)

            return ApertureMetrics(
                count=count,
                mean_area=mean_area,
                std_area=std_area,
                min_area=min_area,
                max_area=max_area,
                median_area=median_area,
                min_circularity=min_circularity,
                mean_circularity=mean_circularity,
                mean_ellipticity=0.0,  # Placeholder
                aspect_ratios=aspect_ratios_list,
                mean_aspect_ratio=mean_aspect_ratio
            )

    def calculate_spatial_metrics(self) -> SpatialMetrics:
        """
        Вычисляет и возвращает полностью заполненный объект SpatialMetrics.
        """
        with self._timed_section("ВЫЧИСЛЕНИЕ ПРОСТРАНСТВЕННЫХ МЕТРИК"):
            if len(self.contours) < 2:
                if self.debug_mode:
                    self.debug_fmt.warn(
                        "Недостаточно контуров для пространственных метрик (нужно ≥ 2)")
                else:
                    self.logger.warning(
                        "Недостаточно контуров для пространственных метрик (нужно ≥ 2)")
                return SpatialMetrics(0, 0, 0, {})

            # Вычисляем центроиды всех контуров
            centroids = []
            if self.debug_mode:
                self.debug_fmt.debug(
                    f"Вычисление центроидов для {len(self.contours)} контуров...")

            for i, contour in enumerate(self.contours):
                m = cv2.moments(contour)
                if m["m00"] != 0:
                    cx = m["m10"] / m["m00"]
                    cy = m["m01"] / m["m00"]
                    centroids.append((cx, cy))

                    if self.debug_mode and i < 3:
                        self.debug_fmt.debug(
                            f"Контур {i}: центроид ({cx:.2f}, {cy:.2f})", indent=1)

            if not centroids:
                if self.debug_mode:
                    self.debug_fmt.error("Не найдено валидных центроидов")
                else:
                    self.logger.warning("Не найдено валидных центроидов")
                return SpatialMetrics(0, 0, 0, {})

            # Рассчитываем попарные расстояния
            if self.debug_mode:
                self.debug_fmt.debug(
                    f"Вычисление попарных расстояний для {len(centroids)} центроидов...", indent=1)

            distances_px = pdist(np.array(centroids))
            distances_mm = distances_px * self.scale_factor

            min_clearance = float(np.min(distances_mm))
            mean_clearance = float(np.mean(distances_mm))
            std_clearance = float(np.std(distances_mm))
            percentiles = {
                p: float(np.percentile(distances_mm, p))
                for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
            }

            if self.debug_mode:
                spatial_stats = {
                    'Минимальный зазор': f"{min_clearance:.4f} мм",
                    'Средний зазор': f"{mean_clearance:.4f} мм",
                    'Стандартное отклонение': f"{std_clearance:.4f}",
                    'P1': f"{percentiles[1]:.4f} мм",
                    'P50 (медиана)': f"{percentiles[50]:.4f} мм",
                    'P99': f"{percentiles[99]:.4f} мм"
                }
                self.debug_fmt.box("ПРОСТРАНСТВЕННЫЕ МЕТРИКИ",
                                   spatial_stats, width=50)

            return SpatialMetrics(
                min_clearance_global=min_clearance,
                clearance_mean=mean_clearance,
                clearance_std=std_clearance,
                clearance_percentiles=percentiles
            )
