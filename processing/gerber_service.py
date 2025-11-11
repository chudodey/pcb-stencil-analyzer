# processing/gerber_service.py
"""
Сервис для Gerber с единообразным выводом.

РЕФАКТОРИНГ: Все отладочные выводы через DebugFormatter.
"""

from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from domain.data_models import BoardShape, StencilReference
from infrastructure import ConfigService
from infrastructure.debug_formatter import DebugFormatter
from infrastructure.logging_service import LoggingService

from .gerber_processor import GerberProcessor, GerberRasterizer
from .metrics_calculator import MetricsCalculator


class GerberService:
    """Инкапсулирует парсинг, растеризацию и анализ Gerber."""

    def __init__(self, config_service: ConfigService):
        self.config_service = config_service
        self.logger = LoggingService.get_logger(__name__)
        self.debug_mode = config_service.debug_mode

        # Инициализируем форматтер
        self.fmt = DebugFormatter(config_service.debug_mode, __name__)

        # Инициализация процессора
        self._processor = GerberProcessor()

    def analyze_gerber(self,
                       order_number: str,
                       gerber_path: Path,
                       gerber_content: str) -> StencilReference:
        """Анализирует Gerber-файл и создает эталон."""
        with self.fmt.timed_section("АНАЛИЗ GERBER-ФАЙЛА"):
            try:
                # 1. Парсинг
                gerber_result = self._processor.parse(gerber_content)
                self._log_gerber_parsing_debug(gerber_result)

                # 2. Валидация
                if not gerber_result['contours']:
                    raise ValueError("Gerber-файл не содержит контуров")

                # 3. Создание эталона
                stencil_reference = self._create_stencil_reference(
                    order_number, gerber_path, gerber_result
                )

                # 4. Краткий отчет (без дублирования метрик)
                if self.debug_mode:
                    self._log_stencil_reference_summary(stencil_reference)

                self.fmt.success(
                    f"Gerber-файл успешно проанализирован: {len(gerber_result['contours'])} контуров")
                return stencil_reference

            except Exception as e:
                error_msg = f"Ошибка анализа Gerber-файла: {e}"
                self.fmt.error(error_msg)
                raise ValueError(error_msg) from e

    def rasterize_gerber(self, stencil_ref: StencilReference, dpi: int) -> None:
        """Создает растровое изображение из Gerber контуров."""
        with self.fmt.timed_section("РАСТЕРИЗАЦИЯ GERBER"):
            gerber_image_exists = (
                hasattr(stencil_ref, 'gerber_image') and
                stencil_ref.gerber_image is not None and
                isinstance(stencil_ref.gerber_image, np.ndarray) and
                stencil_ref.gerber_image.size > 0
            )

            if not gerber_image_exists:
                self.fmt.debug(
                    "Gerber image отсутствует, выполняется растеризация")

                contours = stencil_ref.contours or []
                bounds = stencil_ref.gerber_bounds_mm or (
                    0.0, 0.0, 100.0, 100.0)

                rasterizer = GerberRasterizer(contours=contours, bounds=bounds)
                gerber_image = rasterizer.render(
                    dpi=dpi,
                    margin_mm=self.config_service.gerber_margin_mm
                )

                if gerber_image is None:
                    raise RuntimeError("Растеризатор Gerber вернул None")

                stencil_ref.gerber_image = gerber_image

            gerber_image_exists_after = (
                hasattr(stencil_ref, 'gerber_image') and
                stencil_ref.gerber_image is not None and
                isinstance(stencil_ref.gerber_image, np.ndarray) and
                stencil_ref.gerber_image.size > 0
            )

            if not gerber_image_exists_after:
                raise RuntimeError("Не удалось создать Gerber изображение")

            # Краткая информация о растеризации
            if self.debug_mode and hasattr(stencil_ref, 'gerber_image'):
                img = stencil_ref.gerber_image
                self.fmt.debug(
                    f"Растеризация завершена: {img.shape[1]}x{img.shape[0]} пикселей")

    def _create_stencil_reference(
        self,
        order_number: str,
        gerber_path: Path,
        gerber_result: Dict[str, Any]
    ) -> StencilReference:
        """Создает эталонный объект StencilReference."""
        with self.fmt.timed_section("СОЗДАНИЕ STENCIL REFERENCE"):
            # Расчет метрик (теперь вывод происходит внутри MetricsCalculator)
            metrics_calculator = MetricsCalculator(
                gerber_result['contours'],
                scale_factor=1.0,
                config_service=self.config_service  # Передаем для debug вывода
            )
            aperture_metrics = metrics_calculator.calculate_aperture_metrics()
            spatial_metrics = metrics_calculator.calculate_spatial_metrics()

            # Определение размеров и формы
            metrics = gerber_result['metrics']
            width = metrics.get('board_width_mm', 0.0)
            height = metrics.get('board_height_mm', 0.0)
            board_shape = self._determine_board_shape(width, height)

            stencil_ref = StencilReference(
                order_number=order_number,
                gerber_filename=gerber_path.name,
                gerber_path=gerber_path,
                stencil_size_mm=(
                    width + 2 * self.config_service.gerber_margin_mm,
                    height + 2 * self.config_service.gerber_margin_mm,
                ),
                board_shape=board_shape,
                aperture_metrics=aperture_metrics,
                spatial_metrics=spatial_metrics,
                contours=gerber_result['contours'],
                gerber_bounds_mm=gerber_result['bounds_mm']
            )

            self.fmt.debug(
                f"Создан StencilReference: {width:.1f}x{height:.1f}мм, {board_shape.value}")
            return stencil_ref

    def _determine_board_shape(self, width: float, height: float) -> BoardShape:
        """Определяет форму платы по размерам."""
        if width == 0 or height == 0:
            return BoardShape.SQUARE

        ratio = width / height

        if ratio > 1.2:
            return BoardShape.HORIZONTAL
        elif ratio < 0.8:
            return BoardShape.VERTICAL
        else:
            return BoardShape.SQUARE

    # ========================================================================
    # УПРОЩЕННЫЕ МЕТОДЫ ЛОГИРОВАНИЯ (без дублирования метрик)
    # ========================================================================

    def _log_gerber_parsing_debug(self, gerber_result: Dict[str, Any]) -> None:
        """Логирует базовую информацию о парсинге Gerber."""
        if not self.debug_mode:
            return

        # Только основная информация
        parsing_info = {
            'Контуры': f"{len(gerber_result['contours'])} шт",
            'Тип контуров': type(gerber_result['contours']).__name__
        }
        self.fmt.metrics_table("Базовая информация",
                               parsing_info, indent=1)

        # Только ключевые метрики из Gerber
        if 'metrics' in gerber_result:
            metrics = gerber_result['metrics']
            key_metrics = {}
            for key in ['board_width_mm', 'board_height_mm', 'contour_count']:
                if key in metrics:
                    key_metrics[key] = f"{metrics[key]:.2f}" if isinstance(
                        metrics[key], float) else str(metrics[key])

            if key_metrics:
                self.fmt.metrics_table(
                    "Ключевые метрики", key_metrics, indent=1)

    def _log_stencil_reference_summary(self, stencil_ref: StencilReference) -> None:
        """Краткая сводка StencilReference (без детальных метрик)."""
        if not self.debug_mode:
            return

        self.fmt.subheader("STENCIL REFERENCE SUMMARY", width=50, indent=0)

        # Только основная информация
        main_info = {
            'Order number': stencil_ref.order_number,
            'Gerber file': stencil_ref.gerber_filename,
            'Board shape': stencil_ref.board_shape.value,
            'Stencil size': f"{stencil_ref.stencil_size_mm[0]:.1f}×{stencil_ref.stencil_size_mm[1]:.1f} мм",
            'Total apertures': getattr(stencil_ref.aperture_metrics, 'count', 'N/A') if hasattr(stencil_ref, 'aperture_metrics') else 'N/A'
        }
        self.fmt.box("ОСНОВНАЯ ИНФОРМАЦИЯ", main_info, width=48)

        # Статус наличия данных
        self.fmt.debug("Статус данных:")
        checks = {
            'Контуры': hasattr(stencil_ref, 'contours') and stencil_ref.contours,
            'Gerber изображение': hasattr(stencil_ref, 'gerber_image') and stencil_ref.gerber_image is not None,
            'Метрики апертур': hasattr(stencil_ref, 'aperture_metrics') and stencil_ref.aperture_metrics,
            'Пространственные метрики': hasattr(stencil_ref, 'spatial_metrics') and stencil_ref.spatial_metrics
        }

        for name, check in checks.items():
            # status = self.fmt.SYMBOLS['success'] if check else self.fmt.SYMBOLS['error']
            status = self.fmt._SYMBOLS['success'] if check else self.fmt._SYMBOLS['error']
            self.fmt.debug(f"  {status} {name}", indent=1)

        self.fmt.success("StencilReference создан успешно")
