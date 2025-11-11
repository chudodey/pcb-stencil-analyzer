"""
matplotlib_saver.py

Модуль визуализации этапов обработки данных с использованием Matplotlib.
Предназначен для генерации профессиональных сравнительных графиков стратегий и финальных отчетов
в рамках пайплайна обработки изображений.

🧩 Архитектура:
- Класс MatplotlibVisualizer инкапсулирует логику построения и сохранения визуализаций.
- Поддерживает асинхронные вызовы через run_in_executor для совместимости с event loop.
- Использует безопасный бэкенд 'Agg' для рендеринга вне главного потока.

📦 Зависимости:
- numpy, matplotlib
- domain.data_models.PipelineStage
- infrastructure.{ConfigService, FileManager, LoggingService, DebugFormatter}
- strategies.evaluation.EvaluationResult

📌 Особенности:
- Инкрементальное сохранение визуализаций по этапам обработки.
- Автоматическое извлечение изображений из результатов стратегий.
- Форматирование метрик (SSIM, PSNR, ROI и др.) в заголовках графиков.
- Финальный отчет включает оригинал, эталон, выравнивание и анализ различий.

⚠️ Важно:
- Для корректной работы в асинхронной среде используется matplotlib.use('Agg').
- Все визуализации сохраняются в PNG с высоким DPI (300) и отключенным GUI.
"""

import asyncio
import logging
import time
from typing import Any, List, Optional, Tuple

import matplotlib
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from domain.data_models import PipelineStage
from infrastructure import ConfigService, FileManager
from infrastructure.debug_formatter import DebugFormatter
from infrastructure.logging_service import LoggingService

from .strategies.evaluation import EvaluationResult

# Используем безопасный бэкенд для асинхронной визуализации
matplotlib.use('Agg')

# Отключаем логирование поиска шрифтов
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Установим стиль графиков
plt.style.use('seaborn-v0_8-whitegrid')


class MatplotlibVisualizer:
    """
    Класс для создания и сохранения профессиональных визуализаций всех этапов обработки данных.
    Поддерживает инкрементальное сохранение и асинхронные операции.
    """

    # Оптимальные настройки DPI для разных случаев
    DPI_SETTINGS = {
        'high': 300,    # Для финальных отчетов
        'medium': 150,  # Для промежуточных этапов
        'low': 100      # Для быстрого предпросмотра
    }

    def __init__(self, config_service: ConfigService, file_manager: FileManager):
        """
        Инициализация визуализатора.

        Args:
            config_service: Сервис конфигурации.
            file_manager: Сервис управления файлами.
        """
        self.config_service = config_service
        self.file_manager = file_manager
        self.logger = LoggingService.get_logger(__name__)
        self.debug_formatter = DebugFormatter(
            debug_mode=config_service.get('debug_mode', False),
            module_name=__name__
        )
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Оптимизированные настройки matplotlib
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 150
        plt.rcParams['figure.max_open_warning'] = 0

        # Предварительная инициализация часто используемых объектов
        self._precompute_common_elements()

    def _precompute_common_elements(self) -> None:
        """Предварительное вычисление часто используемых элементов."""
        self._color_cache = {
            'match': np.array([0, 255, 0], dtype=np.uint8),
            'diff': np.array([255, 0, 0], dtype=np.uint8)
        }

    async def save_stage_comparison(
        self,
        order_number: str,
        stage: PipelineStage,
        evaluations: List[EvaluationResult],
        original_image: Optional[np.ndarray] = None,
        dpi_quality: str = 'medium'
    ) -> str:
        """
        Асинхронно создает и сохраняет сравнение стратегий для этапа обработки.

        Args:
            order_number: Номер заказа.
            stage: Этап обработки данных.
            evaluations: Список результатов оценок стратегий.
            original_image: Оригинальное изображение для сравнения.
            dpi_quality: Качество DPI ('low', 'medium', 'high').

        Returns:
            Путь к сохраненному файлу.
        """
        start_time = time.time()

        result = await asyncio.get_event_loop().run_in_executor(
            None, self._save_stage_comparison_sync,
            order_number, stage, evaluations, original_image, dpi_quality
        )

        elapsed = time.time() - start_time
        self.debug_formatter.debug(
            f"Stage comparison generated in {elapsed:.3f}s")
        return result

    def _save_stage_comparison_sync(
        self,
        order_number: str,
        stage: PipelineStage,
        evaluations: List[EvaluationResult],
        original_image: Optional[np.ndarray] = None,
        dpi_quality: str = 'medium'
    ) -> str:
        """
        Синхронная реализация сохранения сравнения стратегий для этапа.

        Args:
            order_number: Номер заказа.
            stage: Этап обработки данных.
            evaluations: Список результатов оценок стратегий.
            original_image: Оригинальное изображение для сравнения.
            dpi_quality: Качество DPI ('low', 'medium', 'high').

        Returns:
            Путь к сохраненному файлу.
        """
        if not evaluations:
            self.debug_formatter.warn(
                f"Нет данных для визуализации этапа '{stage.value}'.")
            return ""

        valid_evaluations = self._fast_filter_evaluations(evaluations)

        if not valid_evaluations:
            self.debug_formatter.warn(
                f"Не найдено изображений для визуализации в этапе '{stage.value}'.")
            return ""

        try:
            # Используем кэшированные настройки DPI
            dpi = self.DPI_SETTINGS.get(dpi_quality, 150)

            fig = self._create_optimized_stage_figure(
                stage, valid_evaluations, original_image)
            workspace = self.file_manager.create_order_workspace(order_number)
            stage_name = stage.value.lower()
            output_path = workspace / \
                f"{order_number}_{stage_name}_comparison.png"

            # Оптимизированное сохранение
            self._optimized_savefig(fig, str(output_path), dpi=dpi)

            self.debug_formatter.success(
                f"Сохранена визуализация для этапа '{stage.value}' в {output_path.name}")
            return str(output_path)

        except Exception as e:
            self.debug_formatter.error(
                f"Ошибка при создании визуализации для этапа '{stage.value}': {e}")
            return ""
        finally:
            plt.close('all')

    def _fast_filter_evaluations(self, evaluations: List[EvaluationResult]) -> List[Tuple[EvaluationResult, np.ndarray]]:
        """
        Оптимизированная фильтрация результатов.

        Args:
            evaluations: Список результатов оценок стратегий.

        Returns:
            Список кортежей (EvaluationResult, image).
        """
        valid_evaluations = []
        for eval_result in evaluations:
            image = self._fast_extract_image(eval_result)
            if image is not None:
                valid_evaluations.append((eval_result, image))
        return valid_evaluations

    def _fast_extract_image(self, eval_result: EvaluationResult) -> Optional[np.ndarray]:
        """
        Быстрое извлечение изображения с приоритетом по вероятности.

        Args:
            eval_result: Результат оценки стратегии.

        Returns:
            Изображение или None, если изображение не найдено.
        """
        result_data = eval_result.strategy_result.result_data

        if isinstance(result_data, np.ndarray):
            return result_data

        if isinstance(result_data, dict):
            # Проверяем наиболее вероятные ключи в порядке частоты использования
            for key in ['result_image', 'processed_image', 'aligned_image', 'binary_image', 'roi_image', 'image']:
                image = result_data.get(key)
                if isinstance(image, np.ndarray):
                    return image

        self.debug_formatter.debug(
            f"No image found in EvaluationResult for strategy {eval_result.strategy_name}")
        return None

    def _create_optimized_stage_figure(
        self,
        stage: PipelineStage,
        evaluations: List[Tuple[EvaluationResult, np.ndarray]],
        original_image: Optional[np.ndarray] = None
    ) -> Figure:
        """
        Оптимизированное создание фигуры.

        Args:
            stage: Этап обработки данных.
            evaluations: Список кортежей (EvaluationResult, image).
            original_image: Оригинальное изображение для сравнения.

        Returns:
            Объект Figure.
        """
        n_strategies = len(evaluations)

        # Оптимальный размер фигуры
        fig_width = max(4 * n_strategies, 12)
        fig_height = 8 if original_image is not None else 6

        # Низкий DPI для быстрого рендеринга
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)

        # Оптимизированная компоновка
        if original_image is not None:
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.1)
            ax_original = fig.add_subplot(gs[0])
            self._fast_plot_image(
                ax_original, original_image, "Original Image")
            ax_strategies = fig.add_subplot(gs[1])
        else:
            ax_strategies = fig.add_subplot(111)

        # Быстрое отображение стратегий
        self._plot_strategies_fast(fig, ax_strategies, evaluations)

        plt.suptitle(f"{stage.value.upper()} Stage - Strategy Comparison",
                     fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        return fig

    def _plot_strategies_fast(
        self,
        fig: Figure,
        parent_ax: Axes,
        evaluations: List[Tuple[EvaluationResult, np.ndarray]]
    ) -> None:
        """
        Быстрое отображение стратегий.

        Args:
            fig: Объект Figure.
            parent_ax: Родительская ось.
            evaluations: Список кортежей (EvaluationResult, image).
        """
        n_strategies = len(evaluations)
        strategy_gs = gridspec.GridSpecFromSubplotSpec(
            1, n_strategies, subplot_spec=parent_ax.get_subplotspec()
        )

        # Сортировка только если действительно необходимо
        if n_strategies > 1:
            evaluations.sort(key=lambda x: x[0].strategy_name)

        for i, (eval_result, image) in enumerate(evaluations):
            ax = fig.add_subplot(strategy_gs[0, i])
            self._fast_plot_evaluation(ax, eval_result, image)

        parent_ax.axis('off')

    def _fast_plot_evaluation(self, ax: Axes, eval_result: EvaluationResult, image: np.ndarray) -> None:
        """
        Быстрое отображение результата оценки.

        Args:
            ax: Основание для графика.
            eval_result: Результат оценки стратегии.
            image: Изображение для отображения.
        """
        # Определяем colormap один раз
        cmap = 'gray' if self._is_grayscale(image) else None
        ax.imshow(image, cmap=cmap)

        # Оптимизированное форматирование заголовка
        title = self._fast_format_title(eval_result)
        ax.set_title(title, fontsize=10, pad=10)
        ax.axis('off')

    def _is_grayscale(self, image: np.ndarray) -> bool:
        """
        Быстрая проверка на grayscale.

        Args:
            image: Изображение для проверки.

        Returns:
            True если изображение в градациях серого.
        """
        return len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1)

    def _fast_format_title(self, eval_result: EvaluationResult) -> str:
        """
        Оптимизированное форматирование заголовка.

        Args:
            eval_result: Результат оценки стратегии.

        Returns:
            Строка с названием стратегии и метриками.
        """
        strategy_name = eval_result.strategy_name
        metrics = eval_result.strategy_result.metrics or {}

        # Быстрое извлечение основных метрик
        lines = []

        # Время обработки
        if 'processing_time' in metrics:
            lines.append(f"Time: {metrics['processing_time']:.3f}s")

        # Качественные метрики
        quality_parts = []
        for key, fmt in [('ssim', 'SSIM: {:.3f}'), ('correlation', 'Corr: {:.3f}'), ('psnr', 'PSNR: {:.1f}')]:
            if key in metrics:
                quality_parts.append(fmt.format(metrics[key]))
        if quality_parts:
            lines.append(" | ".join(quality_parts))

        # ROI метрики
        if 'roi_metrics' in metrics:
            roi_parts = []
            roi_data = metrics['roi_metrics']
            if 'boundary_match' in roi_data:
                roi_parts.append(f"Match: {roi_data['boundary_match']:.1%}")
            if 'detection_confidence' in roi_data:
                roi_parts.append(
                    f"Conf: {roi_data['detection_confidence']:.1%}")
            if roi_parts:
                lines.append(" | ".join(roi_parts))

        metrics_str = "\n".join(lines)
        return f"{strategy_name}\n{metrics_str}" if metrics_str else strategy_name

    def _fast_plot_image(self, ax: Axes, image: np.ndarray, title: str) -> None:
        """
        Быстрое отображение изображения.

        Args:
            ax: Основание для графика.
            image: Изображение для отображения.
            title: Заголовок для изображения.
        """
        if self._is_grayscale(image):
            ax.imshow(image, cmap='gray')
        else:
            ax.imshow(image)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    def _optimized_savefig(self, fig: Figure, path: str, dpi: int = 150) -> None:
        """
        Оптимизированное сохранение фигуры.

        Args:
            fig: Объект Figure для сохранения.
            path: Путь для сохранения.
            dpi: Разрешение DPI.
        """
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            optimize=True,           # Оптимизация PNG
            pil_kwargs={'compress_level': 6}  # Компрессия для PNG
        )

    async def save_final_comparison(
        self,
        processed_scan: Any,
        order_number: str,
        dpi_quality: str = 'high'
    ) -> str:
        """
        Асинхронно создает и сохраняет финальный отчет с результатами выравнивания и анализа.

        Args:
            processed_scan: Обработанное сканирование.
            order_number: Номер заказа.
            dpi_quality: Качество DPI ('low', 'medium', 'high').

        Returns:
            Путь к сохраненному файлу отчета.
        """
        start_time = time.time()

        result = await asyncio.get_event_loop().run_in_executor(
            None, self._save_final_comparison_sync, processed_scan, order_number, dpi_quality
        )

        elapsed = time.time() - start_time
        self.debug_formatter.debug(
            f"Final comparison generated in {elapsed:.3f}s")
        return result

    def _save_final_comparison_sync(
        self,
        processed_scan: Any,
        order_number: str,
        dpi_quality: str = 'high'
    ) -> str:
        """
        Синхронная реализация сохранения финального отчета с результатами выравнивания и анализа.

        Args:
            processed_scan: Обработанное сканирование.
            order_number: Номер заказа.
            dpi_quality: Качество DPI ('low', 'medium', 'high').

        Returns:
            Путь к сохраненному файлу отчета.
        """
        try:
            dpi = self.DPI_SETTINGS.get(dpi_quality, 300)
            fig = self._create_optimized_final_figure(processed_scan)
            workspace = self.file_manager.create_order_workspace(order_number)
            output_path = workspace / f"{order_number}_final_report.png"

            self._optimized_savefig(fig, str(output_path), dpi=dpi)

            self.debug_formatter.success(
                f"Сохранен финальный отчет: {output_path.name}")
            return str(output_path)

        except Exception as e:
            self.debug_formatter.error(
                f"Ошибка при создании финального отчета: {e}")
            return ""
        finally:
            plt.close('all')

    def _create_optimized_final_figure(self, processed_scan: Any) -> Figure:
        """
        Оптимизированное создание финального отчета.

        Args:
            processed_scan: Обработанное сканирование.

        Returns:
            Объект Figure для отчета.
        """
        fig = plt.figure(figsize=(15, 10), dpi=100)
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

        # Быстрое отображение всех компонентов
        components = [
            (gs[0, 0], 'original_scan', "Original Scan"),
            (gs[0, 1], 'reference_image', "Reference Gerber"),
            (gs[1, 0], 'aligned_scan', "Aligned Scan")
        ]

        for spec, attr, title in components:
            if hasattr(processed_scan, attr) and getattr(processed_scan, attr) is not None:
                ax = fig.add_subplot(spec)
                self._fast_plot_image(ax, getattr(processed_scan, attr), title)

        # Оптимизированный анализ различий
        ax_diff = fig.add_subplot(gs[1, 1])
        self._fast_difference_analysis(ax_diff, processed_scan)

        plt.suptitle("Final Analysis Report - Alignment Results",
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def _fast_difference_analysis(self, ax: Axes, processed_scan: Any) -> None:
        """
        Оптимизированный анализ различий.

        Args:
            ax: Основание для графика.
            processed_scan: Обработанное сканирование.
        """
        try:
            if (hasattr(processed_scan, 'aligned_scan') and processed_scan.aligned_scan is not None and
                    hasattr(processed_scan, 'reference_image') and processed_scan.reference_image is not None):

                aligned = processed_scan.aligned_scan
                reference = processed_scan.reference_image

                # Быстрая нормализация размеров
                h, w = reference.shape[:2]
                if aligned.shape != reference.shape:
                    aligned = aligned[:h,
                                      :w] if aligned.shape[0] >= h and aligned.shape[1] >= w else aligned

                # Быстрая конвертация в grayscale
                if len(aligned.shape) == 3:
                    aligned = np.mean(aligned, axis=2)
                if len(reference.shape) == 3:
                    reference = np.mean(reference, axis=2)

                # Векторизованное вычисление различий
                diff_mask = np.abs(aligned.astype(
                    np.float32) - reference.astype(np.float32)) > 0.1

                # Быстрое создание визуализации
                diff_visualization = np.zeros((h, w, 3), dtype=np.uint8)
                # Зеленый
                diff_visualization[~diff_mask] = self._color_cache['match']
                # Красный
                diff_visualization[diff_mask] = self._color_cache['diff']

                ax.imshow(diff_visualization)
                ax.set_title("Difference Analysis\n(Green: Match, Red: Difference)",
                             fontsize=12, fontweight='bold')

                # Быстрый расчет процента совпадения
                match_percentage = 1.0 - (np.sum(diff_mask) / diff_mask.size)
                ax.text(0.02, 0.98, f"Match: {match_percentage:.1%}", transform=ax.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            else:
                ax.text(0.5, 0.5, "Difference analysis\nnot available",
                        ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title("Difference Analysis")

        except Exception as e:
            self.debug_formatter.warn(
                f"Could not create difference visualization: {e}")
            ax.text(0.5, 0.5, "Error in difference analysis",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Difference Analysis")

    def cleanup(self) -> None:
        """Очистка ресурсов."""
        plt.close('all')
