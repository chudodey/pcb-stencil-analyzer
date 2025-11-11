# processing/processing_engine.py
"""
Оптимизированный движок обработки изображений - Оркестратор.

Управляет "плоским" 8-этапным конвейером, делегируя выполнение
специализированным сервисам (GerberService, ScanService, StageRunner, ResultService).
"""

import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, List

import numpy as np

from domain.data_models import (PipelineStage, ProcessedScan,
                                ScanImage, StencilReference, EvaluationResult, PipelineResult)
from infrastructure import ConfigService, FileManager
from infrastructure.debug_formatter import DebugFormatter

# ИМПОРТЫ СТРАТЕГИЙ И ФАБРИКИ
from .strategies import strategy_registry
from .strategies.base_strategies import ProcessingStrategy

# ИМПОРТЫ НОВЫХ СЕРВИСОВ-ИСПОЛНИТЕЛЕЙ
from .gerber_service import GerberService
from .matplotlib_saver import MatplotlibVisualizer
from .result_service import ResultService
from .scan_service import ScanService
from .stage_runner import StageRunner
from .strategies.image_analyzer import ImageAnalyzer


class ProcessingEngine:
    """
    Оркестратор "плоского" конвейера обработки.
    """

    # Кэш для избежания повторной инициализации стратегий
    _strategies_cache: Dict[str, bool] = {}

    def __init__(
        self,
        config_service: ConfigService,
        file_manager: FileManager,
        progress_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Инициализирует движок и всех его "Исполнителей".
        """
        self.config_service = config_service
        self.file_manager = file_manager
        self.progress_callback = progress_callback or (lambda msg: None)
        self.debug_mode = config_service.debug_mode
        self.debug_fmt = DebugFormatter(config_service.debug_mode, __name__)

        # Инициализация Исполнителей (DI)
        self.gerber_service = GerberService(config_service)
        self.scan_service = ScanService(file_manager, config_service)
        self.image_analyzer = ImageAnalyzer(config_service.debug_mode)
        self.stage_runner = StageRunner(config_service)
        self.result_service = ResultService(config_service, file_manager)

        # Ленивая инициализация
        self._visualizer = None

        # Хранилище стратегий
        self.strategies: Dict[PipelineStage, List[ProcessingStrategy]] = {
            stage: [] for stage in PipelineStage
        }
        self._initialize_strategies()

    def _initialize_strategies(self):
        """Загружает и кэширует стратегии с помощью фабрики."""
        pipeline_key = f"all_strategies_{self.debug_mode}"

        if pipeline_key not in self._strategies_cache:
            self.debug_fmt.info("Инициализация всех стратегий")

            config = {
                'debug_mode': self.debug_mode}
            strategy_registry.register_all_strategies(self, config)

            self._strategies_cache[pipeline_key] = True
            self.debug_fmt.success("Все стратегии инициализированы")
        else:
            self.debug_fmt.debug(
                "Стратегии уже инициализированы (используется кэш)")

    def register_strategy(self, stage: PipelineStage, strategy: ProcessingStrategy) -> None:
        """Метод, вызываемый фабрикой для регистрации стратегий."""
        self.strategies[stage].append(strategy)

    # ==================== INCREMENTAL VISUALIZATION CALLBACK ====================

    def _on_stage_completed(self, stage: PipelineStage, evaluations: List[EvaluationResult], context: Dict[str, Any]):
        """
        Колбэк, вызываемый из StageRunner.
        Запускает сохранение визуализации.
        """
        try:
            order_number = context.get('order_number')
            if not order_number:
                return

            def visualization_task():
                try:
                    path = self.visualizer._save_stage_comparison_sync(
                        order_number=order_number,
                        stage=stage,
                        evaluations=evaluations,
                        original_image=None
                    )
                    if path:
                        self.debug_fmt.debug(
                            f"Инкрементальная визуализация сохранена для этапа {stage.value}")
                except Exception as e:
                    self.debug_fmt.warn(
                        f"Ошибка при инкрементальном сохранении этапа {stage.value}: {e}")

            if self.debug_mode:
                visualization_task()
            else:
                thread = threading.Thread(
                    target=visualization_task, daemon=True)
                thread.start()

        except Exception as e:
            self.debug_fmt.error(
                f"Критическая ошибка в колбэке _on_stage_completed: {e}")

    # ==================== PUBLIC INTERFACE ====================

    def analyze_gerber(self, order_number: str, gerber_path: Path) -> StencilReference:
        """
        Анализирует Gerber-файл. (Делегирование)
        """
        self._report_progress("Анализ Gerber-файла")
        # TODO сомнительное место, нужно ли отправлять callback обратно в OrderManager
        self.debug_fmt.info(f"Анализ Gerber: {gerber_path.name}")

        gerber_content = self.file_manager.read_text(gerber_path)

        # ДЕЛЕГИРОВАНИЕ
        stencil_ref = self.gerber_service.analyze_gerber(
            order_number, gerber_path, gerber_content
        )

        self.debug_fmt.success("Gerber-файл успешно проанализирован")
        return stencil_ref

    def process_scan(self, scan_path: Path, stencil_ref: StencilReference) -> ProcessedScan:
        """
        Обрабатывает скан изображения в "плоском" 8-этапном конвейере.
        """
        self.debug_fmt.header("НАЧАЛО ПРОЦЕССИНГА СКАНА")
        start_time = time.time()

        # Хранилища результатов
        stage_results: Dict[PipelineStage, EvaluationResult] = {}
        stage_evaluations: Dict[PipelineStage, List[EvaluationResult]] = {}
        intermediate_images: Dict[str, Any] = {}

        try:
            # === Шаг 1: Получение скана (Делегирование) ===
            self._report_progress("Фаза 1: Подготовка данных скана")
            scan_info, scan_image = self.scan_service.load_scan(
                scan_path, stencil_ref.order_number)

            # Контекст конвейера
            context = self._create_pipeline_context(scan_info, stencil_ref)
            current_data = scan_image  # Данные, передаваемые по цепочке

            # === Шаг 2: Растеризация Gerber (Делегирование) ===
            self._report_progress("Фаза 2: Растеризация Gerber")
            self.gerber_service.rasterize_gerber(stencil_ref, scan_info.dpi)

            # Обновляем контекст
            context['reference_image'] = stencil_ref.gerber_image
            context['expected_size_px'] = (
                stencil_ref.gerber_image.shape[1],
                stencil_ref.gerber_image.shape[0]
            )

            # === Шаг 3: Анализ изображений (Делегирование) ===
            self._report_progress("Фаза 3: Анализ изображений")
            context['global_analysis'] = self.image_analyzer.analyze_images(
                scan_image, stencil_ref.gerber_image)

            # === Шаг 4: Предобработка скана (Делегирование) ===
            self._report_progress("Фаза 4: Предобработка")
            current_data, best_res, evals = self.stage_runner.execute_stage(
                PipelineStage.PREPROCESSING, self.strategies[PipelineStage.PREPROCESSING],
                current_data, context, self._on_stage_completed
            )
            stage_results[PipelineStage.PREPROCESSING] = best_res
            stage_evaluations[PipelineStage.PREPROCESSING] = evals
            if 'lean_metrics' in best_res.strategy_result.metrics:
                context['lean_metrics'] = best_res.strategy_result.metrics['lean_metrics']

            # === Шаг 5: Бинаризация скана (Делегирование) ===
            self._report_progress("Фаза 5: Бинаризация")
            current_data, best_res, evals = self.stage_runner.execute_stage(
                PipelineStage.BINARIZATION, self.strategies[PipelineStage.BINARIZATION],
                current_data, context, self._on_stage_completed
            )
            stage_results[PipelineStage.BINARIZATION] = best_res
            stage_evaluations[PipelineStage.BINARIZATION] = evals

            # === Шаг 6: Выделение области интереса (ROI) (Делегирование) ===
            self._report_progress("Фаза 6: Выделение ROI")
            current_data, best_res, evals = self.stage_runner.execute_stage(
                PipelineStage.ROI_EXTRACTION, self.strategies[PipelineStage.ROI_EXTRACTION],
                current_data, context, self._on_stage_completed
            )
            stage_results[PipelineStage.ROI_EXTRACTION] = best_res
            stage_evaluations[PipelineStage.ROI_EXTRACTION] = evals

            # === Шаг 7: Совмещение (Делегирование) ===
            self._report_progress("Фаза 7: Совмещение")
            alignment_input = self._prepare_alignment_input(
                current_data, stencil_ref.gerber_image
            )
            current_data, best_res, evals = self.stage_runner.execute_stage(
                PipelineStage.ALIGNMENT, self.strategies[PipelineStage.ALIGNMENT],
                alignment_input, context, self._on_stage_completed
            )
            stage_results[PipelineStage.ALIGNMENT] = best_res
            stage_evaluations[PipelineStage.ALIGNMENT] = evals

            # === Сборка финального DTO (PipelineResult) ===
            self._report_progress("Фаза 8: Сборка результата")
            global_valid = self.stage_runner.evaluator.validate_global_result(
                current_data, context)

            pipeline_result = PipelineResult(
                success=True,
                final_result=current_data,
                stage_results=stage_results,
                total_processing_time=time.time() - start_time,
                strategy_combination={
                    stage: res.strategy_name for stage, res in stage_results.items()
                },
                global_validation_passed=global_valid,
                stage_evaluations=stage_evaluations,
                intermediate_images=intermediate_images
            )

            # === Финальная сборка и сохранение (Делегирование) ===
            processed_scan = self.result_service.assemble_and_save(
                scan_info, stencil_ref, scan_image, pipeline_result
            )

            processing_time = time.time() - start_time
            self.debug_fmt.success(
                f"Процессинг завершен успешно за {processing_time:.2f}с")

            # Статистика обработки
            if self.debug_mode:
                stats = {
                    "Общее время": f"{processing_time:.2f}с",
                    "Успешные этапы": len(stage_results),
                    "Использованные стратегии": len(pipeline_result.strategy_combination)
                }
                self.debug_fmt.metrics_table("Статистика процессинга", stats)

            return processed_scan

        except Exception as e:
            self._handle_processing_error(e, scan_path, stencil_ref)
            raise

    # ==================== ХЕЛПЕРЫ ОРКЕСТРАТОРА ====================

    def _create_pipeline_context(self, scan_info: ScanImage, stencil_ref: StencilReference) -> Dict[str, Any]:
        """
        Создает контекст для выполнения конвейера.
        """
        return {
            'order_number': scan_info.order_number,
            'dpi': scan_info.dpi,
            'debug_mode': self.debug_mode,
            'expected_contour_count': stencil_ref.aperture_metrics.count,
            'min_contour_area': self.file_manager.mm2_to_pixels(
                stencil_ref.aperture_metrics.min_area *
                self.config_service.min_contour_coefficient,
                scan_info.dpi
            ),
            'max_contour_area': self.file_manager.mm2_to_pixels(
                stencil_ref.aperture_metrics.max_area,
                scan_info.dpi
            ),
        }

    def _prepare_alignment_input(self,
                                 input_data: Any,
                                 reference_image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Подготавливает входные данные для этапа alignment.
        """
        scan_image_for_alignment = input_data
        if isinstance(input_data, dict) and 'image' in input_data:
            scan_image_for_alignment = input_data['image']

        return {
            'reference': reference_image,
            'scan': scan_image_for_alignment
        }

    def _report_progress(self, message: str) -> None:
        """Отправляет сообщение о прогрессе."""
        self.progress_callback(message)

    def _handle_processing_error(self, error: Exception, scan_path: Path, stencil_ref: StencilReference) -> None:
        """Обрабатывает ошибки в процессе обработки."""
        error_msg = f"Критическая ошибка при обработке скана: {error}"
        self.debug_fmt.error(error_msg)
        if self.debug_mode:
            self.debug_fmt.error("Полный traceback:")
            self.debug_fmt.error(traceback.format_exc())

    # ==================== PROPERTIES ====================

    @property
    def visualizer(self) -> MatplotlibVisualizer:
        """Ленивая инициализация визуализатора."""
        if self._visualizer is None:
            self._visualizer = MatplotlibVisualizer(
                self.config_service, self.file_manager)
            self.debug_fmt.debug("Визуализатор инициализирован")
        return self._visualizer
