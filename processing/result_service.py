# processing/result_service.py
"""
Сервис для сборки финального результата и сохранения артефактов.
Использует ResultAssembler и ArtifactSaver.
"""
import threading
from typing import Any

import numpy as np

from domain.data_models import ProcessedScan, ScanImage, StencilReference, PipelineResult
from infrastructure import ConfigService, FileManager
from infrastructure.logging_service import LoggingService

from .artifact_saver import ArtifactSaver
from .result_assembler import ResultAssembler


class ResultService:
    """Инкапсулирует финальную сборку и сохранение."""

    def __init__(self, config_service: ConfigService, file_manager: FileManager):
        self.config_service = config_service
        self.file_manager = file_manager
        self.logger = LoggingService.get_logger(__name__)

        # Инициализация "Исполнителей"
        self._result_assembler = ResultAssembler(self.config_service)
        self._artifact_saver = ArtifactSaver(
            self.config_service, self.file_manager)

    def assemble_and_save(
        self,
        scan_info: ScanImage,
        stencil_ref: StencilReference,
        scan_image: np.ndarray,
        pipeline_result: PipelineResult
    ) -> ProcessedScan:
        """
        Собирает результаты и сохраняет артефакты.
        (Перенесено из ProcessingEngine._assemble_and_save_results)
        """
        self.logger.debug("Фаза 4: Сбор и сохранение результатов")

        # Асинхронное сохранение базовых артефактов
        self._save_compatibility_artifacts_async(
            scan_info, stencil_ref, scan_image, pipeline_result
        )

        # Финальная сборка
        processed_scan = self._result_assembler.create_processed_scan(
            pipeline_result, scan_info, stencil_ref
        )

        self.logger.debug(
            "Результат обработки: aligned=%s, contours=%s",
            processed_scan.alignment.is_aligned,
            len(processed_scan.scan_analysis.contours)
        )

        return processed_scan

    def _save_compatibility_artifacts_async(
        self,
        scan_info: ScanImage,
        stencil_ref: StencilReference,
        scan_image: np.ndarray,
        pipeline_result: PipelineResult,
    ) -> None:
        """
        Асинхронно сохраняет базовые артефакты.
        (Перенесено из ProcessingEngine._save_compatibility_artifacts_async)
        """
        def save_task():
            try:
                self._artifact_saver.save_compatibility_artifacts(
                    scan_info, stencil_ref, scan_image, pipeline_result
                )
            except Exception as save_error:
                self.logger.warning(
                    "Фоновая ошибка сохранения compatibility-артефактов: %s", save_error)

        thread = threading.Thread(target=save_task, daemon=True)
        thread.start()
