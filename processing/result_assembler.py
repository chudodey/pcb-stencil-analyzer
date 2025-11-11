# processing/result_assembler.py
"""
Оптимизированный сборщик результатов.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging

# pylint: disable=no-member
import cv2

import numpy as np

from domain.data_models import (
    ProcessedScan, ScanImage, StencilReference, ScanAnalysisResult,
    AlignmentResult, ComparisonResult, ApertureMetrics, SpatialMetrics,
    AlignmentMetrics, AlignmentStatus, PipelineStage, PipelineResult,
    PreprocessingMetrics, ContourDetectionMetrics, ROIMetrics
)
from infrastructure import ConfigService
from .metrics_calculator import MetricsCalculator

logger = logging.getLogger(__name__)


class ResultAssembler:
    """Оптимизированный сборщик результатов."""

    def __init__(self, config_service: ConfigService):
        self.config_service = config_service

    def create_processed_scan(self, pipeline_result: 'PipelineResult',
                              scan_info: ScanImage, stencil_ref: StencilReference) -> ProcessedScan:
        """Быстрая сборка ProcessedScan."""
        alignment = self._extract_alignment_result(pipeline_result)
        processed_image = self._extract_processed_image(pipeline_result)

        contours = self._extract_contours_fast(processed_image)
        scale_factor = 25.4 / scan_info.dpi

        aperture_metrics, spatial_metrics = self._calculate_metrics_fast(
            contours, scale_factor)
        aligned_image = self._extract_aligned_image(
            pipeline_result.final_result)

        # 🔧 Дописанный функционал: Defaults/расчёт для missing метрик
        pre_metrics = self._get_preprocessing_metrics(pipeline_result)
        cont_metrics = self._get_contour_metrics(pipeline_result, contours)
        roi_metrics = self._get_roi_metrics(pipeline_result)

        scan_analysis = ScanAnalysisResult(
            preprocessing_metrics=pre_metrics,
            contour_metrics=cont_metrics,
            roi_metrics=roi_metrics,
            aperture_metrics=aperture_metrics,
            spatial_metrics=spatial_metrics,
            contours=contours,
            processed_image=processed_image,
            aligned_image=aligned_image,
        )

        comparison = self._create_comparison_fast(
            alignment, stencil_ref, aperture_metrics)

        return ProcessedScan(
            scan_info=scan_info,
            scan_analysis=scan_analysis,
            alignment=alignment,
            comparison=comparison,
            stencil_reference=stencil_ref
        )

    def _get_preprocessing_metrics(self, pipeline_result: PipelineResult) -> PreprocessingMetrics:
        """Расчёт/извлечение метрик предобработки (дописанный функционал)."""
        stage = pipeline_result.stage_results.get(PipelineStage.PREPROCESSING)
        if stage and stage.strategy_result.metrics:
            return PreprocessingMetrics(**stage.strategy_result.metrics.get('preprocessing', {}))
        return PreprocessingMetrics()  # Default

    def _get_contour_metrics(self, pipeline_result: PipelineResult, contours: List) -> ContourDetectionMetrics:
        """Расчёт метрик контуров."""
        stage = pipeline_result.stage_results.get(
            PipelineStage.BINARIZATION)  # Или ROI
        if stage and stage.strategy_result.metrics:
            metrics = stage.strategy_result.metrics.get('contour', {})
            return ContourDetectionMetrics(
                total_detected=len(contours),
                **metrics
            )
        return ContourDetectionMetrics(total_detected=len(contours))

    def _get_roi_metrics(self, pipeline_result: PipelineResult) -> ROIMetrics:
        """Расчёт метрик ROI."""
        stage = pipeline_result.stage_results.get(PipelineStage.ROI_EXTRACTION)
        if stage and stage.strategy_result.metrics:
            return ROIMetrics(**stage.strategy_result.metrics.get('roi', {}))
        return ROIMetrics()

    def _extract_alignment_result(self, pipeline_result: 'PipelineResult') -> AlignmentResult:
        """Быстрое извлечение результата выравнивания."""
        alignment_stage = pipeline_result.stage_results.get(
            PipelineStage.ALIGNMENT)

        if not alignment_stage or not alignment_stage.strategy_result:
            return self._create_default_alignment_result()

        metrics_data = alignment_stage.strategy_result.metrics
        result_data = alignment_stage.strategy_result.result_data

        alignment_metrics = self._create_alignment_metrics_object(
            metrics_data, result_data)

        correlation = alignment_metrics.correlation if alignment_metrics else 0.0
        is_aligned = correlation >= self.config_service.medium_correlation_threshold

        return AlignmentResult(
            is_aligned=is_aligned,
            alignment_score=correlation,
            alignment_metrics=alignment_metrics,
            alignment_status=AlignmentStatus.SUCCESS if is_aligned else AlignmentStatus.FAILED
        )

    def _create_alignment_metrics_object(self, metrics_data: Dict[str, Any], result_data: Any) -> Optional[AlignmentMetrics]:
        """Создание AlignmentMetrics с защитой от unknown kwargs."""
        try:
            alignment_metrics_dict = metrics_data.get('alignment_metrics', {})

            known_fields = {
                'correlation', 'rotation_angle', 'shift_x_px', 'shift_y_px', 'shift_x_mm', 'shift_y_mm',
                'scale_factor', 'homography_matrix', 'iou', 'dice_coefficient', 'intersection_pixels',
                'union_pixels', 'mean_contour_distance', 'ref_contours_count', 'aligned_contours_count',
                'ref_nonzero_pixels', 'aligned_nonzero_pixels', 'phase_shift_x', 'phase_shift_y', 'phase_response'
            }
            filtered_dict = {
                k: v for k, v in alignment_metrics_dict.items() if k in known_fields}

            if isinstance(result_data, dict):
                filtered_dict['correlation'] = result_data.get(
                    'correlation', filtered_dict.get('correlation', 0.0))
                filtered_dict['rotation_angle'] = result_data.get(
                    'rotation_angle', filtered_dict.get('rotation_angle', 0.0))
                filtered_dict['shift_x_px'] = result_data.get(
                    'offset_x_px', filtered_dict.get('shift_x_px', 0.0))
                filtered_dict['shift_y_px'] = result_data.get(
                    'offset_y_px', filtered_dict.get('shift_y_px', 0.0))
                filtered_dict['scale_factor'] = result_data.get(
                    'scale_factor', filtered_dict.get('scale_factor', 1.0))

            return AlignmentMetrics(**filtered_dict)

        except (KeyError, AttributeError, TypeError) as e:
            logger.warning("Ошибка создания AlignmentMetrics: %s", e)
            logger.debug("Переданные метрики: %s", list(
                metrics_data.keys()) if metrics_data else "None")
            return None

    def _create_default_alignment_result(self) -> AlignmentResult:
        """Создание результата выравнивания по умолчанию."""
        return AlignmentResult(
            is_aligned=False,
            alignment_score=0.0,
            alignment_metrics=None,
            alignment_status=AlignmentStatus.FAILED
        )

    def _extract_processed_image(self, pipeline_result: 'PipelineResult') -> Optional[np.ndarray]:
        """Быстрое извлечение обработанного изображения."""
        binarization_stage = pipeline_result.stage_results.get(
            PipelineStage.BINARIZATION)

        if not binarization_stage or not binarization_stage.strategy_result:
            return None

        result_data = binarization_stage.strategy_result.result_data

        if isinstance(result_data, dict):
            return result_data.get('binary_image')
        elif isinstance(result_data, np.ndarray):
            return result_data

        return None

    def _extract_contours_fast(self, image: Optional[np.ndarray]) -> List:
        """Быстрое извлечение контуров."""
        if image is None:
            return []

        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return list(contours)

    def _calculate_metrics_fast(self, contours: List, scale_factor: float) -> Tuple[ApertureMetrics, SpatialMetrics]:
        """Быстрый расчет метрик."""
        if not contours:
            return (
                ApertureMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                SpatialMetrics(0.0, 0.0, 0.0, {})
            )

        calculator = MetricsCalculator(
            contours, scale_factor, self.config_service)
        return (
            calculator.calculate_aperture_metrics(),
            calculator.calculate_spatial_metrics()
        )

    def _extract_aligned_image(self, final_data: Any) -> Optional[np.ndarray]:
        """Быстрое извлечение выровненного изображения."""
        if final_data is None:
            return None

        if isinstance(final_data, dict):
            return final_data.get('aligned_image')
        elif isinstance(final_data, np.ndarray):
            return final_data

        return None

    def _create_comparison_fast(self, alignment: AlignmentResult,
                                stencil_ref: StencilReference,
                                scan_metrics: ApertureMetrics) -> ComparisonResult:
        """Быстрое создание результата сравнения."""
        expected = stencil_ref.aperture_metrics.count
        actual = scan_metrics.count

        # 🔧 Дописанный функционал: Расчёт для missing полей
        match_percentage = alignment.alignment_score * \
            100 if alignment.alignment_metrics else 0.0
        # Рассчитайте реальные, если есть данные (e.g., из alignment_metrics.mismatch_areas)
        mismatch_areas = []

        return ComparisonResult(
            match_percentage=match_percentage,
            mismatch_areas=mismatch_areas,
            missing_apertures=max(0, expected - actual),
            excess_apertures=max(0, actual - expected),
            quality_score=alignment.alignment_score,
            aperture_count_diff=actual - expected,
            area_deviation_px=0.0,
            clearance_deviation_px=0.0
        )
