# domain/__init__.py
"""
Пакет доменных моделей данных
"""

from .data_models import (
    Operator,
    ProcessingSession,
    OrderResult,
    ProcessedScan,
    StencilReference,
    ScanImage,
    AlignmentResult,
    ComparisonResult,
    ScanAnalysisResult,
    BoardShape,
    AlignmentStatus,
    PipelineStage,
    ApertureMetrics,
    SpatialMetrics,
    PreprocessingMetrics,
    ContourDetectionMetrics,
    ROIMetrics,
    AlignmentMetrics,
    calculate_alignment_status
)

__all__ = [
    'Operator',
    'ProcessingSession',
    'OrderResult',
    'ProcessedScan',
    'StencilReference',
    'ScanImage',
    'AlignmentResult',
    'ComparisonResult',
    'ScanAnalysisResult',
    'BoardShape',
    'AlignmentStatus',
    'PipelineStage',
    'ApertureMetrics',
    'SpatialMetrics',
    'PreprocessingMetrics',
    'ContourDetectionMetrics',
    'ROIMetrics',
    'AlignmentMetrics',
    'calculate_alignment_status'
]
