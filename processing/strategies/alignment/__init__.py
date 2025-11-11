# processing/strategies/alignment/__init__.py
"""
Стратегии совмещения изображений
"""
# from .base_alignment import AlignmentStrategy
from .contour_based import ContourBasedAlignmentStrategy
from .reference_transform import ReferenceTransformAlignmentStrategy
from .projection_based import ProjectionBasedAlignmentStrategy
from .global_correlation import GlobalCorrelationAlignmentStrategy
from .moment_scale import MomentScaleAlignmentStrategy
from .alignment_utils import AlignmentUtils

__all__ = [
    # 'AlignmentStrategy',
    'ContourBasedAlignmentStrategy',
    'ReferenceTransformAlignmentStrategy',
    'ProjectionBasedAlignmentStrategy',
    'GlobalCorrelationAlignmentStrategy',
    'MomentScaleAlignmentStrategy',
    'AlignmentUtils'
]
