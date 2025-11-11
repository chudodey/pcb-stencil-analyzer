# processing/strategies/roi_extraction/__init__.py
"""
Пакет стратегий выделения области интереса (ROI)
"""

from .base_roi_extraction import BaseROIExtractionStrategy
from .convex_hull_strategy import ConvexHullROIStrategy
from .bounding_box_strategy import BoundingBoxROIStrategy
from .projection_strategy import ProjectionROIStrategy

__all__ = [
    'BaseROIExtractionStrategy',
    'ConvexHullROIStrategy',
    'BoundingBoxROIStrategy', 
    'ProjectionROIStrategy'
]