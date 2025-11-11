# processing/strategies/binarization/__init__.py
"""
Пакет стратегий бинаризации.
"""

from .base_binarization import BaseBinarizationStrategy
from .iterative_adaptive_strategy import IterativeAdaptiveStrategy
from .adaptive_thresholding_strategy import AdaptiveThresholdingBinarizationStrategy
from .adaptive_otsu_hybrid_strategy import AdaptiveOtsuHybridBinarizationStrategy
from .local_adaptive_strategy import LocalAdaptiveStrategy
from .otsu_strategy import OtsuBinarizationStrategy
from .simple_threshold_strategy import SimpleThresholdBinarizationStrategy

# Автоматический импорт всех стратегий для регистрации
__all__ = [
    'BaseBinarizationStrategy',
    'IterativeAdaptiveStrategy',
    'AdaptiveThresholdingBinarizationStrategy',
    'AdaptiveOtsuHybridBinarizationStrategy',
    'LocalAdaptiveStrategy',
    'OtsuBinarizationStrategy',
    'SimpleThresholdBinarizationStrategy'
]
