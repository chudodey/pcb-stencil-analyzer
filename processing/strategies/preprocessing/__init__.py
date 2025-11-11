# processing/strategies/preprocessing_strategies/__init__.py
"""
Пакет стратегий предобработки изображений.

Каждая стратегия реализована в отдельном файле для лучшей организации кода.
"""
from .base_preprocessing import BasePreprocessingStrategy
from .gaussian_blur_strategy import GaussianBlurPreprocessingStrategy
from .clahe_bilateral_strategy import CLAHEBilateralPreprocessingStrategy
from .background_subtraction_strategy import BackgroundSubtractionPreprocessingStrategy
from .median_blur_strategy import MedianBlurPreprocessingStrategy
from .unsharp_mask_strategy import UnsharpMaskPreprocessingStrategy
from .gamma_correction_strategy import GammaCorrectionPreprocessingStrategy

__all__ = [
    'BasePreprocessingStrategy',
    'GaussianBlurPreprocessingStrategy',
    'CLAHEBilateralPreprocessingStrategy',
    'BackgroundSubtractionPreprocessingStrategy',
    'MedianBlurPreprocessingStrategy',
    'UnsharpMaskPreprocessingStrategy',
    'GammaCorrectionPreprocessingStrategy',
]
