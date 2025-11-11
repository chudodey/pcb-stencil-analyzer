# processing/__init__.py
"""
Processing Layer - модуль обработки изображений
"""

from .processing_engine import ProcessingEngine
from .gerber_processor import GerberProcessor, GerberRasterizer
from .result_assembler import ResultAssembler
from .artifact_saver import ArtifactSaver
from .metrics_calculator import MetricsCalculator
from .matplotlib_saver import MatplotlibVisualizer

__all__ = [
    'ProcessingEngine',
    'GerberProcessor',
    'GerberRasterizer',
    'ResultAssembler',
    'ArtifactSaver',
    'MetricsCalculator',
    'MatplotlibVisualizer'
]
