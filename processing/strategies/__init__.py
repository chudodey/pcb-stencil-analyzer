# processing/strategies/__init__.py
"""
Модуль стратегий обработки изображений.

Экспортирует фабричные функции для регистрации стратегий.
"""

# Импорт модулей, содержащих классы стратегий,
# чтобы strategy_registry мог их обнаружить
from . import preprocessing
from . import binarization
from . import roi_extraction
from . import alignment

# Импорт и экспорт публичных функций из нового реестра
from .strategy_registry import (
    register_all_strategies,  # ДОБАВИТЬ
    get_available_strategies
)

__all__ = [
    'register_all_strategies',  # ДОБАВИТЬ
    'get_available_strategies',
    # Оставляем модули для возможного прямого импорта
    'preprocessing',
    'binarization',
    'roi_extraction',
    'alignment'
]
