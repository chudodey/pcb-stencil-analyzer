# order/__init__.py
"""
Пакет бизнес-логики обработки заказов
"""

from .order_coordinator import OrderCoordinator
from .order_manager import OrderManager

__all__ = [
    'OrderCoordinator',
    'OrderManager'
]