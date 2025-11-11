# processing/strategies/strategy_registry.py
"""
Фабрика для обнаружения и регистрации стратегий обработки.
"""

import inspect
from typing import Any, Dict, List, Optional, Type, Protocol

# Импорт PipelineStage из правильного места
from domain.data_models import PipelineStage
from infrastructure import ConfigService
from infrastructure.debug_formatter import DebugFormatter

# Импорт всех модулей со стратегиями для автоматического обнаружения
from . import alignment, binarization, preprocessing, roi_extraction
from .base_strategies import ProcessingStrategy


class StrategyRegistrar(Protocol):
    """Протокол для объекта, который может регистрировать стратегии (например, ProcessingEngine)."""

    def register_strategy(self, stage: PipelineStage, strategy: ProcessingStrategy) -> None:
        ...


class StrategyRegistry:
    """Фабрика для обнаружения и регистрации стратегий."""

    def __init__(self, config_service: Optional[ConfigService] = None):
        self.config_service = config_service or ConfigService()
        self.debug_fmt = DebugFormatter(
            self.config_service.debug_mode, __name__)
        # Кэш для найденных стратегий
        self._strategy_cache: Dict[PipelineStage, List[str]] = {}

    def register_all_strategies(self, engine: StrategyRegistrar, config: Dict[str, Any]) -> None:
        """
        Автоматически обнаруживает и регистрирует ВСЕ стратегии в движок.
        """
        self.debug_fmt.section("Начало регистрации всех стратегий")

        self._auto_register_strategies(engine, config)

        if not self._strategy_cache:
            self.get_available_strategies()

        strategy_count = sum(len(v) for v in self._strategy_cache.values())
        self.debug_fmt.success(
            f"Всего стратегий: {strategy_count}")

        # ВЫВОД СПИСКА СТРАТЕГИЙ
        self.debug_fmt.info("Зарегистрированные стратегии:")
        for stage, strategies in self._strategy_cache.items():
            if strategies:
                self.debug_fmt.info(
                    f"  {stage.value}: {len(strategies)} стратегий")
                for strategy in strategies:
                    self.debug_fmt.info(f"    - {strategy}")

        # Детальная информация о зарегистрированных стратегиях (только в debug режиме)
        if self.config_service.debug_mode:
            self.debug_fmt.debug("Детальная информация о стратегиях:")
            for stage, strategies in self._strategy_cache.items():
                if strategies:
                    self.debug_fmt.debug(f"  Этап {stage.value}:")
                    for strategy in strategies:
                        self.debug_fmt.debug(f"    - {strategy}", indent=2)

    def _auto_register_strategies(self, engine: StrategyRegistrar, config: Dict[str, Any]):
        """Автоматически обнаруживает и регистрирует все доступные стратегии."""
        # Модули для сканирования
        strategy_modules = [
            (preprocessing, PipelineStage.PREPROCESSING),
            (binarization, PipelineStage.BINARIZATION),
            (roi_extraction, PipelineStage.ROI_EXTRACTION),
            (alignment, PipelineStage.ALIGNMENT)
        ]

        total_registered = 0
        for module, stage in strategy_modules:
            # ДИАГНОСТИКА: проверяем, что модуль загружен
            self.debug_fmt.debug(f"Сканирование модуля: {module.__name__}")

            strategies = self._discover_strategies_in_module(module, stage)
            stage_registered = 0

            # ДИАГНОСТИКА: выводим найденные классы
            all_classes = [name for name, obj in inspect.getmembers(module, inspect.isclass)
                           if obj.__module__.startswith(module.__name__)]
            self.debug_fmt.debug(
                f"Все классы в {module.__name__}: {all_classes}")

            for strategy_class in strategies:
                try:
                    # Создаем стратегию с правильными параметрами
                    strategy_instance = self._instantiate_strategy(
                        strategy_class, config)
                    engine.register_strategy(stage, strategy_instance)

                    self.debug_fmt.debug(
                        f"Зарегистрирована стратегия: {strategy_class.__name__} -> {stage.value}")
                    stage_registered += 1
                    total_registered += 1

                except Exception as e:
                    self.debug_fmt.warn(
                        f"Не удалось создать стратегию {strategy_class.__name__}: {e}")

            if stage_registered > 0:
                self.debug_fmt.debug(
                    f"Этап {stage.value}: зарегистрировано {stage_registered} стратегий")
            else:
                self.debug_fmt.warn(
                    f"Этап {stage.value}: не найдено ни одной стратегии")

        self.debug_fmt.success(
            f"Всего зарегистрировано стратегий: {total_registered}")

    def _discover_strategies_in_module(self, module, stage: PipelineStage) -> List[Type[ProcessingStrategy]]:
        """Обнаруживает стратегии в модуле с улучшенной диагностикой."""
        strategies = []

        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Проверяем, что класс из этого модуля
            if not obj.__module__.startswith(module.__name__):
                continue

            # ДИАГНОСТИКА: выводим информацию о каждом классе
            is_processing_strategy = issubclass(obj, ProcessingStrategy)
            is_not_base = obj != ProcessingStrategy and not name.startswith(
                'Base')

            self.debug_fmt.debug(
                f"  Класс {name}: "
                f"ProcessingStrategy={is_processing_strategy}, "
                f"not_base={is_not_base}"
            )

            if (is_processing_strategy and is_not_base):
                strategies.append(obj)
                self.debug_fmt.debug(f"    -> ДОБАВЛЕНА стратегия: {name}")

        self.debug_fmt.debug(
            f"Найдено стратегий в {module.__name__}: {[s.__name__ for s in strategies]}")
        return strategies

    def _instantiate_strategy(self, strategy_class: Type[ProcessingStrategy], config: Dict[str, Any], name: Optional[str] = None) -> ProcessingStrategy:
        """Создает экземпляр стратегии, обрабатывая разные конструкторы."""
        strategy_name = name or strategy_class.__name__

        try:
            sig = inspect.signature(strategy_class.__init__)
            params = list(sig.parameters.keys())

            if len(params) >= 3:  # self + name + config
                return strategy_class(strategy_name, config)
            elif len(params) >= 2:  # self + config
                return strategy_class(strategy_name, config)
            else:  # только self
                return strategy_class(strategy_name, config)
        except Exception as e:
            self.debug_fmt.warn(
                f"Ошибка при создании стратегии {strategy_name}: {e}")
            # Пробуем создать с минимальными параметрами
            try:
                return strategy_class(strategy_name)
            except:
                raise e

    def get_available_strategies(self) -> Dict[PipelineStage, List[str]]:
        """Возвращает список всех доступных стратегий по стадиям."""
        if not self._strategy_cache:
            self._build_strategy_cache()
        return self._strategy_cache.copy()

    def _build_strategy_cache(self):
        """Строит кэш всех доступных стратегий."""
        self.debug_fmt.debug("Построение кэша доступных стратегий")

        modules = [
            (preprocessing, PipelineStage.PREPROCESSING),
            (binarization, PipelineStage.BINARIZATION),
            (roi_extraction, PipelineStage.ROI_EXTRACTION),
            (alignment, PipelineStage.ALIGNMENT)
        ]

        self._strategy_cache = {stage: [] for stage in PipelineStage}
        total_strategies = 0

        for module, stage in modules:
            strategies = self._discover_strategies_in_module(module, stage)
            self._strategy_cache[stage] = [s.__name__ for s in strategies]
            total_strategies += len(strategies)

        self.debug_fmt.debug(
            f"Кэш стратегий построен: всего {total_strategies} стратегий")


# Глобальный экземпляр фабрики для удобства использования
_factory_instance = StrategyRegistry()


def register_all_strategies(engine: StrategyRegistrar, config: Dict[str, Any]) -> None:
    """
    Функция для быстрой регистрации всех стратегий.
    """
    _factory_instance.register_all_strategies(engine, config)


def get_available_strategies() -> Dict[PipelineStage, List[str]]:
    """
    Возвращает список всех доступных стратегий.
    """
    return _factory_instance.get_available_strategies()


# Экспортируемые функции
__all__ = [
    'StrategyRegistry',
    'register_all_strategies',
    'get_available_strategies'
]
