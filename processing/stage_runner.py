# processing/stage_runner.py
"""
Исполнитель этапов конвейера.

StageRunner (Менеджер этапа) отвечает за выполнение ОДНОГО этапа обработки:
1. Объявляет запуск этапа в Фазе Процесса
2. Выполняет оценку всех стратегий через StrategyEvaluator
3. В Фазе Сводки выводит таблицу результатов и лучшую стратегию
4. Вызывает колбэк для инкрементальной визуализации

СООТВЕТСТВИЕ ИНСТРУКЦИИ:
- Уровень 2 в иерархии логирования
- Говорит разработчику о структуре этапа
- Объявляет [запуск] и выводит [СВОДКУ] с таблицей
- Не логирует детали выполнения стратегий (это уровень 4)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from domain.data_models import (
    EvaluationResult, PipelineStage, StrategyResult)
from infrastructure import ConfigService
from infrastructure.debug_formatter import DebugFormatter
from infrastructure.logging_service import LoggingService

from .strategies.base_strategies import ProcessingStrategy
from .strategies.evaluation import StrategyEvaluator


class StageRunner:
    """
    Менеджер выполнения одного этапа конвейера.

    Отвечает за:
    - Организацию выполнения этапа (Фаза Процесса)
    - Отображение структурированной сводки (Фаза Сводки) 
    - Выбор лучшей стратегии и передачу данных следующему этапу
    """

    def __init__(self, config_service: ConfigService):
        """
        Инициализация менеджера этапа.

        Args:
            config_service: Сервис конфигурации для получения настроек пайплайна
        """
        self.config = config_service.get_pipeline_config()
        self.evaluator = StrategyEvaluator(self.config)
        self.logger = LoggingService.get_logger(__name__)
        self.debug_formatter = DebugFormatter(
            config_service.debug_mode, __name__)

    def execute_stage(self,
                      stage: PipelineStage,
                      strategies: List[ProcessingStrategy],
                      input_data: Any,
                      context: Dict[str, Any],
                      callback: Optional[Callable] = None
                      ) -> Tuple[Any, EvaluationResult, List[EvaluationResult]]:
        """
        Выполняет один этап конвейера с четким разделением на фазы.

        АРХИТЕКТУРА ВЫПОЛНЕНИЯ:
        1. ФАЗА ПРОЦЕССА: Объявление запуска этапа → Выполнение оценки стратегий
        2. ФАЗА СВОДКИ: Выбор лучшей → Колбэк → Таблица → Лучшая стратегия → Успех

        Args:
            stage: Этап пайплайна (PREPROCESSING, BINARIZATION, etc.)
            strategies: Список стратегий для оценки на данном этапе
            input_data: Входные данные для обработки
            context: Контекст выполнения (debug_mode, order_number, etc.)
            callback: Колбэк для инкрементальной визуализации (опционально)

        Returns:
            Tuple[result_data, best_result, evaluations]:
            - result_data: Данные для передачи следующему этапу
            - best_result: Лучшая оцененная стратегия этапа
            - evaluations: Все результаты оценки для анализа

        Raises:
            RuntimeError: Если не найдена успешная стратегия для этапа
        """
        debug_mode = context.get('debug_mode', False)

        # =============================================================
        # ФАЗА 1: ПРОЦЕСС
        # Объявление запуска этапа и выполнение оценки стратегий
        # =============================================================

        if debug_mode:
            self.debug_formatter.section(
                f"Этап: {stage.value}", phase="запуск")

        if not strategies:
            self.logger.warning(f"Нет стратегий для этапа {stage.value}")
            return input_data, EvaluationResult.empty(), []

        # Выполнение оценки всех стратегий (StrategyEvaluator и ProcessingStrategy
        # логируют детали выполнения на своих уровнях)
        evaluations = self.evaluator.evaluate_strategies(
            strategies, input_data, context, stage.value
        )

        # =============================================================
        # ФАЗА 2: СВОДКА
        # Все операции сводки выполняются последовательно после завершения процесса
        # =============================================================

        best_result = None
        if debug_mode:
            self.debug_formatter.subheader(
                f"Сводка этапа: {stage.value}", indent=1)

            # 1. Выбор лучшей стратегии
            best_result = self.evaluator.select_best_strategy(
                evaluations, stage.value, context
            )

            # 2. Вызов колбэка для визуализации
            if callback:
                try:
                    callback(stage=stage, evaluations=evaluations,
                             context=context)
                except Exception as e:
                    self.logger.warning(
                        f"Ошибка в колбэке визуализации для этапа {stage.value}: {e}")

            # 3. Отображение таблицы результатов
            if evaluations:
                self.debug_formatter.strategy_table_start(
                    stage.value, len(strategies), indent=1
                )
                self._display_strategy_results(evaluations, stage.value)
                self.debug_formatter.strategy_table_end(indent=1)
            else:
                self.debug_formatter.warn(
                    f"Нет результатов оценки для {stage.value}", indent=1)

            # 4. Анонс лучшей стратегии
            if best_result:
                self._display_best_strategy(best_result, stage.value)
            else:
                self.debug_formatter.error(
                    f"Не удалось выбрать лучшую стратегию для {stage.value}", indent=1)

            # 5. Финальное сообщение об успешном завершении этапа
            self.debug_formatter.success(
                f"Этап {stage.value} завершен успешно", indent=1
            )

        # Для production-режима (без debug_mode) выполняем минимально необходимые операции
        if best_result is None:
            best_result = self.evaluator.select_best_strategy(
                evaluations, stage.value, context
            )

        if callback and not debug_mode:
            try:
                callback(stage=stage, evaluations=evaluations, context=context)
            except Exception as e:
                self.logger.warning(f"Ошибка в колбэке: {e}")

        # Проверка на сбой этапа
        if not best_result:
            self.logger.error(f"No successful strategy for {stage.value}")
            raise RuntimeError(
                f"Сбой на этапе {stage.value}: не найдена успешная стратегия")

        # Извлечение данных для следующего этапа
        result_data = self._extract_result_data(best_result.strategy_result)

        return result_data, best_result, evaluations

    def _display_strategy_results(self,
                                  evaluations: List[EvaluationResult],
                                  stage_name: str):
        """
        Отображает результаты всех стратегий в табличном формате.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - StageRunner выводит таблицу сравнения стратегий
        - Каждая строка содержит имя, оценку, время и статус стратегии

        Args:
            evaluations: Список результатов оценки стратегий
            stage_name: Название этапа для отладки
        """
        for eval_result in evaluations:
            strategy_name = eval_result.strategy_name
            quality_score = eval_result.quality_score
            processing_time = eval_result.evaluation_time
            passed = eval_result.strategy_result.success

            # Отображение строки таблицы через DebugFormatter
            self.debug_formatter.strategy_table_row(
                name=strategy_name,
                score=quality_score,
                time_s=processing_time,
                passed=passed,
                indent=1
            )

    def _display_best_strategy(self,
                               best_result: EvaluationResult,
                               stage_name: str):
        """
        Отображает информацию о выбранной лучшей стратегии.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - StageRunner анонсирует лучшую стратегию с оценкой и временем
        - НЕ логирует метрики стратегии (это делает сама стратегия)

        Args:
            best_result: Лучший результат оценки этапа
            stage_name: Название этапа для контекста
        """
        if best_result and best_result.strategy_result.success:
            self.debug_formatter.debug(
                f"Лучшая стратегия: {best_result.strategy_name} "
                f"(оценка: {best_result.quality_score:.3f}, "
                f"время: {best_result.evaluation_time:.2f}с)",
                indent=2
            )
            # StageRunner НЕ логирует метрики стратегии - это делает сама стратегия
            # в своем методе _log_success через metrics_table()
        elif best_result:
            self.debug_formatter.warn(
                f"Выбрана стратегия с ошибками: {best_result.strategy_name} "
                f"(оценка: {best_result.quality_score:.3f})",
                indent=2
            )

    @staticmethod
    def _extract_result_data(strategy_result: StrategyResult) -> Any:
        """
        Извлекает данные из результата стратегии для передачи следующему этапу.

        Args:
            strategy_result: Результат выполнения стратегии

        Returns:
            Any: Данные для следующего этапа обработки
        """
        if not strategy_result:
            return None

        return strategy_result.result_data


# ========================================================================
# ДОПОЛНИТЕЛЬНЫЕ СЕРВИСЫ (если требуются для других частей системы)
# ========================================================================

class StageValidationService:
    """Сервис валидации результатов этапа (вынесен для чистоты StageRunner)."""

    def __init__(self, debug_formatter: DebugFormatter):
        self.debug_formatter = debug_formatter

    def validate_stage_result(self,
                              stage: PipelineStage,
                              result_data: Any,
                              best_result: EvaluationResult) -> bool:
        """
        Валидация результата этапа.

        Args:
            stage: Этап пайплайна
            result_data: Данные результата
            best_result: Лучший результат оценки

        Returns:
            bool: True если результат валиден
        """
        if result_data is None:
            self.debug_formatter.warn(
                f"Результат этапа {stage.value} содержит None данные")
            return False

        if not best_result.strategy_result.success:
            self.debug_formatter.warn(
                f"Лучшая стратегия этапа {stage.value} завершилась с ошибкой")
            return False

        # Дополнительная валидация в зависимости от типа этапа
        if stage == PipelineStage.BINARIZATION:
            if isinstance(result_data, np.ndarray):
                # Проверка что изображение бинаризовано
                unique_vals = np.unique(result_data)
                if len(unique_vals) > 2:
                    self.debug_formatter.warn(
                        f"Бинаризация этапа {stage.value} содержит более 2 значений")
                    return False

        return True


class StageReportingService:
    """Сервис генерации отчетов по этапам (вынесен для чистоты StageRunner)."""

    def __init__(self, debug_formatter: DebugFormatter):
        self.debug_formatter = debug_formatter

    def log_stage_summary(self,
                          stage: PipelineStage,
                          evaluations: List[EvaluationResult],
                          best_result: EvaluationResult,
                          indent: int = 0):
        """
        Логирует компактную сводку по этапу для финального отчета.

        Args:
            stage: Этап пайплайна
            evaluations: Все результаты оценки
            best_result: Лучший результат
            indent: Уровень отступа для вывода
        """
        successful_count = sum(
            1 for e in evaluations if e.strategy_result.success)
        total_count = len(evaluations)

        self.debug_formatter.info(
            f"Этап {stage.value}: {successful_count}/{total_count} "
            f"стратегий успешно",
            indent=indent
        )

        if best_result and best_result.strategy_result.success:
            self.debug_formatter.debug(
                f"Выбрана: {best_result.strategy_name} "
                f"(оценка: {best_result.quality_score:.3f})",
                indent=indent + 1
            )
