# processing/strategies/evaluation.py
"""
Оптимизированная система оценки стратегий с прагматичным подходом.

StrategyEvaluator (Оценщик) отвечает за:
- Использование готовых метрик качества из стратегий
- Легкую корректировку оценок на основе контекста выполнения
- Прагматичный выбор лучшей стратегии без перерасчета метрик

СООТВЕТСТВИЕ ИНСТРУКЦИИ:
- Уровень 3 в иерархии логирования
- Минимальное логирование внутренней кухни оценки
- Не дублирует логирование выполнения стратегий (уровень 4)
- Не выводит итоговые таблицы (уровень 2)

ФИЛОСОФИЯ:
Доверяем стратегиям расчет их собственного качества, оценщик только
корректирует готовые оценки на основе контекста выполнения.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from domain.data_models import EvaluationResult, StrategyResult
from infrastructure.config_service import ConfigService
from infrastructure.debug_formatter import DebugFormatter
from infrastructure.logging_service import LoggingService
from .base_strategies import ProcessingStrategy


class StrategyEvaluator:
    """
    Прагматичный оценщик стратегий обработки изображений.

    Основная ответственность:
    - Использование composite_score из метрик стратегий как основной оценки
    - Легкая корректировка на основе времени выполнения и контекста
    - Минимальный резервный расчет когда стратегия не предоставляет метрики
    - Эффективный выбор лучшей стратегии по скорректированным оценкам
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация прагматичного оценщика стратегий.

        Args:
            config: Конфигурация пайплайна с порогами качества и факторами коррекции
        """
        self.config = config or {}
        self.logger = LoggingService.get_logger(__name__)

        # Инициализация DebugFormatter для минимального внутреннего логирования
        config_service = ConfigService()
        self.debug_formatter = DebugFormatter(
            config_service.debug_mode, __name__)

        # Пороги качества для различных этапов
        self.quality_thresholds: Dict[str, float] = self.config.get('quality_thresholds', {
            'preprocessing': 0.5,
            'binarization': 0.4,
            'roi_extraction': 0.5,
            'alignment': 0.5
        })

        # Факторы корректировки оценок на основе контекста
        self.BOOST_FACTOR: float = self.config.get('boost_factor', 1.3)
        self.PENALTY_FACTOR: float = self.config.get('penalty_factor', 0.7)

    def evaluate_strategies(self,
                            strategies: List[ProcessingStrategy],
                            input_data: Any,
                            context: Dict[str, Any],
                            stage: str) -> List[EvaluationResult]:
        """
        Эффективная оценка списка стратегий с использованием готовых метрик.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Не логирует запуск стратегий (это делает ProcessingStrategy)
        - Минимальное логирование процесса оценки
        - Возвращает отсортированные результаты без вывода таблиц

        Args:
            strategies: Список стратегий для оценки
            input_data: Входные данные для обработки
            context: Контекст выполнения (debug_mode, global_analysis, etc.)
            stage: Название текущего этапа ('preprocessing', 'binarization', etc.)

        Returns:
            List[EvaluationResult]: Отсортированный по убыванию качества список результатов
        """
        results: List[EvaluationResult] = []
        debug_mode: bool = context.get('debug_mode', False)

        if debug_mode:
            self.debug_formatter.debug(
                f"Доступно стратегий для оценки: {len(strategies)}", indent=2)

        for strategy in strategies:
            # Проверка применимости стратегии
            if not strategy.is_applicable(input_data, context):
                if debug_mode:
                    self.debug_formatter.debug(
                        f"Пропуск {strategy.name} - не применима", indent=2)
                continue

            total_start_time = time.perf_counter()

            try:
                # Выполнение стратегии (логирование на уровне ProcessingStrategy)
                strategy_result = strategy.execute_with_logging(
                    input_data, context)

                # Прагматичный расчет качества на основе метрик стратегии
                quality_start_time = time.perf_counter()
                quality_score = self._calculate_pragmatic_quality_score(
                    strategy_result, context, stage)
                quality_eval_time = time.perf_counter() - quality_start_time

                total_loop_time = time.perf_counter() - total_start_time

                results.append(EvaluationResult(
                    strategy_name=strategy.name,
                    quality_score=quality_score,
                    strategy_result=strategy_result,
                    evaluation_time=total_loop_time
                ))

                # Минимальное логирование процесса оценки
                if debug_mode and strategy_result.success:
                    self._log_evaluation_summary(
                        strategy.name, quality_score,
                        strategy_result.processing_time, quality_eval_time
                    )

            except Exception as e:
                # Молчаливая обработка ошибок выполнения стратегии
                total_loop_time = time.perf_counter() - total_start_time
                self.logger.error("Сбой при выполнении стратегии %s: %s",
                                  strategy.name, e, exc_info=debug_mode)

                failed_result = StrategyResult(
                    strategy_name=strategy.name,
                    success=False,
                    result_data=None,
                    metrics={},
                    processing_time=total_loop_time,
                    error_message=str(e)
                )

                results.append(EvaluationResult(
                    strategy_name=strategy.name,
                    quality_score=0.0,
                    strategy_result=failed_result,
                    evaluation_time=total_loop_time
                ))

        # Корректировка оценок на основе контекста анализа
        results = self._adjust_scores_based_on_context(
            results, context, stage, debug_mode)

        # Сортировка по убыванию качества
        results.sort(key=lambda x: x.quality_score, reverse=True)

        return results

    def _calculate_pragmatic_quality_score(self,
                                           strategy_result: StrategyResult,
                                           context: Dict[str, Any],
                                           stage: str) -> float:
        """
        Прагматичная оценка качества на основе метрик стратегии.

        ПРИОРИТЕТ РАСЧЕТА:
        1. composite_score из корня метрик
        2. composite_score из contour_metrics
        3. quality_score из корня метрик
        4. Резервный расчет для конкретного этапа

        Args:
            strategy_result: Результат выполнения стратегии
            context: Контекст выполнения с анализом изображений
            stage: Название этапа для маршрутизации оценки

        Returns:
            float: Оценка качества в диапазоне [0.0, 1.0]
        """
        if not strategy_result.success:
            return 0.0

        debug_mode = context.get('debug_mode', False)

        # ПРИОРИТЕТ 1: composite_score из корня метрик
        composite_score = strategy_result.metrics.get('composite_score', 0.0)

        # ПРИОРИТЕТ 2: composite_score из contour_metrics
        if composite_score <= 0 and 'contour_metrics' in strategy_result.metrics:
            composite_score = strategy_result.metrics['contour_metrics'].get(
                'composite_score', 0.0)

        # ПРИОРИТЕТ 3: quality_score из корня метрик
        if composite_score <= 0:
            composite_score = strategy_result.metrics.get('quality_score', 0.0)

        # ПРИОРИТЕТ 4: Резервный расчет для конкретного этапа
        if composite_score <= 0:
            composite_score = self._calculate_fallback_score(
                strategy_result, context, stage)

        # Легкая коррекция на время выполнения
        time_penalty = self._calculate_time_penalty(
            strategy_result.processing_time)
        final_score = composite_score * time_penalty

        if debug_mode:
            self.debug_formatter.debug(
                f"[Оценка {strategy_result.strategy_name}]: "
                f"composite={composite_score:.3f}, time_penalty={time_penalty:.3f}, "
                f"final={final_score:.3f}",
                indent=3
            )

        return max(0.0, min(1.0, final_score))

    def _calculate_fallback_score(self,
                                  strategy_result: StrategyResult,
                                  context: Dict[str, Any],
                                  stage: str) -> float:
        """
        Резервный расчет оценки когда стратегия не предоставляет метрики.

        Args:
            strategy_result: Результат выполнения стратегии
            context: Контекст выполнения
            stage: Название этапа для выбора метода расчета

        Returns:
            float: Резервная оценка качества
        """
        debug_mode = context.get('debug_mode', False)

        if debug_mode:
            self.debug_formatter.debug(
                f"[Резервный расчет] для {strategy_result.strategy_name}", indent=3)

        if stage == 'binarization':
            return self._fallback_binarization_quality(strategy_result, context)
        elif stage == 'preprocessing':
            return self._fallback_preprocessing_quality(strategy_result, context)
        else:
            # Базовая оценка для других этапов
            return 0.5 if strategy_result.success else 0.0

    def _fallback_binarization_quality(self,
                                       strategy_result: StrategyResult,
                                       context: Dict[str, Any]) -> float:
        """
        Минимальный резервный расчет качества бинаризации.

        Args:
            strategy_result: Результат бинаризации
            context: Контекст выполнения

        Returns:
            float: Оценка качества бинаризации [0.0, 1.0]
        """
        try:
            result_data = strategy_result.result_data
            if not result_data:
                return 0.0

            contours = result_data.get('contours', [])
            binary_image = result_data.get('binary_image')

            if not contours or binary_image is None:
                return 0.0

            # Простая оценка на основе количества контуров и покрытия
            contour_count = len(contours)
            white_pixels = np.sum(binary_image == 255)
            coverage_ratio = white_pixels / binary_image.size

            # Нормализация количества контуров (для PCB 1000-5000 - хорошо)
            count_score = min(1.0, contour_count / 5000.0)

            # Качество покрытия (для PCB 0.2-0.6 - хорошо)
            if 0.2 <= coverage_ratio <= 0.6:
                coverage_score = 1.0
            elif 0.15 <= coverage_ratio <= 0.7:
                coverage_score = 0.7
            else:
                coverage_score = 0.3

            fallback_score = 0.6 * count_score + 0.4 * coverage_score

            debug_mode = context.get('debug_mode', False)
            if debug_mode:
                self.debug_formatter.debug(
                    f"[Резервная бинаризация] count={contour_count}, "
                    f"coverage={coverage_ratio:.3f}, score={fallback_score:.3f}",
                    indent=4
                )

            return fallback_score

        except Exception as e:
            debug_mode = context.get('debug_mode', False)
            if debug_mode:
                self.debug_formatter.debug(
                    f"[Ошибка резервной бинаризации]: {e}", indent=4)
            return 0.3

    def _fallback_preprocessing_quality(self,
                                        strategy_result: StrategyResult,
                                        context: Dict[str, Any]) -> float:
        """
        Минимальный резервный расчет качества предобработки.

        Args:
            strategy_result: Результат предобработки
            context: Контекст выполнения

        Returns:
            float: Оценка качества предобработки [0.0, 1.0]
        """
        try:
            result_data = strategy_result.result_data
            if not result_data or 'image' not in result_data:
                return 0.0

            processed_image = result_data['image']

            # Простая оценка контраста
            contrast_score = min(1.0, np.std(processed_image) / 80.0)

            debug_mode = context.get('debug_mode', False)
            if debug_mode:
                self.debug_formatter.debug(
                    f"[Резервная предобработка] contrast_score={contrast_score:.3f}",
                    indent=4
                )

            return contrast_score

        except Exception as e:
            debug_mode = context.get('debug_mode', False)
            if debug_mode:
                self.debug_formatter.debug(
                    f"[Ошибка резервной предобработки]: {e}", indent=4)
            return 0.3

    def _calculate_time_penalty(self, processing_time: float) -> float:
        """
        Легкий штраф за очень долгое выполнение.

        Args:
            processing_time: Время выполнения стратегии в секундах

        Returns:
            float: Множитель штрафа [0.7, 1.0]
        """
        if processing_time < 2.0:
            return 1.0
        elif processing_time < 5.0:
            return 0.95
        elif processing_time < 10.0:
            return 0.9
        elif processing_time < 30.0:
            return 0.8
        else:
            return 0.7

    def _adjust_scores_based_on_context(self,
                                        evaluation_results: List[EvaluationResult],
                                        context: Dict[str, Any],
                                        stage: str,
                                        debug_mode: bool) -> List[EvaluationResult]:
        """
        Корректировка оценок на основе контекста анализа изображений.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Логирует только внутреннюю логику корректировки
        - Помогает понять, почему оценки были изменены

        Args:
            evaluation_results: Список результатов оценки
            context: Контекст с анализом изображений
            stage: Текущий этап обработки
            debug_mode: Режим отладки

        Returns:
            List[EvaluationResult]: Список с скорректированными оценками
        """
        global_analysis = context.get('global_analysis')
        lean_metrics = context.get('lean_metrics')

        if not global_analysis:
            return evaluation_results

        # Используем lean_metrics если доступны (после предобработки)
        source_metrics = lean_metrics or global_analysis.get(
            'scan_analysis', {})
        scan_analysis = global_analysis.get('scan_analysis', {})

        if stage == 'binarization':
            is_bimodal = source_metrics.get('is_bimodal', False)
            has_uneven_lighting = scan_analysis.get(
                'brightness_distribution', {}).get('has_uneven_lighting', False)

            # Внутренний лог контекста для понимания корректировки
            if debug_mode:
                self.debug_formatter.debug(
                    f"Контекст для бинаризации: is_bimodal={is_bimodal}, "
                    f"uneven_lighting={has_uneven_lighting}",
                    indent=2
                )

            for result in evaluation_results:
                # Корректировка Otsu на основе бимодальности гистограммы
                if "Otsu" in result.strategy_name:
                    if is_bimodal:
                        old_score = result.quality_score
                        result.quality_score *= self.BOOST_FACTOR
                        if debug_mode:
                            self.debug_formatter.debug(
                                f"Повышение Otsu: {old_score:.3f} → {result.quality_score:.3f}",
                                indent=3
                            )
                    else:
                        old_score = result.quality_score
                        result.quality_score *= self.PENALTY_FACTOR
                        if debug_mode:
                            self.debug_formatter.debug(
                                f"Понижение Otsu: {old_score:.3f} → {result.quality_score:.3f}",
                                indent=3
                            )

                # Корректировка Adaptive на основе неравномерности освещения
                if "Adaptive" in result.strategy_name and has_uneven_lighting:
                    old_score = result.quality_score
                    result.quality_score *= self.BOOST_FACTOR
                    if debug_mode:
                        self.debug_formatter.debug(
                            f"Повышение Adaptive: {old_score:.3f} → {result.quality_score:.3f}",
                            indent=3
                        )

        return evaluation_results

    def _log_evaluation_summary(self,
                                strategy_name: str,
                                quality_score: float,
                                strategy_time: float,
                                quality_eval_time: float):
        """
        Минимальное логирование итогов оценки стратегии.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Только итоговая информация об оценке
        - Не дублирует детали выполнения стратегии

        Args:
            strategy_name: Название оцениваемой стратегии
            quality_score: Рассчитанная оценка качества
            strategy_time: Время выполнения стратегии
            quality_eval_time: Время расчета оценки качества
        """
        self.debug_formatter.debug(
            f"[Итог оценки {strategy_name}]: "
            f"quality={quality_score:.3f}, "
            f"strategy_time={strategy_time:.3f}s, "
            f"eval_time={quality_eval_time:.3f}s",
            indent=3
        )

    def select_best_strategy(self,
                             evaluation_results: List[EvaluationResult],
                             stage: str,
                             context: Dict[str, Any]) -> Optional[EvaluationResult]:
        """
        Прагматичный выбор лучшей стратегии.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Не логирует выбор результата (это делает StageRunner)
        - Только внутренняя логика выбора без вывода

        Args:
            evaluation_results: Отсортированный список результатов оценки
            stage: Название этапа для определения порога
            context: Контекст выполнения

        Returns:
            Optional[EvaluationResult]: Лучшая стратегия или None если нет подходящих
        """
        debug_mode = context.get('debug_mode', False)

        if not evaluation_results:
            return None

        threshold = self.quality_thresholds.get(stage, 0.4)

        if debug_mode:
            self.debug_formatter.debug(
                f"Порог качества для {stage}: {threshold:.3f}", indent=2)

        # 1. Поиск первой успешной стратегии выше порога
        for result in evaluation_results:
            if result.strategy_result.success and result.quality_score >= threshold:
                return result

        # 2. Лучшая из успешных стратегий (если ни одна не прошла порог)
        successful_results = [
            r for r in evaluation_results if r.strategy_result.success]

        if successful_results:
            return successful_results[0]  # Уже отсортированы по качеству

        return None

    def validate_global_result(self,
                               final_result: Any,
                               context: Dict[str, Any]) -> bool:
        """
        Упрощенная валидация итогового результата пайплайна.

        Args:
            final_result: Финальный результат обработки
            context: Контекст выполнения

        Returns:
            bool: True если результат содержит валидные данные
        """
        if final_result is None:
            return False

        # Проверка наличия контуров в результате
        if hasattr(final_result, 'contours'):
            return len(final_result.contours) > 0
        elif isinstance(final_result, dict) and 'contours' in final_result:
            return len(final_result['contours']) > 0

        return True
