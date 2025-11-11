# processing/strategies/base_strategies.py
"""
Базовые абстрактные классы для стратегий обработки с прагматичной оценкой качества.

ProcessingStrategy (Исполнитель) отвечает за:
- Централизованный отчет о СВОЕЙ работе
- Логирование запуска, параметров, успеха/ошибки и метрик
- Гарантированный расчет composite_score для оценки качества
- Детальное описание процесса выполнения конкретной стратегии

СООТВЕТСТВИЕ ИНСТРУКЦИИ:
- Уровень 4 в иерархии логирования
- Сердце отладочного вывода - пишет: [START], [OK], [ERROR]
- Единственный, кто логирует детали выполнения ОДНОЙ стратегии
- НЕ логирует сводные таблицы и сравнения (это уровень 2)

ФИЛОСОФИЯ:
Каждая стратегия сама знает, как оценить свое качество и гарантированно
предоставляет composite_score для использования оценщиком.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from domain.data_models import StrategyResult
from infrastructure.debug_formatter import DebugFormatter


class ProcessingStrategy(ABC):
    """
    Базовый класс для всех стратегий обработки изображений.

    Основная ответственность:
    - Централизованное логирование выполнения одной стратегии
    - Гарантированный расчет composite_score для оценки качества
    - Отчет о запуске, параметрах, результате и метриках
    - Обработка ошибок с детальным логированием
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация стратегии обработки.

        Args:
            name: Уникальное имя стратегии для идентификации
            config: Конфигурация стратегии с параметрами выполнения
        """
        self.name = name
        self.config = config or {}

        # Централизованный отладочный форматтер для стратегии
        debug_mode = config.get('debug_mode', False) if config else False
        self.debug_fmt = DebugFormatter(debug_mode, self.__class__.__module__)

    def execute_with_logging(self, input_data: Any, context: Dict[str, Any]) -> StrategyResult:
        """
        Выполнение стратегии с полным логированием процесса.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Всегда выполняется через этот метод для единообразного логирования
        - Логирует [START], параметры, тип, [OK]/[ERROR], метрики
        - Гарантирует наличие composite_score в метриках результата

        Args:
            input_data: Входные данные для обработки (изображение, контуры и т.д.)
            context: Контекст выполнения (debug_mode, order_number, анализ и т.д.)

        Returns:
            StrategyResult: Результат выполнения стратегии с метриками и временем
        """
        debug_mode = context.get('debug_mode', False)

        # 1. ЗАГОЛОВОК ЗАПУСКА СТРАТЕГИИ
        if debug_mode:
            self.debug_fmt.section(
                f"Запуск стратегии: {self.name}", phase="START")

        start_time = time.time()

        try:
            # 2. ЛОГИРОВАНИЕ ПАРАМЕТРОВ И ТИПА СТРАТЕГИИ
            if debug_mode:
                self._log_strategy_parameters(context)

            # 3. ВЫПОЛНЕНИЕ ОСНОВНОЙ ЛОГИКИ
            result = self.execute(input_data, context)
            execution_time = time.time() - start_time

            # 4. ГАРАНТИЯ НАЛИЧИЯ COMPOSITE_SCORE В МЕТРИКАХ
            if result.success and result.metrics:
                result.metrics = self._ensure_composite_score(
                    result.metrics, context)

            # 5. ЛОГИРОВАНИЕ РЕЗУЛЬТАТА (успех/ошибка)
            if result.success:
                self._log_success(result, execution_time, context)
            else:
                self._log_failure(result, execution_time, context)

            return result

        except Exception as error:
            # 6. ОБРАБОТКА ИСКЛЮЧЕНИЙ С ЛОГИРОВАНИЕМ
            execution_time = time.time() - start_time
            return self._handle_execution_error(error, execution_time, context)

    def _ensure_composite_score(self, metrics: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Гарантирует наличие composite_score в метриках стратегии.

        Args:
            metrics: Исходные метрики стратегии
            context: Контекст выполнения

        Returns:
            Dict[str, Any]: Метрики с гарантированным composite_score
        """
        debug_mode = context.get('debug_mode', False)

        # Если composite_score уже есть - ничего не делаем
        if 'composite_score' in metrics and metrics['composite_score'] > 0:
            return metrics

        # Расчет composite_score если отсутствует
        composite_score = self._calculate_composite_score(metrics, context)

        if debug_mode:
            self.debug_fmt.debug(
                f"[Гарантия composite_score]: {composite_score:.3f}", indent=2)

        # Возвращаем обновленные метрики
        return {**metrics, 'composite_score': composite_score}

    def _calculate_composite_score(self, metrics: Dict[str, Any], context: Dict[str, Any]) -> float:
        """
        Расчет composite_score на основе метрик стратегии.

        Args:
            metrics: Метрики стратегии
            context: Контекст выполнения

        Returns:
            float: Композитная оценка качества [0.0, 1.0]
        """
        debug_mode = context.get('debug_mode', False)

        # Приоритет 1: composite_score из contour_metrics
        contour_metrics = metrics.get('contour_metrics', {})
        if 'composite_score' in contour_metrics:
            score = contour_metrics['composite_score']
            if debug_mode:
                self.debug_fmt.debug(
                    f"Использован composite_score из contour_metrics: {score:.3f}", indent=3)
            return score

        # Приоритет 2: quality_score из корня метрик
        if 'quality_score' in metrics:
            score = metrics['quality_score']
            if debug_mode:
                self.debug_fmt.debug(
                    f"Использован quality_score: {score:.3f}", indent=3)
            return score

        # Приоритет 3: расчет на основе типа стратегии
        strategy_type = self.__class__.__name__.lower()
        score = self._calculate_strategy_type_score(
            metrics, strategy_type, context)

        if debug_mode:
            self.debug_fmt.debug(
                f"Рассчитан composite_score по типу стратегии: {score:.3f}", indent=3)

        return score

    def _calculate_strategy_type_score(self, metrics: Dict[str, Any],
                                       strategy_type: str,
                                       context: Dict[str, Any]) -> float:
        """
        Расчет composite_score на основе типа стратегии и доступных метрик.

        Args:
            metrics: Метрики стратегии
            strategy_type: Тип стратегии (из имени класса)
            context: Контекст выполнения

        Returns:
            float: Композитная оценка качества
        """
        debug_mode = context.get('debug_mode', False)

        if 'binarization' in strategy_type:
            # Для бинаризации используем contour_metrics
            contour_metrics = metrics.get('contour_metrics', {})
            return self._calculate_binarization_composite(contour_metrics, debug_mode)

        elif 'preprocessing' in strategy_type:
            # Для предобработки используем доступные метрики
            preprocessing_metrics = metrics.get('preprocessing_metrics', {})
            return self._calculate_preprocessing_composite(preprocessing_metrics, debug_mode)

        else:
            # Для остальных типов - среднее по числовым метрикам
            return self._calculate_default_composite(metrics, debug_mode)

    def _calculate_binarization_composite(self, contour_metrics: Dict[str, Any],
                                          debug_mode: bool) -> float:
        """
        Расчет composite_score для стратегий бинаризации.

        Args:
            contour_metrics: Метрики контуров
            debug_mode: Флаг отладки

        Returns:
            float: Оценка качества бинаризации
        """
        if not contour_metrics:
            return 0.0

        # Извлечение ключевых метрик бинаризации
        component_count = contour_metrics.get('component_count', 0)
        size_uniformity = contour_metrics.get('size_uniformity', 0.0)
        shape_consistency = contour_metrics.get('shape_consistency', 0.0)
        noise_ratio = contour_metrics.get('noise_ratio', 1.0)
        coverage_quality = contour_metrics.get('coverage_quality', 0.0)

        # Композитная оценка
        composite_score = (
            # Нормализованное количество
            0.2 * min(1.0, component_count / 5000.0) +
            0.25 * size_uniformity +
            0.25 * shape_consistency +
            0.2 * (1.0 - noise_ratio) +
            0.1 * coverage_quality
        )

        if debug_mode:
            self.debug_fmt.debug(
                f"Метрики бинаризации: count={component_count}, "
                f"uniformity={size_uniformity:.3f}, consistency={shape_consistency:.3f}, "
                f"noise={noise_ratio:.3f}, coverage={coverage_quality:.3f}",
                indent=4
            )

        return max(0.0, min(1.0, composite_score))

    def _calculate_preprocessing_composite(self, preprocessing_metrics: Dict[str, Any],
                                           debug_mode: bool) -> float:
        """
        Расчет composite_score для стратегий предобработки.

        Args:
            preprocessing_metrics: Метрики предобработки
            debug_mode: Флаг отладки

        Returns:
            float: Оценка качества предобработки
        """
        if not preprocessing_metrics:
            return 0.0

        # Извлечение ключевых метрик предобработки
        contrast = preprocessing_metrics.get('contrast_improvement', 0.0)
        noise_reduction = preprocessing_metrics.get('noise_reduction', 0.0)
        detail_preservation = preprocessing_metrics.get(
            'detail_preservation', 0.0)

        # Композитная оценка
        composite_score = (
            0.4 * contrast +
            0.3 * noise_reduction +
            0.3 * detail_preservation
        )

        if debug_mode:
            self.debug_fmt.debug(
                f"Метрики предобработки: contrast={contrast:.3f}, "
                f"noise_reduction={noise_reduction:.3f}, "
                f"detail_preservation={detail_preservation:.3f}",
                indent=4
            )

        return max(0.0, min(1.0, composite_score))

    def _calculate_default_composite(self, metrics: Dict[str, Any], debug_mode: bool) -> float:
        """
        Расчет composite_score по умолчанию для неизвестных типов стратегий.

        Args:
            metrics: Метрики стратегии
            debug_mode: Флаг отладки

        Returns:
            float: Средняя оценка по всем числовым метрикам
        """
        # Сбор всех числовых значений метрик
        numeric_values = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and 0 <= value <= 1:
                numeric_values.append(value)
            elif isinstance(value, dict):
                # Рекурсивный обход вложенных словарей
                for sub_value in value.values():
                    if isinstance(sub_value, (int, float)) and 0 <= sub_value <= 1:
                        numeric_values.append(sub_value)

        if not numeric_values:
            return 0.5  # Базовая оценка при отсутствии метрик

        # Усреднение нормализованных значений
        composite_score = sum(numeric_values) / len(numeric_values)

        if debug_mode:
            self.debug_fmt.debug(
                f"Рассчитан default composite: {composite_score:.3f} "
                f"из {len(numeric_values)} метрик",
                indent=4
            )

        return composite_score

    def _log_strategy_parameters(self, context: Dict[str, Any]) -> None:
        """
        Логирование параметров и типа стратегии.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия сообщает о своем типе и параметрах
        - Помогает понять, какая именно стратегия выполняется
        """
        # Определение типа стратегии по имени класса
        strategy_type = self._get_strategy_type()
        self.debug_fmt.debug(f"Тип: {strategy_type}", indent=1)

        # Логирование параметров конфигурации
        if self.config:
            self.debug_fmt.debug("Параметры стратегии:", indent=1)
            for key, value in self.config.items():
                if key != 'debug_mode':  # Пропускаем служебные параметры
                    self.debug_fmt.debug(f"{key}: {value}", indent=2)

    def _get_strategy_type(self) -> str:
        """
        Определение типа стратегии по имени класса.

        Returns:
            str: Человеко-читаемое описание типа стратегии
        """
        class_name = self.__class__.__name__.lower()

        if 'preprocessing' in class_name:
            return "Предобработка изображений"
        elif 'binarization' in class_name:
            return "Бинаризация изображений"
        elif 'roi' in class_name or 'extraction' in class_name:
            return "Выделение области интереса (ROI)"
        elif 'alignment' in class_name:
            return "Совмещение изображений"
        else:
            return "Обработка изображений"

    def _log_success(self, result: StrategyResult, execution_time: float,
                     context: Dict[str, Any]) -> None:
        """
        Логирование успешного выполнения стратегии.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия логирует свой успех с качеством и временем
        - Выводит таблицу метрик выполнения
        """
        debug_mode = context.get('debug_mode', False)
        if not debug_mode:
            return

        # Получение composite_score для отображения
        composite_score = result.metrics.get(
            'composite_score', 0.0) if result.metrics else 0.0

        # 1. СООБЩЕНИЕ ОБ УСПЕХЕ
        self.debug_fmt.success(
            f"Успешно | Качество: {composite_score:.3f} | Время: {execution_time:.3f}с",
            indent=1
        )

        # 2. ТАБЛИЦА МЕТРИК ВЫПОЛНЕНИЯ
        if result.metrics:
            self.debug_fmt.metrics_table(
                "Метрики выполнения", result.metrics, indent=1)

    def _log_failure(self, result: StrategyResult, execution_time: float,
                     context: Dict[str, Any]) -> None:
        """
        Логирование неудачного выполнения стратегии.

        Args:
            result: Результат с информацией об ошибке
            execution_time: Время выполнения до ошибки
            context: Контекст выполнения
        """
        debug_mode = context.get('debug_mode', False)
        if not debug_mode:
            return

        self.debug_fmt.error(
            f"Ошибка | Время: {execution_time:.3f}с | {result.error_message}",
            indent=1
        )

    def _handle_execution_error(self, error: Exception, execution_time: float,
                                context: Dict[str, Any]) -> StrategyResult:
        """
        Обработка исключений при выполнении стратегии.

        Args:
            error: Пойманное исключение
            execution_time: Время выполнения до ошибки
            context: Контекст выполнения

        Returns:
            StrategyResult: Результат с информацией об ошибке
        """
        debug_mode = context.get('debug_mode', False)

        # Логирование исключения
        if debug_mode:
            self.debug_fmt.error(
                f"Исключение | Время: {execution_time:.3f}с | {str(error)}",
                indent=1
            )

        # Создание результата с ошибкой
        return StrategyResult(
            strategy_name=self.name,
            success=False,
            result_data=None,
            metrics={},
            processing_time=execution_time,
            error_message=str(error)
        )

    @abstractmethod
    def execute(self, input_data: Any, context: Dict[str, Any]) -> StrategyResult:
        """
        Основная логика выполнения стратегии.

        Args:
            input_data: Входные данные для обработки
            context: Контекст выполнения

        Returns:
            StrategyResult: Результат выполнения стратегии

        Raises:
            Exception: Любые ошибки выполнения должны пробрасываться для обработки
        """
        pass

    def is_applicable(self, input_data: Any, context: Dict[str, Any]) -> bool:
        """
        Проверка применимости стратегии к входным данным.

        Args:
            input_data: Данные для проверки применимости
            context: Контекст выполнения

        Returns:
            bool: True если стратегия может быть применена к данным
        """
        return True


# ========================================================================
# СПЕЦИАЛИЗИРОВАННЫЕ БАЗОВЫЕ КЛАССЫ ДЛЯ КОНКРЕТНЫХ ТИПОВ СТРАТЕГИЙ
# ========================================================================

class PreprocessingStrategy(ProcessingStrategy):
    """Базовый класс для стратегий предобработки изображений."""

    def _get_strategy_type(self) -> str:
        """Переопределение типа для стратегий предобработки."""
        return "Предобработка изображений"


class BinarizationStrategy(ProcessingStrategy):
    """Базовый класс для стратегий бинаризации изображений."""

    def _get_strategy_type(self) -> str:
        """Переопределение типа для стратегий бинаризации."""
        return "Бинаризация изображений"


class ROIExtractionStrategy(ProcessingStrategy):
    """Базовый класс для стратегий выделения области интереса."""

    def _get_strategy_type(self) -> str:
        """Переопределение типа для стратегий ROI."""
        return "Выделение области интереса (ROI)"


class AlignmentStrategy(ProcessingStrategy):
    """Базовый класс для стратегий совмещения изображений."""

    def _get_strategy_type(self) -> str:
        """Переопределение типа для стратегий совмещения."""
        return "Совмещение изображений"
