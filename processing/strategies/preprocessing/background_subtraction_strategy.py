# processing/strategies/preprocessing_strategies/background_subtraction_strategy.py
"""
Стратегия умного вычитания фона с адаптацией под неравномерность освещения.

BackgroundSubtractionPreprocessingStrategy (Конкретный исполнитель) отвечает за:
- Адаптивное вычитание фона с учетом неравномерности освещения
- Интеллектуальную настройку параметров размытия на основе анализа изображения
- Детальное логирование процесса вычитания фона и адаптивных настроек

СООТВЕТСТВИЕ ИНСТРУКЦИИ:
- Уровень 4 в иерархии логирования (конкретная реализация стратегии)
- Логирует специфичные детали КОНКРЕТНОГО метода вычитания фона
- Объясняет адаптивные настройки параметров для этого метода
"""

import time
from typing import Any, Dict

import cv2
import numpy as np

from .base_preprocessing import BasePreprocessingStrategy, StrategyResult


class BackgroundSubtractionPreprocessingStrategy(BasePreprocessingStrategy):
    """
    Умное вычитание фона с адаптацией под неравномерность освещения.

    Особенности реализации:
    - Адаптивный расчет размера размытия фона на основе текстуры изображения
    - Интеллектуальная обработка неравномерного освещения
    - Детальное логирование процесса вычитания фона
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Инициализация стратегии вычитания фона.

        Args:
            name: Уникальное имя стратегии (фиксируется как "BackgroundSubtraction")
            config: Конфигурация с базовыми параметрами обработки
        """
        super().__init__("BackgroundSubtraction", config)

        # Базовые параметры из конфигурации
        self.background_blur = self.config.get('background_blur', 51)

    def _process_image(self, image: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """
        Основная логика адаптивного вычитания фона.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Конкретная стратегия детально логирует свой уникальный процесс
        - Объясняет адаптивные настройки параметров для вычитания фона
        - Показывает последовательность операций вычитания

        Args:
            image: Входное изображение для обработки
            context: Контекст выполнения с global_analysis

        Returns:
            np.ndarray: Обработанное изображение после вычитания фона
        """
        debug_mode = context.get('debug_mode', False)

        if debug_mode:
            self.debug_fmt.debug("Начало адаптивного вычитания фона", indent=1)

        # Конвертация в grayscale если необходимо
        gray = self._convert_to_grayscale(image)

        # 1. АДАПТИВНЫЙ РАСЧЕТ РАЗМЕРА РАЗМЫТИЯ ФОНА
        background_blur = self._calculate_adaptive_blur_size()

        if debug_mode:
            self.debug_fmt.debug(
                "Адаптивные параметры вычитания фона:", indent=1)
            self.debug_fmt.debug(
                f"Размер размытия фона: {background_blur}", indent=2)

        # 2. СОЗДАНИЕ МОДЕЛИ ФОНА ЧЕРЕЗ РАЗМЫТИЕ
        background = cv2.GaussianBlur(
            gray, (background_blur, background_blur), 0)

        if debug_mode:
            self.debug_fmt.debug("Модель фона создана успешно", indent=1)

        # 3. ВЫЧИТАНИЕ ФОНА ИЗ ОРИГИНАЛЬНОГО ИЗОБРАЖЕНИЯ
        foreground = cv2.subtract(gray, background)

        if debug_mode:
            self.debug_fmt.debug("Фон успешно вычтен", indent=1)

        # 4. АДАПТИВНАЯ НОРМАЛИЗАЦИЯ РЕЗУЛЬТАТА
        processed_image = self._adaptive_normalize(foreground)

        if debug_mode:
            self.debug_fmt.debug("Нормализация результата завершена", indent=1)
            self.debug_fmt.debug("Вычитание фона завершено", indent=1)

        return processed_image

    def _calculate_adaptive_blur_size(self) -> int:
        """
        Адаптивный расчет размера размытия фона на основе анализа изображения.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Конкретная стратегия объясняет логику настройки СВОИХ параметров
        - Показывает как анализ изображения влияет на размер размытия

        Returns:
            int: Адаптивный размер размытия (всегда нечетный)
        """
        # Базовый размер из конфигурации
        blur_size = self.background_blur

        if not self.scan_analysis:
            return blur_size if blur_size % 2 == 1 else blur_size + 1

        # ЛОГИКА АДАПТИВНОЙ НАСТРОЙКИ ДЛЯ КОНКРЕТНОЙ СТРАТЕГИИ ВЫЧИТАНИЯ ФОНА

        # 1. НАСТРОЙКА ДЛЯ СИЛЬНОЙ ТЕКСТУРЫ ФОНА
        gradient_std = self.scan_analysis.get(
            'gradient_analysis', {}).get('std_gradient', 0)

        if gradient_std > 50:  # Сильно текстурированный фон
            blur_size = 71  # Увеличиваем для лучшего усреднения текстуры
        elif gradient_std > 30:
            blur_size = 51  # Стандартный размер для умеренной текстуры
        else:  # Гладкий фон
            blur_size = 31  # Уменьшаем для сохранения деталей

        # 2. НАСТРОЙКА ДЛЯ НЕРАВНОМЕРНОГО ОСВЕЩЕНИЯ
        if self.scan_analysis.get('has_uneven_lighting', False):
            blur_size = min(81, blur_size + 20)  # Увеличиваем для компенсации

        # 3. НАСТРОЙКА ДЛЯ МЕЛКИХ ДЕТАЛЕЙ
        if (self.reference_analysis and
                self.reference_analysis.get('has_small_features', False)):
            # Уменьшаем для сохранения мелких деталей
            blur_size = max(21, blur_size - 10)

        # Логирование адаптивных параметров
        debug_mode = getattr(self.debug_fmt, 'debug_mode', False)
        if debug_mode:
            self.debug_fmt.debug(
                "Логика настройки размера размытия:", indent=2)
            if gradient_std > 50:
                self.debug_fmt.debug(
                    "→ Сильная текстура фона: увеличен размер размытия", indent=3)
            elif self.scan_analysis.get('has_uneven_lighting', False):
                self.debug_fmt.debug(
                    "→ Неравномерное освещение: увеличен размер размытия", indent=3)
            if (self.reference_analysis and
                    self.reference_analysis.get('has_small_features', False)):
                self.debug_fmt.debug(
                    "→ Мелкие детали: уменьшен размер размытия", indent=3)

        # Гарантия нечетности для OpenCV
        return blur_size if blur_size % 2 == 1 else blur_size + 1

    def _log_adaptive_parameters(self) -> None:
        """
        Специфичное логирование адаптивных параметров для стратегии вычитания фона.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Конкретная стратегия логирует свои уникальные адаптивные параметры
        - Объясняет логику настройки для этого конкретного метода

        Returns:
            None
        """
        debug_mode = getattr(self.debug_fmt, 'debug_mode', False)
        if not debug_mode:
            return

        # Расчет адаптивных параметров
        background_blur = self._calculate_adaptive_blur_size()

        # Логирование параметров стратегии вычитания фона
        self.debug_fmt.debug(
            "Адаптивные параметры BackgroundSubtraction:", indent=1)
        self.debug_fmt.debug(
            f"Размер размытия фона: {background_blur}", indent=2)

        # Объяснение логики настройки
        if self.scan_analysis:
            self.debug_fmt.debug("Логика адаптации:", indent=2)

            gradient_std = self.scan_analysis.get(
                'gradient_analysis', {}).get('std_gradient', 0)

            if gradient_std > 50:
                self.debug_fmt.debug(
                    "→ Высокая текстура фона → увеличение размытия", indent=3)
            elif self.scan_analysis.get('has_uneven_lighting', False):
                self.debug_fmt.debug(
                    "→ Неравномерное освещение → увеличение размытия", indent=3)

            if (self.reference_analysis and
                    self.reference_analysis.get('has_small_features', False)):
                self.debug_fmt.debug(
                    "→ Мелкие детали → уменьшение размытия", indent=3)

        # Сохранение для использования в детальном логировании
        self._current_parameters = {
            'background_blur_size': background_blur
        }

    def _create_error_result(self, error: Exception, start_time: float) -> StrategyResult:
        """
        Создание результата с информацией об ошибке выполнения.

        Args:
            error: Исключение, возникшее при выполнении стратегии
            start_time: Время начала выполнения для расчета длительности

        Returns:
            StrategyResult: Результат с информацией об ошибке
        """
        execution_time = time.time() - start_time
        debug_mode = getattr(self.debug_fmt, 'debug_mode', False)

        if debug_mode:
            self.debug_fmt.error(
                f"Ошибка выполнения вычитания фона: {error}", indent=1)

        return StrategyResult(
            strategy_name=self.name,
            success=False,
            result_data=None,
            metrics={},
            processing_time=execution_time,
            error_message=str(error)
        )
