# processing/strategies/preprocessing_strategies/median_blur_strategy.py
"""
Стратегия умного медианного фильтра для импульсного шума.

MedianBlurPreprocessingStrategy (Конкретный исполнитель) отвечает за:
- Адаптивный медианный фильтр для подавления импульсного шума (соль-перец)
- Интеллектуальную настройку размера ядра на основе анализа шума и деталей
- Детальное логирование процесса медианной фильтрации и адаптивных настроек

СООТВЕТСТВИЕ ИНСТРУКЦИИ:
- Уровень 4 в иерархии логирования (конкретная реализация стратегии)
- Логирует специфичные детали КОНКРЕТНОГО метода медианной фильтрации
- Объясняет адаптивные настройки размера ядра на основе анализа шума и деталей
"""

import time
from typing import Any, Dict

import cv2
import numpy as np

from .base_preprocessing import BasePreprocessingStrategy, StrategyResult


class MedianBlurPreprocessingStrategy(BasePreprocessingStrategy):
    """
    Умный медианный фильтр для импульсного шума.

    Особенности реализации:
    - Адаптивный расчет размера ядра на основе уровня импульсного шума
    - Интеллектуальная балансировка между подавлением шума и сохранением деталей
    - Детальное логирование процесса медианной фильтрации
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Инициализация стратегии медианного фильтра.

        Args:
            name: Уникальное имя стратегии (фиксируется как "MedianBlur")
            config: Конфигурация с базовыми параметрами обработки
        """
        super().__init__("MedianBlur", config)

        # Базовые параметры из конфигурации
        self.kernel_size = self.config.get('kernel_size', 5)

    def _process_image(self, image: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """
        Основная логика адаптивного медианного фильтра.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Конкретная стратегия детально логирует свой уникальный процесс
        - Объясняет адаптивные настройки размера ядра на основе анализа шума
        - Показывает последовательность операций медианной фильтрации

        Args:
            image: Входное изображение для обработки
            context: Контекст выполнения с global_analysis

        Returns:
            np.ndarray: Обработанное изображение после медианной фильтрации
        """
        debug_mode = context.get('debug_mode', False)

        if debug_mode:
            self.debug_fmt.debug(
                "Начало адаптивной медианной фильтрации", indent=1)

        # Конвертация в grayscale если необходимо
        gray = self._convert_to_grayscale(image)

        # 1. АДАПТИВНЫЙ РАСЧЕТ РАЗМЕРА ЯДРА
        kernel_size = self._calculate_adaptive_median_kernel()

        if debug_mode:
            self.debug_fmt.debug(
                "Адаптивные параметры медианного фильтра:", indent=1)
            self.debug_fmt.debug(f"Размер ядра: {kernel_size}", indent=2)
            self._log_kernel_decision_logic(kernel_size)

        # 2. ПРИМЕНЕНИЕ МЕДИАННОГО ФИЛЬТРА
        processed_image = cv2.medianBlur(gray, kernel_size)

        if debug_mode:
            self.debug_fmt.debug("Медианный фильтр применен успешно", indent=1)
            self.debug_fmt.debug("Медианная фильтрация завершена", indent=1)

        return processed_image

    def _calculate_adaptive_median_kernel(self) -> int:
        """
        Адаптивный расчет размера ядра медианного фильтра.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Конкретная стратегия объясняет логику настройки СВОЕГО размера ядра
        - Показывает как анализ шума и деталей влияет на выбор размера ядра

        Returns:
            int: Адаптивный размер ядра (всегда нечетный)
        """
        # Базовый размер из конфигурации
        kernel_size = self.kernel_size

        if not self.scan_analysis:
            return kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        # ЛОГИКА АДАПТИВНОЙ НАСТРОЙКИ ДЛЯ КОНКРЕТНОЙ СТРАТЕГИИ МЕДИАННОГО ФИЛЬТРА

        # 1. КОРРЕКЦИЯ ДЛЯ СИЛЬНОГО ИМПУЛЬСНОГО ШУМА
        noise_level = self.scan_analysis.get('noise_level', 0)
        if noise_level > 0.15:
            kernel_size = 5  # Увеличиваем для сильного шума
        elif noise_level > 0.25:
            kernel_size = 7  # Максимальное увеличение для очень сильного шума

        # 2. ОБНАРУЖЕНИЕ ИМПУЛЬСНОГО ШУМА В ГИСТОГРАММЕ
        peaks = self.scan_analysis.get('histogram_peaks', [])
        if len(peaks) > 2 and any(peak[1] < 0.05 for peak in peaks):
            # Мелкие изолированные пики могут указывать на импульсный шум
            kernel_size = max(kernel_size, 5)  # Увеличиваем если нужно

        # 3. КОРРЕКЦИЯ ДЛЯ СОХРАНЕНИЯ МЕЛКИХ ДЕТАЛЕЙ
        if (self.reference_analysis and
                self.reference_analysis.get('component_sizes', {}).get('mean_size', 0) < 30):
            kernel_size = 3  # Уменьшаем для сохранения мелких деталей

        # 4. КОРРЕКЦИЯ ДЛЯ ВЫСОКОЙ ПЛОТНОСТИ ДЕТАЛЕЙ
        if (self.reference_analysis and
                self.reference_analysis.get('is_high_density', False)):
            kernel_size = 3  # Уменьшаем для высокой плотности

        # 5. КОРРЕКЦИЯ ДЛЯ ТОНКИХ ЛИНИЙ И ГРАНИЦ
        if (self.reference_analysis and
                self.reference_analysis.get('has_thin_lines', False)):
            kernel_size = 3  # Минимальный размер для сохранения тонких линий

        # Логирование адаптивных решений
        debug_mode = getattr(self.debug_fmt, 'debug_mode', False)
        if debug_mode:
            self.debug_fmt.debug("Логика настройки размера ядра:", indent=2)

            if noise_level > 0.25:
                self.debug_fmt.debug(
                    "→ Очень сильный шум → kernel=7", indent=3)
            elif noise_level > 0.15:
                self.debug_fmt.debug("→ Сильный шум → kernel=5", indent=3)
            else:
                self.debug_fmt.debug("→ Умеренный шум → kernel=3", indent=3)

            if len(peaks) > 2 and any(peak[1] < 0.05 for peak in peaks):
                self.debug_fmt.debug(
                    "→ Обнаружен импульсный шум в гистограмме", indent=3)

            if (self.reference_analysis and
                    self.reference_analysis.get('component_sizes', {}).get('mean_size', 0) < 30):
                self.debug_fmt.debug(
                    "→ Мелкие детали → уменьшение ядра", indent=3)

            if (self.reference_analysis and
                    self.reference_analysis.get('is_high_density', False)):
                self.debug_fmt.debug(
                    "→ Высокая плотность → уменьшение ядра", indent=3)

            if (self.reference_analysis and
                    self.reference_analysis.get('has_thin_lines', False)):
                self.debug_fmt.debug(
                    "→ Тонкие линии → минимальное ядро", indent=3)

        # Гарантия нечетности для OpenCV
        return kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    def _log_kernel_decision_logic(self, final_kernel: int) -> None:
        """
        Детальное логирование логики принятия решения по размеру ядра.

        Args:
            final_kernel: Финальный размер ядра после всех корректировок
        """
        debug_mode = getattr(self.debug_fmt, 'debug_mode', False)
        if not debug_mode or not self.scan_analysis:
            return

        noise_level = self.scan_analysis.get('noise_level', 0)
        peaks = self.scan_analysis.get('histogram_peaks', [])

        self.debug_fmt.debug("Анализ для принятия решения:", indent=2)
        self.debug_fmt.debug(f"Уровень шума: {noise_level:.3f}", indent=3)
        self.debug_fmt.debug(f"Пиков в гистограмме: {len(peaks)}", indent=3)

        # Анализ характеристик деталей
        if self.reference_analysis:
            mean_size = self.reference_analysis.get(
                'component_sizes', {}).get('mean_size', 0)
            has_thin_lines = self.reference_analysis.get(
                'has_thin_lines', False)
            is_high_density = self.reference_analysis.get(
                'is_high_density', False)

            self.debug_fmt.debug(
                f"Средний размер деталей: {mean_size:.1f}px", indent=3)
            if has_thin_lines:
                self.debug_fmt.debug("Обнаружены тонкие линии", indent=3)
            if is_high_density:
                self.debug_fmt.debug("Обнаружена высокая плотность", indent=3)

    def _log_adaptive_parameters(self) -> None:
        """
        Специфичное логирование адаптивных параметров для стратегии медианного фильтра.

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
        kernel_size = self._calculate_adaptive_median_kernel()

        # Логирование параметров стратегии медианного фильтра
        self.debug_fmt.debug("Адаптивные параметры MedianBlur:", indent=1)
        self.debug_fmt.debug(f"Размер ядра: {kernel_size}", indent=2)

        # Детальное объяснение логики настройки
        if self.scan_analysis:
            self._log_kernel_decision_logic(kernel_size)

        # Сохранение для использования в детальном логировании
        self._current_parameters = {
            'median_kernel_size': kernel_size
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
                f"Ошибка выполнения медианной фильтрации: {error}", indent=1)

        return StrategyResult(
            strategy_name=self.name,
            success=False,
            result_data=None,
            metrics={},
            processing_time=execution_time,
            error_message=str(error)
        )
