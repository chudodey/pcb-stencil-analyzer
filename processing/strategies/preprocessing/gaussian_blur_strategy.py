# processing/strategies/preprocessing_strategies/gaussian_blur_strategy.py
"""
Стратегия умного Гауссова размытия с адаптацией под шум и размер features.

GaussianBlurPreprocessingStrategy (Конкретный исполнитель) отвечает за:
- Адаптивное Гауссово размытие для подавления шума и сглаживания
- Интеллектуальную настройку размера ядра и параметра sigma
- Детальное логирование процесса Гауссовой фильтрации и адаптивных настроек

СООТВЕТСТВИЕ ИНСТРУКЦИИ:
- Уровень 4 в иерархии логирования (конкретная реализация стратегии)
- Логирует специфичные детали КОНКРЕТНОГО метода Гауссова размытия
- Объясняет адаптивные настройки параметров на основе анализа шума и деталей
"""

import time
from typing import Any, Dict

import cv2
import numpy as np

from .base_preprocessing import BasePreprocessingStrategy, StrategyResult


class GaussianBlurPreprocessingStrategy(BasePreprocessingStrategy):
    """
    Умное Гауссово размытие с адаптацией под шум и размер features.

    Особенности реализации:
    - Адаптивный расчет размера ядра и sigma на основе уровня шума
    - Интеллектуальная балансировка между сглаживанием и сохранением деталей
    - Детальное логирование процесса Гауссовой фильтрации
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Инициализация стратегии Гауссова размытия.

        Args:
            name: Уникальное имя стратегии (фиксируется как "GaussianBlur")
            config: Конфигурация с базовыми параметрами обработки
        """
        super().__init__("GaussianBlur", config)

        # Базовые параметры из конфигурации
        self.kernel_size = self.config.get('kernel_size', 5)

    def _process_image(self, image: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """
        Основная логика адаптивного Гауссова размытия.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Конкретная стратегия детально логирует свой уникальный процесс
        - Объясняет адаптивные настройки размера ядра и параметра sigma
        - Показывает последовательность операций Гауссовой фильтрации

        Args:
            image: Входное изображение для обработки
            context: Контекст выполнения с global_analysis

        Returns:
            np.ndarray: Обработанное изображение после Гауссова размытия
        """
        debug_mode = context.get('debug_mode', False)

        if debug_mode:
            self.debug_fmt.debug(
                "Начало адаптивного Гауссова размытия", indent=1)

        # Конвертация в grayscale если необходимо
        gray = self._convert_to_grayscale(image)

        # 1. АДАПТИВНЫЙ РАСЧЕТ ПАРАМЕТРОВ РАЗМЫТИЯ
        kernel_size = self._calculate_adaptive_kernel_size()
        sigma = self._calculate_adaptive_sigma()

        if debug_mode:
            self.debug_fmt.debug(
                "Адаптивные параметры Гауссова размытия:", indent=1)
            self.debug_fmt.debug(f"Размер ядра: {kernel_size}", indent=2)
            self.debug_fmt.debug(f"Sigma: {sigma:.2f}", indent=2)
            self._log_blur_decision_logic(kernel_size, sigma)

        # 2. ПРИМЕНЕНИЕ ГАУССОВА РАЗМЫТИЯ
        processed_image = cv2.GaussianBlur(
            gray, (kernel_size, kernel_size), sigma)

        if debug_mode:
            self.debug_fmt.debug(
                "Гауссово размытие применено успешно", indent=1)
            self.debug_fmt.debug("Гауссова фильтрация завершена", indent=1)

        return processed_image

    def _calculate_adaptive_kernel_size(self) -> int:
        """
        Адаптивный расчет размера ядра для Гауссова размытия.

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

        # ЛОГИКА АДАПТИВНОЙ НАСТРОЙКИ ДЛЯ КОНКРЕТНОЙ СТРАТЕГИИ ГАУССОВА РАЗМЫТИЯ

        noise_level = self.scan_analysis.get('noise_level', 0)

        # 1. КОРРЕКЦИЯ ДЛЯ СИЛЬНОГО ШУМА
        if noise_level > 0.2:
            kernel_size = 7  # Увеличиваем для сильного шума
        elif noise_level > 0.1:
            kernel_size = 5  # Стандартное увеличение для умеренного шума
        else:
            kernel_size = 3  # Минимальное размытие для слабого шума

        # 2. КОРРЕКЦИЯ ДЛЯ СОХРАНЕНИЯ МЕЛКИХ ДЕТАЛЕЙ
        if (self.reference_analysis and
                self.reference_analysis.get('component_sizes', {}).get('mean_size', 0) < 30):
            kernel_size = 3  # Уменьшаем для сохранения мелких деталей

        # 3. КОРРЕКЦИЯ ДЛЯ ВЫСОКОЙ ПЛОТНОСТИ ДЕТАЛЕЙ
        if (self.reference_analysis and
                self.reference_analysis.get('is_high_density', False)):
            kernel_size = 3  # Уменьшаем для высокой плотности

        # 4. КОРРЕКЦИЯ ДЛЯ ТЕКСТУРИРОВАННЫХ ИЗОБРАЖЕНИЙ
        gradient_std = self.scan_analysis.get(
            'gradient_analysis', {}).get('std_gradient', 0)
        if gradient_std > 50:
            # Увеличиваем для сильной текстуры
            kernel_size = min(7, kernel_size + 2)

        # Логирование адаптивных решений
        debug_mode = getattr(self.debug_fmt, 'debug_mode', False)
        if debug_mode:
            self.debug_fmt.debug("Логика настройки размера ядра:", indent=2)

            if noise_level > 0.2:
                self.debug_fmt.debug("→ Сильный шум → kernel=7", indent=3)
            elif noise_level > 0.1:
                self.debug_fmt.debug("→ Умеренный шум → kernel=5", indent=3)
            else:
                self.debug_fmt.debug("→ Слабый шум → kernel=3", indent=3)

            if (self.reference_analysis and
                    self.reference_analysis.get('component_sizes', {}).get('mean_size', 0) < 30):
                self.debug_fmt.debug(
                    "→ Мелкие детали → уменьшение ядра", indent=3)

            if (self.reference_analysis and
                    self.reference_analysis.get('is_high_density', False)):
                self.debug_fmt.debug(
                    "→ Высокая плотность → уменьшение ядра", indent=3)

            if gradient_std > 50:
                self.debug_fmt.debug(
                    "→ Сильная текстура → увеличение ядра", indent=3)

        # Гарантия нечетности для OpenCV
        return kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    def _calculate_adaptive_sigma(self) -> float:
        """
        Адаптивный расчет параметра sigma для Гауссова размытия.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Конкретная стратегия объясняет логику настройки СВОЕГО параметра sigma
        - Показывает как характеристики изображения влияют на выбор sigma

        Returns:
            float: Адаптивное значение sigma
        """
        # Базовое значение sigma (0 означает авто-расчет на основе kernel_size)
        sigma = 0.0

        if not self.scan_analysis:
            return sigma

        # ЛОГИКА АДАПТИВНОЙ НАСТРОЙКИ ДЛЯ КОНКРЕТНОЙ СТРАТЕГИИ ГАУССОВА РАЗМЫТИЯ

        noise_level = self.scan_analysis.get('noise_level', 0)
        gradient_std = self.scan_analysis.get(
            'gradient_analysis', {}).get('std_gradient', 0)

        # 1. КОРРЕКЦИЯ ДЛЯ СИЛЬНОГО ШУМА
        if noise_level > 0.2:
            sigma = 2.0  # Увеличиваем sigma для сильного шума
        elif noise_level > 0.1:
            sigma = 1.5  # Умеренное увеличение для шума
        else:
            sigma = 1.0  # Стандартное значение

        # 2. КОРРЕКЦИЯ ДЛЯ ТЕКСТУРИРОВАННЫХ ИЗОБРАЖЕНИЙ
        if gradient_std > 60:
            sigma = min(3.0, sigma + 1.0)  # Увеличиваем для сильной текстуры
        elif gradient_std > 40:
            sigma = min(2.0, sigma + 0.5)  # Умеренное увеличение

        # 3. КОРРЕКЦИЯ ДЛЯ ГЛАДКИХ ИЗОБРАЖЕНИЙ
        if gradient_std < 20:
            sigma = max(0.5, sigma - 0.5)  # Уменьшаем для гладких изображений

        # 4. КОРРЕКЦИЯ ДЛЯ МЕЛКИХ ДЕТАЛЕЙ
        if (self.reference_analysis and
                self.reference_analysis.get('component_sizes', {}).get('mean_size', 0) < 30):
            # Уменьшаем для сохранения мелких деталей
            sigma = max(0.5, sigma - 0.3)

        # Логирование адаптивных решений
        debug_mode = getattr(self.debug_fmt, 'debug_mode', False)
        if debug_mode:
            self.debug_fmt.debug("Логика настройки sigma:", indent=2)

            if noise_level > 0.2:
                self.debug_fmt.debug("→ Сильный шум → sigma=2.0", indent=3)
            elif noise_level > 0.1:
                self.debug_fmt.debug("→ Умеренный шум → sigma=1.5", indent=3)
            else:
                self.debug_fmt.debug("→ Слабый шум → sigma=1.0", indent=3)

            if gradient_std > 60:
                self.debug_fmt.debug(
                    "→ Сильная текстура → увеличение sigma", indent=3)
            elif gradient_std > 40:
                self.debug_fmt.debug(
                    "→ Умеренная текстура → увеличение sigma", indent=3)
            elif gradient_std < 20:
                self.debug_fmt.debug(
                    "→ Гладкое изображение → уменьшение sigma", indent=3)

            if (self.reference_analysis and
                    self.reference_analysis.get('component_sizes', {}).get('mean_size', 0) < 30):
                self.debug_fmt.debug(
                    "→ Мелкие детали → уменьшение sigma", indent=3)

        return round(sigma, 2)

    def _log_blur_decision_logic(self, final_kernel: int, final_sigma: float) -> None:
        """
        Детальное логирование логики принятия решений по параметрам размытия.

        Args:
            final_kernel: Финальный размер ядра после всех корректировок
            final_sigma: Финальное значение sigma после всех корректировок
        """
        debug_mode = getattr(self.debug_fmt, 'debug_mode', False)
        if not debug_mode or not self.scan_analysis:
            return

        noise_level = self.scan_analysis.get('noise_level', 0)
        gradient_std = self.scan_analysis.get(
            'gradient_analysis', {}).get('std_gradient', 0)

        self.debug_fmt.debug("Анализ для принятия решений:", indent=2)
        self.debug_fmt.debug(f"Уровень шума: {noise_level:.3f}", indent=3)
        self.debug_fmt.debug(
            f"Стандартное отклонение градиента: {gradient_std:.1f}", indent=3)

        # Анализ характеристик деталей
        if self.reference_analysis:
            mean_size = self.reference_analysis.get(
                'component_sizes', {}).get('mean_size', 0)
            is_high_density = self.reference_analysis.get(
                'is_high_density', False)

            self.debug_fmt.debug(
                f"Средний размер деталей: {mean_size:.1f}px", indent=3)
            if is_high_density:
                self.debug_fmt.debug("Обнаружена высокая плотность", indent=3)

    def _log_adaptive_parameters(self) -> None:
        """
        Специфичное логирование адаптивных параметров для стратегии Гауссова размытия.

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
        kernel_size = self._calculate_adaptive_kernel_size()
        sigma = self._calculate_adaptive_sigma()

        # Логирование параметров стратегии Гауссова размытия
        self.debug_fmt.debug("Адаптивные параметры GaussianBlur:", indent=1)
        self.debug_fmt.debug(f"Размер ядра: {kernel_size}", indent=2)
        self.debug_fmt.debug(f"Sigma: {sigma:.2f}", indent=2)

        # Детальное объяснение логики настройки
        if self.scan_analysis:
            self._log_blur_decision_logic(kernel_size, sigma)

        # Сохранение для использования в детальном логировании
        self._current_parameters = {
            'gaussian_kernel_size': kernel_size,
            'gaussian_sigma': sigma
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
                f"Ошибка выполнения Гауссова размытия: {error}", indent=1)

        return StrategyResult(
            strategy_name=self.name,
            success=False,
            result_data=None,
            metrics={},
            processing_time=execution_time,
            error_message=str(error)
        )
