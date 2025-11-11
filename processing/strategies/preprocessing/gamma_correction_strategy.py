# processing/strategies/preprocessing_strategies/gamma_correction_strategy.py
"""
Стратегия умной гамма-коррекции с адаптацией под яркость и контраст.

GammaCorrectionPreprocessingStrategy (Конкретный исполнитель) отвечает за:
- Адаптивную гамма-коррекцию на основе анализа яркости изображения
- Интеллектуальную настройку параметра gamma на основе гистограммы
- Детальное логирование процесса гамма-коррекции и адаптивных настроек

СООТВЕТСТВИЕ ИНСТРУКЦИИ:
- Уровень 4 в иерархии логирования (конкретная реализация стратегии)
- Логирует специфичные детали КОНКРЕТНОГО метода гамма-коррекции
- Объясняет адаптивные настройки параметра gamma на основе анализа яркости
"""

import time
from typing import Any, Dict

import cv2
import numpy as np

from .base_preprocessing import BasePreprocessingStrategy, StrategyResult


class GammaCorrectionPreprocessingStrategy(BasePreprocessingStrategy):
    """
    Умная гамма-коррекция с адаптацией под яркость и контраст.

    Особенности реализации:
    - Адаптивный расчет параметра gamma на основе анализа гистограммы
    - Интеллектуальная коррекция для темных, нормальных и ярких изображений
    - Детальное логирование процесса гамма-коррекции
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Инициализация стратегии гамма-коррекции.

        Args:
            name: Уникальное имя стратегии (фиксируется как "GammaCorrection")
            config: Конфигурация с базовыми параметрами обработки
        """
        super().__init__("GammaCorrection", config)

        # Базовые параметры из конфигурации
        self.gamma = self.config.get('gamma', 1.2)

    def _process_image(self, image: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """
        Основная логика адаптивной гамма-коррекции.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Конкретная стратегия детально логирует свой уникальный процесс
        - Объясняет адаптивные настройки параметра gamma на основе анализа яркости
        - Показывает последовательность операций гамма-коррекции

        Args:
            image: Входное изображение для обработки
            context: Контекст выполнения с global_analysis

        Returns:
            np.ndarray: Обработанное изображение после гамма-коррекции
        """
        debug_mode = context.get('debug_mode', False)

        if debug_mode:
            self.debug_fmt.debug("Начало адаптивной гамма-коррекции", indent=1)

        # Конвертация в grayscale если необходимо
        gray = self._convert_to_grayscale(image)

        # 1. АДАПТИВНЫЙ РАСЧЕТ ПАРАМЕТРА GAMMA
        gamma = self._calculate_adaptive_gamma()

        if debug_mode:
            self.debug_fmt.debug(
                "Адаптивные параметры гамма-коррекции:", indent=1)
            self.debug_fmt.debug(f"Gamma: {gamma:.3f}", indent=2)
            self._log_gamma_decision_logic(gamma)

        # 2. ПРИМЕНЕНИЕ ГАММА-КОРРЕКЦИИ
        # Нормализация в диапазон [0, 1]
        normalized = gray.astype(np.float32) / 255.0

        if debug_mode:
            self.debug_fmt.debug("Изображение нормализовано", indent=1)

        # Применение гамма-коррекции
        corrected = np.power(normalized, 1.0 / gamma)

        if debug_mode:
            self.debug_fmt.debug("Гамма-коррекция применена", indent=1)

        # Денормализация обратно в [0, 255]
        processed_image = (corrected * 255).astype(np.uint8)

        if debug_mode:
            self.debug_fmt.debug("Изображение денормализовано", indent=1)
            self.debug_fmt.debug("Гамма-коррекция завершена", indent=1)

        return processed_image

    def _calculate_adaptive_gamma(self) -> float:
        """
        Адаптивный расчет параметра gamma на основе анализа яркости изображения.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Конкретная стратегия объясняет логику настройки СВОЕГО параметра gamma
        - Показывает как анализ яркости влияет на выбор значения gamma

        Returns:
            float: Адаптивное значение gamma для коррекции
        """
        # Базовое значение из конфигурации
        gamma = self.gamma

        if not self.scan_analysis:
            return gamma

        # ЛОГИКА АДАПТИВНОЙ НАСТРОЙКИ ДЛЯ КОНКРЕТНОЙ СТРАТЕГИИ ГАММА-КОРРЕКЦИИ

        mean_intensity = self.scan_analysis.get('mean_intensity', 128)

        # 1. КОРРЕКЦИЯ ДЛЯ ТЕМНЫХ ИЗОБРАЖЕНИЙ
        if self.scan_analysis.get('is_dark', False) or mean_intensity < 80:
            gamma = 0.6  # Сильное осветление для темных изображений

        # 2. КОРРЕКЦИЯ ДЛЯ ЯРКИХ ИЗОБРАЖЕНИЙ
        elif self.scan_analysis.get('is_bright', False) or mean_intensity > 180:
            gamma = 1.8  # Затемнение для переэкспонированных изображений

        # 3. СТАНДАРТНАЯ КОРРЕКЦИЯ ДЛЯ НОРМАЛЬНОЙ ЯРКОСТИ
        elif mean_intensity < 100:
            gamma = 1.0  # Нейтральная коррекция
        elif mean_intensity > 150:
            gamma = 1.4  # Умеренное затемнение
        else:
            gamma = 1.2  # Стандартное значение

        # 4. ДОПОЛНИТЕЛЬНАЯ КОРРЕКЦИЯ ДЛЯ ВЫСОКОЙ ПЛОТНОСТИ
        if (self.reference_analysis and
                self.reference_analysis.get('is_high_density', False)):
            # Слегка осветляем для лучшей детализации
            gamma = max(0.7, gamma - 0.1)

        # 5. КОРРЕКЦИЯ ДЛЯ НИЗКОГО КОНТРАСТА
        if self.scan_analysis.get('low_contrast', False):
            gamma = min(2.0, gamma + 0.2)  # Увеличиваем для усиления контраста

        # Логирование адаптивных решений
        debug_mode = getattr(self.debug_fmt, 'debug_mode', False)
        if debug_mode:
            self.debug_fmt.debug("Логика настройки gamma:", indent=2)

            if self.scan_analysis.get('is_dark', False) or mean_intensity < 80:
                self.debug_fmt.debug(
                    "→ Темное изображение → gamma=0.6 (осветление)", indent=3)
            elif self.scan_analysis.get('is_bright', False) or mean_intensity > 180:
                self.debug_fmt.debug(
                    "→ Яркое изображение → gamma=1.8 (затемнение)", indent=3)
            elif mean_intensity < 100:
                self.debug_fmt.debug(
                    "→ Нормальная яркость (низкая) → gamma=1.0", indent=3)
            elif mean_intensity > 150:
                self.debug_fmt.debug(
                    "→ Нормальная яркость (высокая) → gamma=1.4", indent=3)
            else:
                self.debug_fmt.debug(
                    "→ Нормальная яркость → gamma=1.2", indent=3)

            if (self.reference_analysis and
                    self.reference_analysis.get('is_high_density', False)):
                self.debug_fmt.debug(
                    "→ Высокая плотность → уменьшение gamma", indent=3)

            if self.scan_analysis.get('low_contrast', False):
                self.debug_fmt.debug(
                    "→ Низкий контраст → увеличение gamma", indent=3)

        return round(gamma, 3)

    def _log_gamma_decision_logic(self, final_gamma: float) -> None:
        """
        Детальное логирование логики принятия решения по значению gamma.

        Args:
            final_gamma: Финальное значение gamma после всех корректировок
        """
        debug_mode = getattr(self.debug_fmt, 'debug_mode', False)
        if not debug_mode or not self.scan_analysis:
            return

        mean_intensity = self.scan_analysis.get('mean_intensity', 128)

        self.debug_fmt.debug("Анализ яркости для принятия решения:", indent=2)
        self.debug_fmt.debug(
            f"Средняя интенсивность: {mean_intensity:.1f}", indent=3)

        brightness_category = "темное" if mean_intensity < 80 else \
            "яркое" if mean_intensity > 180 else \
            "нормальное"
        self.debug_fmt.debug(
            f"Категория яркости: {brightness_category}", indent=3)

        if self.scan_analysis.get('low_contrast', False):
            self.debug_fmt.debug("Обнаружен низкий контраст", indent=3)

        if (self.reference_analysis and
                self.reference_analysis.get('is_high_density', False)):
            self.debug_fmt.debug(
                "Обнаружена высокая плотность деталей", indent=3)

    def _log_adaptive_parameters(self) -> None:
        """
        Специфичное логирование адаптивных параметров для стратегии гамма-коррекции.

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
        gamma = self._calculate_adaptive_gamma()

        # Логирование параметров стратегии гамма-коррекции
        self.debug_fmt.debug("Адаптивные параметры GammaCorrection:", indent=1)
        self.debug_fmt.debug(f"Gamma: {gamma:.3f}", indent=2)

        # Детальное объяснение логики настройки
        if self.scan_analysis:
            self._log_gamma_decision_logic(gamma)

        # Сохранение для использования в детальном логировании
        self._current_parameters = {
            'gamma_value': gamma
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
                f"Ошибка выполнения гамма-коррекции: {error}", indent=1)

        return StrategyResult(
            strategy_name=self.name,
            success=False,
            result_data=None,
            metrics={},
            processing_time=execution_time,
            error_message=str(error)
        )
