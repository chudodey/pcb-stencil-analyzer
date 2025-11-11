# processing/strategies/preprocessing_strategies/unsharp_mask_strategy.py
"""
Стратегия умного увеличения резкости с адаптацией под исходную резкость.

UnsharpMaskPreprocessingStrategy (Конкретный исполнитель) отвечает за:
- Адаптивное увеличение резкости с использованием метода unsharp mask
- Интеллектуальную настройку силы усиления и размера ядра размытия
- Детальное логирование процесса увеличения резкости и адаптивных настроек

СООТВЕТСТВИЕ ИНСТРУКЦИИ:
- Уровень 4 в иерархии логирования (конкретная реализация стратегии)
- Логирует специфичные детали КОНКРЕТНОГО метода unsharp mask
- Объясняет адаптивные настройки параметров на основе анализа резкости и деталей
"""

import time
from typing import Any, Dict

import cv2
import numpy as np

from .base_preprocessing import BasePreprocessingStrategy, StrategyResult


class UnsharpMaskPreprocessingStrategy(BasePreprocessingStrategy):
    """
    Умное увеличение резкости с адаптацией под исходную резкость.

    Особенности реализации:
    - Адаптивный расчет силы усиления на основе исходной резкости изображения
    - Интеллектуальная настройка размера ядра размытия для создания маски
    - Детальное логирование процесса применения unsharp mask
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Инициализация стратегии unsharp mask.

        Args:
            name: Уникальное имя стратегии (фиксируется как "UnsharpMask")
            config: Конфигурация с базовыми параметрами обработки
        """
        super().__init__("UnsharpMask", config)

        # Базовые параметры из конфигурации
        self.strength = self.config.get('strength', 1.5)
        self.blur_kernel = self.config.get('blur_kernel', 5)

    def _process_image(self, image: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """
        Основная логика адаптивного увеличения резкости методом unsharp mask.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Конкретная стратегия детально логирует свой уникальный процесс
        - Объясняет адаптивные настройки силы усиления и размера ядра
        - Показывает последовательность операций unsharp mask

        Args:
            image: Входное изображение для обработки
            context: Контекст выполнения с global_analysis

        Returns:
            np.ndarray: Обработанное изображение после увеличения резкости
        """
        debug_mode = context.get('debug_mode', False)

        if debug_mode:
            self.debug_fmt.debug(
                "Начало адаптивного увеличения резкости", indent=1)

        # Конвертация в grayscale если необходимо
        gray = self._convert_to_grayscale(image)

        # 1. АДАПТИВНЫЙ РАСЧЕТ ПАРАМЕТРОВ РЕЗКОСТИ
        strength = self._calculate_adaptive_strength()
        blur_kernel = self._calculate_adaptive_blur_kernel()

        if debug_mode:
            self.debug_fmt.debug(
                "Адаптивные параметры unsharp mask:", indent=1)
            self.debug_fmt.debug(f"Сила усиления: {strength:.2f}", indent=2)
            self.debug_fmt.debug(
                f"Размер ядра размытия: {blur_kernel}", indent=2)
            self._log_sharpening_decision_logic(strength, blur_kernel)

        # 2. СОЗДАНИЕ РАЗМЫТОЙ ВЕРСИИ ИЗОБРАЖЕНИЯ (MASK)
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        if debug_mode:
            self.debug_fmt.debug("Маска размытия создана", indent=1)

        # 3. ВЫЧИТАНИЕ MASK ИЗ ОРИГИНАЛА И УСИЛЕНИЕ
        sharpened = cv2.addWeighted(
            gray, 1.0 + strength, blurred, -strength, 0)

        if debug_mode:
            self.debug_fmt.debug(
                "Резкость усилена методом unsharp mask", indent=1)

        # 4. ОБРЕЗКА И НОРМАЛИЗАЦИЯ РЕЗУЛЬТАТА
        processed_image = np.clip(sharpened, 0, 255).astype(np.uint8)

        if debug_mode:
            self.debug_fmt.debug("Результат обрезан и нормализован", indent=1)
            self.debug_fmt.debug("Увеличение резкости завершено", indent=1)

        return processed_image

    def _calculate_adaptive_strength(self) -> float:
        """
        Адаптивный расчет силы усиления резкости.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Конкретная стратегия объясняет логику настройки СВОЕЙ силы усиления
        - Показывает как анализ резкости влияет на выбор параметра strength

        Returns:
            float: Адаптивная сила усиления резкости
        """
        # Базовое значение из конфигурации
        strength = self.strength

        if not self.scan_analysis:
            return strength

        # ЛОГИКА АДАПТИВНОЙ НАСТРОЙКИ ДЛЯ КОНКРЕТНОЙ СТРАТЕГИИ UNSHARP MASK

        sharpness_ratio = self.scan_analysis.get(
            'gradient_analysis', {}).get('sharpness_ratio', 0)

        # 1. КОРРЕКЦИЯ ДЛЯ РАЗМЫТЫХ ИЗОБРАЖЕНИЙ
        if sharpness_ratio < 0.1:
            strength = 2.0  # Сильное усиление для размытых изображений
        elif sharpness_ratio < 0.2:
            strength = 1.8  # Умеренное усиление для слегка размытых

        # 2. КОРРЕКЦИЯ ДЛЯ УЖЕ РЕЗКИХ ИЗОБРАЖЕНИЙ
        elif sharpness_ratio > 0.5:
            strength = 1.0  # Минимальное усиление для резких изображений
        elif sharpness_ratio > 0.3:
            strength = 1.2  # Слабое усиление для достаточно резких

        # 3. КОРРЕКЦИЯ ДЛЯ ВЫСОКОЙ ПЛОТНОСТИ ДЕТАЛЕЙ
        if (self.reference_analysis and
                self.reference_analysis.get('is_high_density', False)):
            # Увеличиваем для лучшей детализации
            strength = min(2.0, strength + 0.3)

        # 4. КОРРЕКЦИЯ ДЛЯ ШУМНЫХ ИЗОБРАЖЕНИЙ
        if self.scan_analysis.get('noise_level', 0) > 0.1:
            # Уменьшаем чтобы не усиливать шум
            strength = max(1.0, strength - 0.2)

        # 5. КОРРЕКЦИЯ ДЛЯ ТЕКСТУРИРОВАННЫХ ИЗОБРАЖЕНИЙ
        gradient_std = self.scan_analysis.get(
            'gradient_analysis', {}).get('std_gradient', 0)
        if gradient_std > 40:
            strength = min(1.8, strength + 0.2)  # Увеличиваем для текстур

        # Логирование адаптивных решений
        debug_mode = getattr(self.debug_fmt, 'debug_mode', False)
        if debug_mode:
            self.debug_fmt.debug("Логика настройки силы усиления:", indent=2)

            if sharpness_ratio < 0.1:
                self.debug_fmt.debug(
                    "→ Сильно размытое → strength=2.0", indent=3)
            elif sharpness_ratio < 0.2:
                self.debug_fmt.debug("→ Размытое → strength=1.8", indent=3)
            elif sharpness_ratio > 0.5:
                self.debug_fmt.debug("→ Очень резкое → strength=1.0", indent=3)
            elif sharpness_ratio > 0.3:
                self.debug_fmt.debug("→ Резкое → strength=1.2", indent=3)
            else:
                self.debug_fmt.debug(
                    "→ Нормальная резкость → strength=1.5", indent=3)

            if (self.reference_analysis and
                    self.reference_analysis.get('is_high_density', False)):
                self.debug_fmt.debug(
                    "→ Высокая плотность → увеличение strength", indent=3)

            if self.scan_analysis.get('noise_level', 0) > 0.1:
                self.debug_fmt.debug(
                    "→ Шумное изображение → уменьшение strength", indent=3)

            if gradient_std > 40:
                self.debug_fmt.debug(
                    "→ Текстурированное → увеличение strength", indent=3)

        return round(strength, 2)

    def _calculate_adaptive_blur_kernel(self) -> int:
        """
        Адаптивный расчет размера ядра размытия для маски.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Конкретная стратегия объясняет логику настройки СВОЕГО размера ядра
        - Показывает как размер деталей влияет на выбор ядра размытия

        Returns:
            int: Адаптивный размер ядра размытия (всегда нечетный)
        """
        # Базовый размер из конфигурации
        kernel_size = self.blur_kernel

        if not self.scan_analysis or not self.reference_analysis:
            return kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        # ЛОГИКА АДАПТИВНОЙ НАСТРОЙКИ ДЛЯ КОНКРЕТНОЙ СТРАТЕГИИ UNSHARP MASK

        # 1. НАСТРОЙКА НА ОСНОВЕ РАЗМЕРА ДЕТАЛЕЙ
        if 'component_sizes' in self.reference_analysis:
            mean_size = self.reference_analysis['component_sizes'].get(
                'mean_size', 0)

            if mean_size > 100:
                kernel_size = 7  # Увеличиваем для крупных features
            elif mean_size < 30:
                kernel_size = 3  # Уменьшаем для мелких features
            else:
                kernel_size = 5  # Стандартный размер

        # 2. КОРРЕКЦИЯ ДЛЯ ТОНКИХ ЛИНИЙ
        if self.reference_analysis.get('has_thin_lines', False):
            # Уменьшаем для сохранения тонких линий
            kernel_size = max(3, kernel_size - 2)

        # 3. КОРРЕКЦИЯ ДЛЯ ВЫСОКОЙ РЕЗКОСТИ
        sharpness_ratio = self.scan_analysis.get(
            'gradient_analysis', {}).get('sharpness_ratio', 0)
        if sharpness_ratio > 0.4:
            # Уменьшаем для уже резких изображений
            kernel_size = max(3, kernel_size - 1)

        # Логирование адаптивных решений
        debug_mode = getattr(self.debug_fmt, 'debug_mode', False)
        if debug_mode:
            self.debug_fmt.debug("Логика настройки ядра размытия:", indent=2)

            if 'component_sizes' in self.reference_analysis:
                mean_size = self.reference_analysis['component_sizes'].get(
                    'mean_size', 0)
                if mean_size > 100:
                    self.debug_fmt.debug(
                        "→ Крупные детали → kernel=7", indent=3)
                elif mean_size < 30:
                    self.debug_fmt.debug(
                        "→ Мелкие детали → kernel=3", indent=3)
                else:
                    self.debug_fmt.debug(
                        "→ Средние детали → kernel=5", indent=3)

            if self.reference_analysis.get('has_thin_lines', False):
                self.debug_fmt.debug(
                    "→ Тонкие линии → уменьшение kernel", indent=3)

            if sharpness_ratio > 0.4:
                self.debug_fmt.debug(
                    "→ Высокая резкость → уменьшение kernel", indent=3)

        # Гарантия нечетности для OpenCV
        return kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    def _log_sharpening_decision_logic(self, final_strength: float, final_kernel: int) -> None:
        """
        Детальное логирование логики принятия решений по параметрам резкости.

        Args:
            final_strength: Финальная сила усиления после всех корректировок
            final_kernel: Финальный размер ядра после всех корректировок
        """
        debug_mode = getattr(self.debug_fmt, 'debug_mode', False)
        if not debug_mode or not self.scan_analysis:
            return

        sharpness_ratio = self.scan_analysis.get(
            'gradient_analysis', {}).get('sharpness_ratio', 0)
        noise_level = self.scan_analysis.get('noise_level', 0)
        gradient_std = self.scan_analysis.get(
            'gradient_analysis', {}).get('std_gradient', 0)

        self.debug_fmt.debug("Анализ для принятия решений:", indent=2)
        self.debug_fmt.debug(
            f"Коэффициент резкости: {sharpness_ratio:.3f}", indent=3)
        self.debug_fmt.debug(f"Уровень шума: {noise_level:.3f}", indent=3)
        self.debug_fmt.debug(
            f"Стандартное отклонение градиента: {gradient_std:.1f}", indent=3)

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
        Специфичное логирование адаптивных параметров для стратегии unsharp mask.

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
        strength = self._calculate_adaptive_strength()
        blur_kernel = self._calculate_adaptive_blur_kernel()

        # Логирование параметров стратегии unsharp mask
        self.debug_fmt.debug("Адаптивные параметры UnsharpMask:", indent=1)
        self.debug_fmt.debug(f"Сила усиления: {strength:.2f}", indent=2)
        self.debug_fmt.debug(f"Размер ядра размытия: {blur_kernel}", indent=2)

        # Детальное объяснение логики настройки
        if self.scan_analysis:
            self._log_sharpening_decision_logic(strength, blur_kernel)

        # Сохранение для использования в детальном логировании
        self._current_parameters = {
            'sharpening_strength': strength,
            'blur_kernel_size': blur_kernel
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
                f"Ошибка выполнения увеличения резкости: {error}", indent=1)

        return StrategyResult(
            strategy_name=self.name,
            success=False,
            result_data=None,
            metrics={},
            processing_time=execution_time,
            error_message=str(error)
        )
