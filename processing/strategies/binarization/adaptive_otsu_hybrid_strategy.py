# processing/strategies/binarization/adaptive_otsu_hybrid_strategy.py
"""
Гибридная стратегия адаптивной бинаризации и морфологических операций.

СООТВЕТСТВИЕ ИНСТРУКЦИИ:
- Уровень 4 в иерархии логирования (наследник BaseBinarizationStrategy)
- Логирует детали выполнения КОНКРЕТНОЙ гибридной стратегии
- Объясняет адаптивные настройки параметров на основе анализа изображения
- Детально описывает процесс гибридной бинаризации и морфологических операций
"""

import time
from typing import Any, Dict

import cv2
import numpy as np

from .base_binarization import BaseBinarizationStrategy


class AdaptiveOtsuHybridBinarizationStrategy(BaseBinarizationStrategy):
    """
    Гибридная стратегия адаптивной бинаризации с морфологической постобработкой.

    Основная ответственность:
    - Адаптивная бинаризация с автоматическим подбором параметров
    - Морфологические операции для улучшения качества контуров
    - Объяснение логики выбора параметров на основе анализа изображения
    """

    def __init__(self, name: str = "AdaptiveOtsuHybrid", config: Dict[str, Any] = None):
        super().__init__(name, config or {})

    def _binarize_image(self, image: np.ndarray, context: Dict[str, Any]) -> tuple:
        """
        Основной метод гибридной бинаризации с адаптивными параметрами.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия объясняет логику расчета адаптивных параметров
        - Логирует процесс бинаризации и морфологических операций
        - Объясняет выбор параметров на основе анализа изображения

        Args:
            image: Входное изображение для бинаризации
            context: Контекст выполнения с настройками и анализом

        Returns:
            tuple: (binary_image, parameters_dict)
        """
        debug_mode = context.get('debug_mode', False)

        if debug_mode:
            self.debug_fmt.debug("Начало гибридной бинаризации", indent=2)
            self.debug_fmt.debug(
                f"Размер изображения: {image.shape}", indent=3)

        # Получение калибровочных параметров для адаптивной настройки
        calibration = self._get_calibration_parameters(context)
        mean_feature_size = calibration['mean_size']
        complexity_level = calibration['complexity_level']

        if debug_mode:
            self.debug_fmt.debug(
                f"Средний размер объектов: {mean_feature_size:.1f}", indent=3)
            self.debug_fmt.debug(
                f"Уровень сложности: {complexity_level:.3f}", indent=3)

        # 1. АДАПТИВНАЯ БИНАРИЗАЦИЯ С АВТОМАТИЧЕСКИМ ПОДБОРОМ ПАРАМЕТРОВ
        if debug_mode:
            self.debug_fmt.debug(
                "Расчет параметров адаптивной бинаризации...", indent=3)

        # Расчет размера блока на основе среднего размера объектов
        block_size = max(21, int(np.sqrt(mean_feature_size) * 2.5) | 1)
        C = 1.5  # Константа вычитания, адаптируется под уровень сложности

        if complexity_level > 0.7:  # Высокая сложность - более агрессивная бинаризация
            C = 2.0
        elif complexity_level < 0.3:  # Низкая сложность - более мягкая бинаризация
            C = 1.0

        if debug_mode:
            bin_params = {
                'Размер блока': block_size,
                'Константа C': C,
                'Метод': 'ADAPTIVE_THRESH_GAUSSIAN_C'
            }
            self.debug_fmt.metrics_table(
                "Параметры адаптивной бинаризации", bin_params, indent=4)

        # Выполнение адаптивной бинаризации
        adaptive_binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, C
        )

        if debug_mode:
            self.debug_fmt.debug(
                "Адаптивная бинаризация завершена", indent=3)
            white_pixels = np.sum(adaptive_binary == 255)
            coverage = white_pixels / adaptive_binary.size
            self.debug_fmt.debug(
                f"Покрытие после бинаризации: {coverage:.3f}", indent=4)

        # 2. МОРФОЛОГИЧЕСКИЕ ОПЕРАЦИИ ДЛЯ УЛУЧШЕНИЯ КАЧЕСТВА
        if debug_mode:
            self.debug_fmt.debug(
                "Применение морфологических операций...", indent=3)

        # Адаптивный выбор размера ядра морфологических операций
        if mean_feature_size < 50:  # Мелкие объекты
            morph_kernel_size = 2
        elif mean_feature_size < 100:  # Средние объекты
            morph_kernel_size = 3
        else:  # Крупные объекты
            morph_kernel_size = 5

        # Создание эллиптического ядра для лучшего сохранения формы
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))

        if debug_mode:
            morph_params = {
                'Размер ядра': morph_kernel_size,
                'Тип ядра': 'MORPH_ELLIPSE',
                'Операции': 'OPEN → CLOSE'
            }
            self.debug_fmt.metrics_table(
                "Параметры морфологических операций", morph_params, indent=4)

        # Последовательное применение открытия и закрытия
        binary_opened = cv2.morphologyEx(
            adaptive_binary, cv2.MORPH_OPEN, kernel)
        binary_final = cv2.morphologyEx(
            binary_opened, cv2.MORPH_CLOSE, kernel)

        if debug_mode:
            self.debug_fmt.debug(
                "Морфологические операции завершены", indent=3)

            # Сравнение покрытия до и после морфологических операций
            white_after_morph = np.sum(binary_final == 255)
            coverage_after = white_after_morph / binary_final.size
            coverage_change = coverage_after - coverage

            morph_stats = {
                'Покрытие до морфологии': f"{coverage:.3f}",
                'Покрытие после морфологии': f"{coverage_after:.3f}",
                'Изменение покрытия': f"{coverage_change:+.3f}"
            }
            self.debug_fmt.metrics_table(
                "Эффект морфологических операций", morph_stats, indent=4)

        # Параметры для логирования
        binarization_params = {
            'adaptive_block_size': block_size,
            'adaptive_C': C,
            'morph_kernel_size': morph_kernel_size,
            'mean_feature_size': mean_feature_size,
            'complexity_level': complexity_level,
            'strategy_type': 'hybrid_adaptive_otsu'
        }

        return binary_final, binarization_params

    def _create_metrics_with_guaranteed_composite(
        self,
        original: np.ndarray,
        binary: np.ndarray,
        contours: list,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Создание метрик с дополнительными показателями для гибридной стратегии.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Использует родительский расчет composite_score
        - Добавляет специфичные для гибридной стратегии метрики

        Args:
            original: Исходное изображение
            binary: Бинаризованное изображение
            contours: Найденные контуры
            context: Контекст выполнения

        Returns:
            Dict[str, Any]: Метрики с гарантированным composite_score
        """
        # Используем родительский метод для гарантированного composite_score
        base_metrics = super()._create_metrics_with_guaranteed_composite(
            original, binary, contours, context
        )

        # ДОБАВЛЯЕМ специфичные метрики для гибридной стратегии
        if contours:
            # Оценка эффективности морфологических операций через solidity
            contour_solidity = []
            for contour in contours:
                area = cv2.contourArea(contour)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area
                    contour_solidity.append(solidity)

            if contour_solidity:
                mean_solidity = np.mean(contour_solidity)
                # Легкий бонус за хорошую форму (максимум +0.1 к оценке)
                morphology_bonus = min(0.1, mean_solidity * 0.1)

                # Обновляем composite_score с учетом бонуса
                current_score = base_metrics.get('composite_score', 0.0)
                base_metrics['composite_score'] = min(
                    1.0, current_score + morphology_bonus)
                base_metrics['morphology_effectiveness'] = mean_solidity

        return base_metrics

    def _log_binarization_details(
        self,
        original: np.ndarray,
        binary: np.ndarray,
        contours: list,
        metrics: Dict[str, Any],
        binarization_params: Dict[str, Any],
        execution_time: float
    ) -> None:
        """
        Детальное логирование специфичное для гибридной стратегии.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Дополняет родительское логирование специфичными параметрами
        - Объясняет логику выбора адаптивных параметров
        """
        # Сначала вызываем родительское логирование
        super()._log_binarization_details(
            original, binary, contours, metrics, binarization_params, execution_time
        )

        # ДОБАВЛЯЕМ специфичное для гибридной стратегии логирование
        self.debug_fmt.debug(
            "Специфичные параметры гибридной стратегии:", indent=2)

        hybrid_params = {
            'Тип стратегии': binarization_params.get('strategy_type', 'hybrid'),
            'Размер блока адапт.': binarization_params.get('adaptive_block_size', 'N/A'),
            'Константа C': binarization_params.get('adaptive_C', 'N/A'),
            'Размер морф. ядра': binarization_params.get('morph_kernel_size', 'N/A'),
            'Эффективность морфологии': f"{metrics.get('morphology_effectiveness', 0.0):.3f}"
        }
        self.debug_fmt.metrics_table(
            "Гибридные параметры", hybrid_params, indent=3)
