# processing/strategies/binarization/adaptive_thresholding_strategy.py
"""
Стратегия адаптивной бинаризации с единообразным выводом.

СООТВЕТСТВИЕ ИНСТРУКЦИИ:
- Уровень 4 в иерархии логирования (наследник BaseBinarizationStrategy)
- Логирует детали выполнения КОНКРЕТНОЙ стратегии адаптивной бинаризации
- Объясняет адаптивные настройки параметров на основе анализа изображения
- Детально описывает процесс адаптивной бинаризации и расчет параметров
"""

import time
from typing import Any, Dict

import cv2
import numpy as np

from .base_binarization import BaseBinarizationStrategy


class AdaptiveThresholdingBinarizationStrategy(BaseBinarizationStrategy):
    """
    Стратегия адаптивной бинаризации с автоматическим подбором параметров.

    Основная ответственность:
    - Адаптивная бинаризация с локальной обработкой разных областей изображения
    - Автоматический расчет параметров на основе характеристик изображения
    - Объяснение логики выбора размера блока и константы вычитания
    """

    def __init__(self, name: str = "AdaptiveThresholding", config: Dict[str, Any] = None):
        super().__init__(name, config or {})

    def _binarize_image(self, image: np.ndarray, context: Dict[str, Any]) -> tuple:
        """
        Основной метод адаптивной бинаризации с автоматическим подбором параметров.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия объясняет логику расчета адаптивных параметров
        - Логирует процесс адаптивной бинаризации с локальной обработкой
        - Объясняет выбор размера блока и константы вычитания на основе анализа

        Args:
            image: Входное изображение для бинаризации
            context: Контекст выполнения с настройками и анализом

        Returns:
            tuple: (binary_image, parameters_dict)
        """
        debug_mode = context.get('debug_mode', False)

        if debug_mode:
            self.debug_fmt.debug("Начало адаптивной бинаризации", indent=2)
            self.debug_fmt.debug(
                f"Размер изображения: {image.shape}", indent=3)
            self.debug_fmt.debug(
                f"Диапазон интенсивностей: [{image.min():.1f}, {image.max():.1f}]", indent=3)

        # Получение калибровочных параметров для адаптивной настройки
        calibration = self._get_calibration_parameters(context)
        mean_feature_size = calibration['mean_size']
        complexity_level = calibration['complexity_level']

        if debug_mode:
            self.debug_fmt.debug(
                f"Средний размер объектов: {mean_feature_size:.1f}", indent=3)
            self.debug_fmt.debug(
                f"Уровень сложности: {complexity_level:.3f}", indent=3)

        # 1. АНАЛИЗ ИЗОБРАЖЕНИЯ ДЛЯ ОПТИМАЛЬНОГО ВЫБОРА ПАРАМЕТРОВ
        if debug_mode:
            self.debug_fmt.debug(
                "Анализ изображения для выбора параметров...", indent=3)

        # Расчет локальной контрастности для определения оптимальной константы C
        local_contrast = self._calculate_local_contrast(image)

        if debug_mode:
            self.debug_fmt.debug(
                f"Локальная контрастность: {local_contrast:.3f}", indent=4)

        # 2. АДАПТИВНЫЙ РАСЧЕТ ПАРАМЕТРОВ БИНАРИЗАЦИИ
        if debug_mode:
            self.debug_fmt.debug(
                "Расчет параметров адаптивной бинаризации...", indent=3)

        # Расчет размера блока на основе среднего размера объектов
        # Блок должен быть достаточно большим для захвата локальных особенностей
        base_block_size = int(np.sqrt(mean_feature_size) * 2.5)
        block_size = max(21, base_block_size | 1)  # Гарантируем нечетность

        # Адаптация константы C на основе локальной контрастности и сложности
        base_C = 1.5
        if local_contrast < 0.1:  # Низкая контрастность
            C = base_C - 0.5
        elif local_contrast > 0.3:  # Высокая контрастность
            C = base_C + 0.5
        else:  # Средняя контрастность
            C = base_C

        # Дополнительная адаптация по уровню сложности
        if complexity_level > 0.7:  # Сложное изображение - более агрессивная обработка
            C += 0.3
        elif complexity_level < 0.3:  # Простое изображение - более мягкая обработка
            C -= 0.3

        C = max(0.5, min(3.0, C))  # Ограничение диапазона

        if debug_mode:
            bin_params = {
                'Базовый размер блока': base_block_size,
                'Финальный размер блока': block_size,
                'Локальная контрастность': f"{local_contrast:.3f}",
                'Базовая константа C': base_C,
                'Адаптированная константа C': f"{C:.2f}",
                'Метод': 'ADAPTIVE_THRESH_GAUSSIAN_C'
            }
            self.debug_fmt.metrics_table(
                "Параметры адаптивной бинаризации", bin_params, indent=4)

        # 3. ВЫПОЛНЕНИЕ АДАПТИВНОЙ БИНАРИЗАЦИИ
        if debug_mode:
            self.debug_fmt.debug(
                "Выполнение адаптивной бинаризации...", indent=3)

        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, C
        )

        if debug_mode:
            self.debug_fmt.debug("Адаптивная бинаризация завершена", indent=3)

            # Анализ результата бинаризации
            white_pixels = np.sum(binary == 255)
            black_pixels = np.sum(binary == 0)
            total_pixels = binary.size
            white_coverage = white_pixels / total_pixels
            black_coverage = black_pixels / total_pixels

            coverage_stats = {
                'Покрытие белым': f"{white_coverage:.3f}",
                'Покрытие черным': f"{black_coverage:.3f}",
                'Баланс': f"{min(white_coverage, black_coverage) / max(white_coverage, black_coverage):.3f}"
            }
            self.debug_fmt.metrics_table(
                "Статистика покрытия после бинаризации", coverage_stats, indent=4)

        # Параметры для логирования
        binarization_params = {
            'adaptive_block_size': block_size,
            'adaptive_C': C,
            'mean_feature_size': mean_feature_size,
            'complexity_level': complexity_level,
            'local_contrast': local_contrast,
            'white_coverage': white_coverage,
            'black_coverage': black_coverage,
            'strategy_type': 'adaptive_thresholding'
        }

        return binary, binarization_params

    def _calculate_local_contrast(self, image: np.ndarray) -> float:
        """
        Расчет локальной контрастности изображения для адаптации параметров.

        Args:
            image: Входное изображение

        Returns:
            float: Мера локальной контрастности (0-1)
        """
        # Используем лапласиан для оценки локальной контрастности
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        local_contrast = np.std(laplacian) / 255.0  # Нормализация к [0,1]
        return min(1.0, local_contrast)

    def _create_metrics_with_guaranteed_composite(
        self,
        original: np.ndarray,
        binary: np.ndarray,
        contours: list,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Создание метрик с дополнительными показателями для адаптивной стратегии.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Использует родительский расчет composite_score
        - Добавляет специфичные для адаптивной стратегии метрики

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

        # ДОБАВЛЯЕМ специфичные метрики для адаптивной стратегии
        if contours:
            # Оценка эффективности адаптивного подхода через консистентность размеров
            areas = [cv2.contourArea(c)
                     for c in contours if cv2.contourArea(c) > 0]

            if areas:
                area_std = np.std(areas)
                area_mean = np.mean(areas)
                size_consistency = 1.0 - (area_std / (area_mean + 1e-6))
                size_consistency = max(0.0, min(1.0, size_consistency))

                # Легкий бонус за хорошую консистентность размеров (максимум +0.1 к оценке)
                adaptive_bonus = min(0.1, size_consistency * 0.1)

                # Обновляем composite_score с учетом бонуса
                current_score = base_metrics.get('composite_score', 0.0)
                base_metrics['composite_score'] = min(
                    1.0, current_score + adaptive_bonus)

                # Добавляем специфичные метрики
                base_metrics['adaptive_effectiveness'] = size_consistency
                base_metrics['size_consistency'] = size_consistency

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
        Детальное логирование специфичное для адаптивной стратегии.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Дополняет родительское логирование специфичными параметрами
        - Объясняет логику выбора адаптивных параметров
        """
        # Сначала вызываем родительское логирование
        super()._log_binarization_details(
            original, binary, contours, metrics, binarization_params, execution_time
        )

        # ДОБАВЛЯЕМ специфичное для адаптивной стратегии логирование
        self.debug_fmt.debug(
            "Специфичные параметры адаптивной стратегии:", indent=2)

        adaptive_params = {
            'Тип стратегии': binarization_params.get('strategy_type', 'adaptive'),
            'Размер блока': binarization_params.get('adaptive_block_size', 'N/A'),
            'Константа C': f"{binarization_params.get('adaptive_C', 0):.2f}",
            'Локальная контрастность': f"{binarization_params.get('local_contrast', 0):.3f}",
            'Эффективность адаптивного подхода': f"{metrics.get('adaptive_effectiveness', 0.0):.3f}",
            'Консистентность размеров': f"{metrics.get('size_consistency', 0.0):.3f}"
        }
        self.debug_fmt.metrics_table(
            "Адаптивные параметры", adaptive_params, indent=3)
