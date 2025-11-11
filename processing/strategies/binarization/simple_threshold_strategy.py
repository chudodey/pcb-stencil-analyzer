# processing/strategies/binarization/simple_threshold_strategy.py
"""
Стратегия простой пороговой бинаризации с единообразным выводом.

СООТВЕТСТВИЕ ИНСТРУКЦИИ:
- Уровень 4 в иерархии логирования (наследник BaseBinarizationStrategy)
- Логирует детали выполнения КОНКРЕТНОЙ стратегии простой бинаризации
- Объясняет выбор порогового значения и его обоснование
- Детально описывает ограничения и условия применения простого порога
"""

import time
from typing import Any, Dict

import cv2
import numpy as np

from .base_binarization import BaseBinarizationStrategy


class SimpleThresholdBinarizationStrategy(BaseBinarizationStrategy):
    """
    Стратегия простой пороговой бинаризации с фиксированным порогом.

    Основная ответственность:
    - Применение фиксированного порога для бинаризации изображения
    - Объяснение выбора порогового значения и его обоснование
    - Анализ эффективности простого подхода для текущего изображения
    - Детальное логирование ограничений метода и условий применения
    """

    def __init__(self, name: str = "SimpleThreshold", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        self.threshold_value = self.config.get('threshold_value', 127)

    def _binarize_image(self, image: np.ndarray, context: Dict[str, Any]) -> tuple:
        """
        Основной метод простой пороговой бинаризации.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия объясняет логику выбора порогового значения
        - Логирует анализ изображения для обоснования порога
        - Объясняет ограничения простого подхода и условия применения

        Args:
            image: Входное изображение для бинаризации
            context: Контекст выполнения с настройками и анализом

        Returns:
            tuple: (binary_image, parameters_dict)
        """
        debug_mode = context.get('debug_mode', False)

        if debug_mode:
            self.debug_fmt.debug(
                "Начало простой пороговой бинаризации", indent=2)
            self.debug_fmt.debug(
                f"Размер изображения: {image.shape}", indent=3)
            self.debug_fmt.debug(
                f"Диапазон интенсивностей: [{image.min():.1f}, {image.max():.1f}]", indent=3)

        # 1. АНАЛИЗ ИЗОБРАЖЕНИЯ ДЛЯ ОБОСНОВАНИЯ ПОРОГА
        if debug_mode:
            self.debug_fmt.debug(
                "Анализ изображения для обоснования порога...", indent=3)

        image_stats = self._analyze_image_statistics(image)
        mean_intensity = image_stats['mean']
        std_intensity = image_stats['std']
        median_intensity = image_stats['median']

        # Автоматическая адаптация порога если не задан явно
        if self.threshold_value == 127:  # Значение по умолчанию
            adapted_threshold = self._calculate_adaptive_threshold(image_stats)
            use_adaptive = True
        else:
            adapted_threshold = self.threshold_value
            use_adaptive = False

        if debug_mode:
            stats_data = {
                'Средняя интенсивность': f"{mean_intensity:.1f}",
                'Медианная интенсивность': f"{median_intensity:.1f}",
                'Стандартное отклонение': f"{std_intensity:.1f}",
                'Заданный порог': self.threshold_value,
                'Адаптированный порог': f"{adapted_threshold:.1f}" if use_adaptive else "не используется",
                'Метод выбора': 'адаптивный' if use_adaptive else 'фиксированный'
            }
            self.debug_fmt.metrics_table(
                "Статистика изображения и порог", stats_data, indent=4)

        # 2. ОБОСНОВАНИЕ ВЫБОРА ПОРОГА
        if debug_mode:
            self.debug_fmt.debug("Обоснование выбора порога:", indent=3)

            if use_adaptive:
                self.debug_fmt.debug(
                    f"- Порог адаптирован на основе статистики изображения", indent=4)
                self.debug_fmt.debug(
                    f"- Средняя интенсивность: {mean_intensity:.1f}", indent=4)
                self.debug_fmt.debug(
                    f"- Медианная интенсивность: {median_intensity:.1f}", indent=4)
            else:
                self.debug_fmt.debug(
                    f"- Используется явно заданный порог: {self.threshold_value}", indent=4)

            # Анализ расстояния до среднего значения
            distance_to_mean = abs(adapted_threshold - mean_intensity)
            if distance_to_mean > 50:
                self.debug_fmt.warn(
                    f"Порог значительно отличается от средней интенсивности ({distance_to_mean:.1f})",
                    indent=4
                )

        # 3. ВЫПОЛНЕНИЕ ПРОСТОЙ ПОРОГОВОЙ БИНАРИЗАЦИИ
        if debug_mode:
            self.debug_fmt.debug(
                "Выполнение простой пороговой бинаризации...", indent=3)

        # Применение пороговой бинаризации
        _, binary = cv2.threshold(
            image, adapted_threshold, 255, cv2.THRESH_BINARY
        )

        if debug_mode:
            self.debug_fmt.debug("Пороговая бинаризация завершена", indent=3)

            # Анализ результата бинаризации
            white_pixels = np.sum(binary == 255)
            black_pixels = np.sum(binary == 0)
            total_pixels = binary.size
            white_coverage = white_pixels / total_pixels
            black_coverage = black_pixels / total_pixels

            threshold_results = {
                'Примененный порог': f"{adapted_threshold:.1f}",
                'Покрытие белым': f"{white_coverage:.3f}",
                'Покрытие черным': f"{black_coverage:.3f}",
                'Баланс покрытия': f"{min(white_coverage, black_coverage) / max(white_coverage, black_coverage):.3f}",
                'Эффективность разделения': self._assess_threshold_effectiveness(white_coverage, black_coverage)
            }
            self.debug_fmt.metrics_table(
                "Результаты пороговой бинаризации", threshold_results, indent=4)

            # Оценка качества разделения
            if white_coverage < 0.1 or white_coverage > 0.9:
                self.debug_fmt.warn(
                    f"Сильный дисбаланс покрытия: белое={white_coverage:.3f}, черное={black_coverage:.3f}",
                    indent=4
                )
            elif 0.3 <= white_coverage <= 0.7:
                self.debug_fmt.debug(
                    "Хороший баланс покрытия между объектом и фоном", indent=4)

        # 4. АНАЛИЗ ОГРАНИЧЕНИЙ ПРОСТОГО ПОДХОДА
        if debug_mode:
            self.debug_fmt.debug(
                "Анализ ограничений простого подхода...", indent=3)

            # Проверка на неравномерность освещения
            lighting_variation = self._assess_lighting_variation(image)
            if lighting_variation > 0.3:
                self.debug_fmt.warn(
                    f"Высокая вариация освещения ({lighting_variation:.3f}) - простой порог может быть неэффективен",
                    indent=4
                )

            # Проверка на низкую контрастность
            contrast_ratio = image.max() - image.min()
            if contrast_ratio < 100:
                self.debug_fmt.warn(
                    f"Низкая контрастность ({contrast_ratio:.1f}) - простой порог может дать плохие результаты",
                    indent=4
                )

        # Параметры для логирования
        binarization_params = {
            'threshold_value': adapted_threshold,
            'original_threshold': self.threshold_value,
            'used_adaptive': use_adaptive,
            'mean_intensity': mean_intensity,
            'median_intensity': median_intensity,
            'std_intensity': std_intensity,
            'white_coverage': white_coverage,
            'black_coverage': black_coverage,
            'lighting_variation': lighting_variation,
            'contrast_ratio': contrast_ratio
        }

        return binary, binarization_params

    def _analyze_image_statistics(self, image: np.ndarray) -> Dict[str, float]:
        """
        Анализ статистических характеристик изображения.

        Args:
            image: Входное изображение

        Returns:
            Dict[str, float]: Статистические характеристики
        """
        return {
            'mean': np.mean(image),
            'median': np.median(image),
            'std': np.std(image),
            'min': np.min(image),
            'max': np.max(image),
            'q1': np.percentile(image, 25),
            'q3': np.percentile(image, 75)
        }

    def _calculate_adaptive_threshold(self, image_stats: Dict[str, float]) -> float:
        """
        Расчет адаптивного порога на основе статистики изображения.

        Args:
            image_stats: Статистические характеристики изображения

        Returns:
            float: Адаптивный порог
        """
        mean = image_stats['mean']
        median = image_stats['median']
        std = image_stats['std']

        # Базовый порог - среднее между средним и медианой
        base_threshold = (mean + median) / 2

        # Корректировка на основе вариативности
        if std > 50:  # Высокая вариативность - смещение к медиане
            adaptive_threshold = 0.7 * median + 0.3 * mean
        elif std < 20:  # Низкая вариативность - смещение к среднему
            adaptive_threshold = 0.3 * median + 0.7 * mean
        else:  # Средняя вариативность
            adaptive_threshold = base_threshold

        return max(0, min(255, adaptive_threshold))

    def _assess_threshold_effectiveness(self, white_coverage: float, black_coverage: float) -> str:
        """
        Оценка эффективности выбранного порога.

        Args:
            white_coverage: Покрытие белым
            black_coverage: Покрытие черным

        Returns:
            str: Текстовая оценка эффективности
        """
        balance_ratio = min(white_coverage, black_coverage) / \
            max(white_coverage, black_coverage)

        if balance_ratio > 0.6:
            return "отличная"
        elif balance_ratio > 0.4:
            return "хорошая"
        elif balance_ratio > 0.2:
            return "удовлетворительная"
        else:
            return "плохая"

    def _assess_lighting_variation(self, image: np.ndarray) -> float:
        """
        Оценка вариации освещения по изображению.

        Args:
            image: Входное изображение

        Returns:
            float: Мера вариации освещения (0-1)
        """
        h, w = image.shape
        # Разбиение на регионы
        regions = [
            image[0:h//3, 0:w//3],           # Верхний левый
            image[0:h//3, w//3:2*w//3],      # Верхний центральный
            image[0:h//3, 2*w//3:w],         # Верхний правый
            image[h//3:2*h//3, 0:w//3],      # Центральный левый
            image[h//3:2*h//3, w//3:2*w//3],  # Центральный
            image[h//3:2*h//3, 2*w//3:w],    # Центральный правый
            image[2*h//3:h, 0:w//3],         # Нижний левый
            image[2*h//3:h, w//3:2*w//3],    # Нижний центральный
            image[2*h//3:h, 2*w//3:w]        # Нижний правый
        ]

        region_means = [np.mean(region)
                        for region in regions if region.size > 0]
        if not region_means:
            return 0.0

        # Коэффициент вариации как мера неравномерности освещения
        cv_variation = np.std(region_means) / np.mean(region_means)
        return min(1.0, cv_variation)

    def _calculate_realistic_metrics(self, contours: list, binary_image: np.ndarray) -> Dict[str, Any]:
        """
        Расчет метрик качества для простой пороговой стратегии.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия объясняет расчет своих специфичных метрик качества
        - Учитывает особенности простого порогового подхода в оценке

        Args:
            contours: Найденные контуры
            binary_image: Бинаризованное изображение
            calibration: Калибровочные параметры

        Returns:
            Dict[str, Any]: Метрики качества бинаризации
        """
        # Базовые метрики из родительского класса
        base_metrics = super()._calculate_realistic_metrics(contours, binary_image)

        # Дополнительные метрики специфичные для простой пороговой стратегии
        if contours and 'contour_metrics' in base_metrics:
            # Оценка стабильности простого подхода
            coverage_stability = self._assess_coverage_stability(binary_image)

            # Простой подход должен давать предсказуемые результаты
            base_metrics['contour_metrics']['coverage_stability'] = coverage_stability

            # Модифицируем композитный score с учетом стабильности
            stability_bonus = coverage_stability * 0.1
            base_metrics['contour_metrics']['composite_score'] = min(1.0,
                                                                     base_metrics['contour_metrics']['composite_score'] + stability_bonus)
            base_metrics['contour_metrics']['simple_threshold_stability'] = coverage_stability

        return base_metrics

    def _assess_coverage_stability(self, binary_image: np.ndarray) -> float:
        """
        Оценка стабильности покрытия для простого порогового подхода.

        Args:
            binary_image: Бинаризованное изображение

        Returns:
            float: Мера стабильности покрытия (0-1)
        """
        # Анализ компактности областей (простой порог должен давать компактные области)
        from scipy import ndimage

        labeled_white, n_white = ndimage.label(binary_image == 255)
        labeled_black, n_black = ndimage.label(binary_image == 0)

        total_components = n_white + n_black
        if total_components == 0:
            return 0.0

        # Идеальная стабильность: умеренное количество компактных компонент
        optimal_components = 100  # Эмпирическое значение
        component_ratio = min(1.0, total_components / optimal_components)
        stability = 1.0 - abs(1.0 - component_ratio)

        return max(0.0, min(1.0, stability))

    def _log_binarization_details(self, original: np.ndarray, binary: np.ndarray,
                                  contours: list, metrics: Dict[str, Any],
                                  binarization_params: Dict[str, Any],
                                  execution_time: float) -> None:
        """
        Детальное логирование процесса простой пороговой бинаризации.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия объясняет процесс выбора и применения порога
        - Логирует ограничения и условия применения простого подхода
        - Показывает преимущества простоты и предсказуемости

        Args:
            original: Исходное изображение
            binary: Бинаризованное изображение
            contours: Найденные контуры
            metrics: Метрики качества
            binarization_params: Параметры бинаризации
            execution_time: Время выполнения
        """
        self.debug_fmt.debug("Детали простой пороговой бинаризации:", indent=1)

        # Логирование параметров и статистики
        threshold_params = {
            'Исходный порог': binarization_params['original_threshold'],
            'Примененный порог': f"{binarization_params['threshold_value']:.1f}",
            'Использована адаптация': binarization_params['used_adaptive'],
            'Средняя интенсивность': f"{binarization_params['mean_intensity']:.1f}",
            'Медианная интенсивность': f"{binarization_params['median_intensity']:.1f}",
            'Покрытие белым': f"{binarization_params['white_coverage']:.3f}",
            'Покрытие черным': f"{binarization_params['black_coverage']:.3f}",
            'Вариация освещения': f"{binarization_params['lighting_variation']:.3f}",
            'Контрастность': f"{binarization_params['contrast_ratio']:.1f}"
        }
        self.debug_fmt.metrics_table(
            "Параметры простой бинаризации", threshold_params, indent=2)

        # Специфичные метрики простой стратегии
        if 'contour_metrics' in metrics and 'simple_threshold_stability' in metrics['contour_metrics']:
            simple_metrics = {
                'Стабильность покрытия': f"{metrics['contour_metrics']['simple_threshold_stability']:.3f}",
                'Композитный score': f"{metrics['contour_metrics']['composite_score']:.3f}",
                'Время выполнения': f"{execution_time:.3f}с"
            }
            self.debug_fmt.metrics_table(
                "Метрики простой стратегии", simple_metrics, indent=2)

        # Рекомендации по применению
        if binarization_params['lighting_variation'] > 0.3:
            self.debug_fmt.debug(
                "Рекомендация: использовать адаптивные методы для этого изображения", indent=2)
