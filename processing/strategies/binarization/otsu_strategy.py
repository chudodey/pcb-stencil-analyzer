# processing/strategies/binarization/otsu_strategy.py
"""
Стратегия бинаризации методом Оцу с проверкой применимости и единообразным выводом.

СООТВЕТСТВИЕ ИНСТРУКЦИИ:
- Уровень 4 в иерархии логирования (наследник BaseBinarizationStrategy)
- Логирует детали выполнения КОНКРЕТНОЙ стратегии бинаризации Оцу
- Объясняет анализ гистограммы и проверку применимости метода
- Детально описывает процесс определения порога и критерии бимодальности
"""

import time
from typing import Any, Dict

import cv2
import numpy as np

from .base_binarization import BaseBinarizationStrategy


class OtsuBinarizationStrategy(BaseBinarizationStrategy):
    """
    Стратегия бинаризации методом Оцу с автоматической проверкой применимости.

    Основная ответственность:
    - Анализ гистограммы изображения на предмет бимодальности
    - Автоматическая проверка применимости метода Оцу
    - Расчет оптимального порога методом Оцу с объяснением
    - Детальное логирование критериев применимости и результатов
    """

    def __init__(self, name: str = "Otsu", config: Dict[str, Any] = None):
        super().__init__(name, config or {})

    def is_applicable(self, input_data: Any, context: Dict[str, Any]) -> bool:
        """
        Проверка применимости метода Оцу с детальным логированием.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия объясняет критерии применимости метода Оцу
        - Логирует анализ гистограммы и условия бимодальности
        - Объясняет причины неприменимости метода если они есть

        Args:
            input_data: Входные данные для анализа
            context: Контекст выполнения

        Returns:
            bool: Применим ли метод Оцу для данного изображения
        """
        debug_mode = context.get('debug_mode', False)

        if debug_mode:
            self.debug_fmt.debug(
                "Проверка применимости метода Оцу...", indent=2)

        try:
            image = self._extract_input_image(input_data)

            # Анализ гистограммы для проверки бимодальности
            histogram_analysis = self._analyze_histogram_bimodality(image)
            is_bimodal = histogram_analysis['is_bimodal']
            bimodality_score = histogram_analysis['bimodality_score']

            # Проверка равномерности освещения
            has_uneven_lighting = self._check_uneven_lighting(image)

            if debug_mode:
                applicability_metrics = {
                    'Бимодальность гистограммы': is_bimodal,
                    'Score бимодальности': f"{bimodality_score:.3f}",
                    'Неравномерное освещение': has_uneven_lighting,
                    'Применимость метода Оцу': is_bimodal and not has_uneven_lighting
                }
                self.debug_fmt.metrics_table(
                    "Анализ применимости метода Оцу", applicability_metrics, indent=3)

                if not is_bimodal:
                    self.debug_fmt.debug(
                        "❌ Метод Оцу неприменим: гистограмма не бимодальна", indent=3)
                elif has_uneven_lighting:
                    self.debug_fmt.debug(
                        "❌ Метод Оцу неприменим: обнаружено неравномерное освещение", indent=3)
                else:
                    self.debug_fmt.debug(
                        "✅ Метод Оцу применим: гистограмма бимодальна и освещение равномерное", indent=3)

            return is_bimodal and not has_uneven_lighting

        except Exception as e:
            if debug_mode:
                self.debug_fmt.error(
                    f"Ошибка при проверке применимости метода Оцу: {e}", indent=3)
            return False

    def _binarize_image(self, image: np.ndarray, context: Dict[str, Any]) -> tuple:
        """
        Основной метод бинаризации методом Оцу с анализом гистограммы.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия объясняет процесс расчета порога методом Оцу
        - Логирует анализ гистограммы и критерии бимодальности
        - Объясняет найденное значение порога и его обоснование

        Args:
            image: Входное изображение для бинаризации
            context: Контекст выполнения с настройками и анализом

        Returns:
            tuple: (binary_image, parameters_dict)
        """
        debug_mode = context.get('debug_mode', False)

        if debug_mode:
            self.debug_fmt.debug("Начало бинаризации методом Оцу", indent=2)
            self.debug_fmt.debug(
                f"Размер изображения: {image.shape}", indent=3)
            self.debug_fmt.debug(
                f"Диапазон интенсивностей: [{image.min():.1f}, {image.max():.1f}]", indent=3)

        # 1. ПРЕДВАРИТЕЛЬНЫЙ АНАЛИЗ ГИСТОГРАММЫ
        if debug_mode:
            self.debug_fmt.debug("Анализ гистограммы изображения...", indent=3)

        histogram_analysis = self._analyze_histogram_bimodality(image)
        is_bimodal = histogram_analysis['is_bimodal']
        bimodality_score = histogram_analysis['bimodality_score']
        histogram_peaks = histogram_analysis['peaks']

        if debug_mode:
            hist_params = {
                'Бимодальность': is_bimodal,
                'Score бимодальности': f"{bimodality_score:.3f}",
                'Количество пиков': len(histogram_peaks),
                'Позиции пиков': str([f"{p:.0f}" for p in histogram_peaks])
            }
            self.debug_fmt.metrics_table(
                "Анализ гистограммы", hist_params, indent=4)

            if not is_bimodal:
                self.debug_fmt.warn(
                    "Гистограмма не является строго бимодальной, метод Оцу может быть неоптимален",
                    indent=4
                )

        # 2. ПРОВЕРКА РАВНОМЕРНОСТИ ОСВЕЩЕНИЯ
        if debug_mode:
            self.debug_fmt.debug(
                "Проверка равномерности освещения...", indent=3)

        has_uneven_lighting = self._check_uneven_lighting(image)
        lighting_score = self._calculate_lighting_uniformity(image)

        if debug_mode:
            lighting_metrics = {
                'Неравномерное освещение': has_uneven_lighting,
                'Score равномерности': f"{lighting_score:.3f}",
                'Рекомендация': 'Оцу применим' if not has_uneven_lighting else 'Оцу не рекомендован'
            }
            self.debug_fmt.metrics_table(
                "Анализ освещения", lighting_metrics, indent=4)

        # 3. ВЫПОЛНЕНИЕ БИНАРИЗАЦИИ МЕТОДОМ ОЦУ
        if debug_mode:
            self.debug_fmt.debug(
                "Выполнение бинаризации методом Оцу...", indent=3)

        # Применение метода Оцу
        threshold_value, binary = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        if debug_mode:
            self.debug_fmt.debug("Бинаризация методом Оцу завершена", indent=3)

            # Анализ результата бинаризации
            white_pixels = np.sum(binary == 255)
            black_pixels = np.sum(binary == 0)
            total_pixels = binary.size
            white_coverage = white_pixels / total_pixels
            black_coverage = black_pixels / total_pixels

            otsu_results = {
                'Найденный порог': f"{threshold_value:.1f}",
                'Покрытие белым': f"{white_coverage:.3f}",
                'Покрытие черным': f"{black_coverage:.3f}",
                'Баланс покрытия': f"{min(white_coverage, black_coverage) / max(white_coverage, black_coverage):.3f}",
                'Эффективность разделения': f"{bimodality_score:.3f}"
            }
            self.debug_fmt.metrics_table(
                "Результаты бинаризации Оцу", otsu_results, indent=4)

            # Интерпретация порога
            self.debug_fmt.debug("Интерпретация порога Оцу:", indent=4)
            if threshold_value < 85:
                self.debug_fmt.debug(
                    "- Низкий порог: преобладают темные тона", indent=5)
            elif threshold_value > 170:
                self.debug_fmt.debug(
                    "- Высокий порог: преобладают светлые тона", indent=5)
            else:
                self.debug_fmt.debug(
                    "- Средний порог: сбалансированное изображение", indent=5)

        # Параметры для логирования
        binarization_params = {
            'otsu_threshold': threshold_value,
            'is_bimodal': is_bimodal,
            'bimodality_score': bimodality_score,
            'has_uneven_lighting': has_uneven_lighting,
            'lighting_uniformity': lighting_score,
            'white_coverage': white_coverage,
            'black_coverage': black_coverage,
            'histogram_peaks': histogram_peaks
        }

        return binary, binarization_params

    def _analyze_histogram_bimodality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Детальный анализ гистограммы на предмет бимодальности.

        Args:
            image: Входное изображение

        Returns:
            Dict[str, Any]: Результаты анализа гистограммы
        """
        # Расчет гистограммы
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist.flatten()

        # Нормализация гистограммы
        hist = hist / hist.sum()

        # Поиск пиков в гистограмме
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(hist, height=0.005, distance=20)

        # Сортировка пиков по высоте
        peak_heights = hist[peaks]
        sorted_indices = np.argsort(peak_heights)[::-1]
        significant_peaks = peaks[sorted_indices[:3]]  # Топ-3 пика

        # Оценка бимодальности
        is_bimodal = len(significant_peaks) >= 2

        # Расчет score бимодальности
        if len(significant_peaks) >= 2:
            # Расстояние между двумя главными пиками
            peak_distance = abs(significant_peaks[0] - significant_peaks[1])
            # Относительная высота пиков
            height_ratio = min(peak_heights[sorted_indices[0]], peak_heights[sorted_indices[1]]) / \
                max(peak_heights[sorted_indices[0]],
                    peak_heights[sorted_indices[1]])

            bimodality_score = (peak_distance / 255) * height_ratio
        else:
            bimodality_score = 0.0

        return {
            'is_bimodal': is_bimodal,
            'bimodality_score': bimodality_score,
            'peaks': significant_peaks.tolist(),
            'peak_heights': peak_heights[sorted_indices[:3]].tolist()
        }

    def _check_uneven_lighting(self, image: np.ndarray) -> bool:
        """
        Проверка наличия неравномерного освещения.

        Args:
            image: Входное изображение

        Returns:
            bool: Наличие неравномерного освещения
        """
        # Разбиение изображения на регионы и сравнение яркости
        h, w = image.shape
        regions = [
            image[0:h//2, 0:w//2],      # Верхний левый
            image[0:h//2, w//2:w],      # Верхний правый
            image[h//2:h, 0:w//2],      # Нижний левый
            image[h//2:h, w//2:w]       # Нижний правый
        ]

        region_means = [np.mean(region) for region in regions]
        max_diff = max(region_means) - min(region_means)

        # Порог для определения неравномерного освещения
        return max_diff > 30  # Эмпирический порог

    def _calculate_lighting_uniformity(self, image: np.ndarray) -> float:
        """
        Расчет меры равномерности освещения.

        Args:
            image: Входное изображение

        Returns:
            float: Score равномерности освещения (0-1)
        """
        h, w = image.shape
        # Разбиение на большее количество регионов для точной оценки
        grid_size = 4
        region_means = []

        for i in range(grid_size):
            for j in range(grid_size):
                y_start = i * h // grid_size
                y_end = (i + 1) * h // grid_size
                x_start = j * w // grid_size
                x_end = (j + 1) * w // grid_size

                region = image[y_start:y_end, x_start:x_end]
                region_means.append(np.mean(region))

        # Score равномерности основан на коэффициенте вариации
        mean_brightness = np.mean(region_means)
        if mean_brightness == 0:
            return 0.0

        cv_brightness = np.std(region_means) / mean_brightness
        uniformity = 1.0 - min(1.0, cv_brightness * 2)  # Масштабирование

        return max(0.0, min(1.0, uniformity))

    def _calculate_realistic_metrics(self, contours: list, binary_image: np.ndarray) -> Dict[str, Any]:
        """
        Расчет метрик качества для стратегии Оцу.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия объясняет расчет своих специфичных метрик качества
        - Учитывает особенности метода Оцу в оценке

        Args:
            contours: Найденные контуры
            binary_image: Бинаризованное изображение
            calibration: Калибровочные параметры

        Returns:
            Dict[str, Any]: Метрики качества бинаризации
        """
        # Базовые метрики из родительского класса
        base_metrics = super()._calculate_realistic_metrics(contours, binary_image)

        # Дополнительные метрики специфичные для стратегии Оцу
        if contours and 'contour_metrics' in base_metrics:
            # Оценка эффективности глобального порога
            separation_quality = self._assess_global_threshold_quality(
                binary_image)

            # Метод Оцу должен обеспечивать хорошее глобальное разделение
            base_metrics['contour_metrics']['separation_quality'] = separation_quality

            # Модифицируем композитный score с учетом качества разделения
            separation_bonus = separation_quality * 0.1
            base_metrics['contour_metrics']['composite_score'] = min(1.0,
                                                                     base_metrics['contour_metrics']['composite_score'] + separation_bonus)
            base_metrics['contour_metrics']['otsu_effectiveness'] = separation_quality

        return base_metrics

    def _assess_global_threshold_quality(self, binary_image: np.ndarray) -> float:
        """
        Оценка качества глобального разделения методом Оцу.

        Args:
            binary_image: Бинаризованное изображение

        Returns:
            float: Мера качества разделения (0-1)
        """
        # Анализ компактности белых и черных областей
        from scipy import ndimage

        # Размеры связанных компонент
        white_labels, white_n = ndimage.label(binary_image == 255)
        black_labels, black_n = ndimage.label(binary_image == 0)

        # Идеальное разделение: несколько крупных компактных областей
        total_components = white_n + black_n
        if total_components == 0:
            return 0.0

        # Score основан на количестве компонент (меньше = лучше для глобального порога)
        component_score = 1.0 - min(1.0, total_components / 1000)

        return component_score

    def _log_binarization_details(self, original: np.ndarray, binary: np.ndarray,
                                  contours: list, metrics: Dict[str, Any],
                                  binarization_params: Dict[str, Any],
                                  execution_time: float) -> None:
        """
        Детальное логирование процесса бинаризации методом Оцу.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия объясняет процесс анализа гистограммы и расчета порога
        - Логирует критерии применимости метода Оцу
        - Показывает преимущества глобального подхода

        Args:
            original: Исходное изображение
            binary: Бинаризованное изображение
            contours: Найденные контуры
            metrics: Метрики качества
            binarization_params: Параметры бинаризации
            execution_time: Время выполнения
        """
        self.debug_fmt.debug("Детали бинаризации методом Оцу:", indent=1)

        # Логирование параметров метода Оцу
        otsu_params = {
            'Порог Оцу': f"{binarization_params['otsu_threshold']:.1f}",
            'Бимодальность': binarization_params['is_bimodal'],
            'Score бимодальности': f"{binarization_params['bimodality_score']:.3f}",
            'Неравномерное освещение': binarization_params['has_uneven_lighting'],
            'Равномерность освещения': f"{binarization_params['lighting_uniformity']:.3f}",
            'Покрытие белым': f"{binarization_params['white_coverage']:.3f}",
            'Покрытие черным': f"{binarization_params['black_coverage']:.3f}"
        }
        self.debug_fmt.metrics_table(
            "Параметры метода Оцу", otsu_params, indent=2)

        # Специфичные метрики стратегии Оцу
        if 'contour_metrics' in metrics and 'otsu_effectiveness' in metrics['contour_metrics']:
            otsu_metrics = {
                'Эффективность разделения': f"{metrics['contour_metrics']['otsu_effectiveness']:.3f}",
                'Композитный score': f"{metrics['contour_metrics']['composite_score']:.3f}",
                'Время выполнения': f"{execution_time:.3f}с"
            }
            self.debug_fmt.metrics_table(
                "Метрики стратегии Оцу", otsu_metrics, indent=2)
