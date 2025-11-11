# processing/strategies/binarization/local_adaptive_strategy.py
"""
Стратегия локальной адаптивной бинаризации для больших изображений.

СООТВЕТСТВИЕ ИНСТРУКЦИИ:
- Уровень 4 в иерархии логирования (наследник BaseBinarizationStrategy)
- Логирует детали выполнения КОНКРЕТНОЙ стратегии локальной бинаризации
- Объясняет логику разбиения на тайлы и адаптивные параметры для каждого тайла
- Детально описывает процесс обработки больших изображений по частям
"""

import time
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from .base_binarization import BaseBinarizationStrategy


class LocalAdaptiveStrategy(BaseBinarizationStrategy):
    """
    Стратегия локальной адаптивной бинаризации для обработки больших изображений.

    Основная ответственность:
    - Разбиение больших изображений на тайлы для локальной обработки
    - Адаптивная бинаризация каждого тайла с индивидуальными параметрами
    - Объединение результатов в единое бинаризованное изображение
    - Объяснение логики выбора размера тайлов и адаптивных параметров
    """

    def __init__(self, name: str = "LocalAdaptive", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        self.tile_size = self.config.get('tile_size', 1500)
        # Перекрытие для избежания артефактов
        self.overlap = self.config.get('overlap', 100)

    def _binarize_image(self, image: np.ndarray, context: Dict[str, Any]) -> tuple:
        """
        Основной метод локальной адаптивной бинаризации больших изображений.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия объясняет логику разбиения на тайлы и выбора параметров
        - Логирует процесс обработки каждого тайла с прогрессом
        - Объясняет параметры адаптивной бинаризации для отдельных тайлов

        Args:
            image: Входное изображение для бинаризации
            context: Контекст выполнения с настройками и анализом

        Returns:
            tuple: (binary_image, parameters_dict)
        """
        debug_mode = context.get('debug_mode', False)

        if debug_mode:
            self.debug_fmt.debug(
                "Начало локальной адаптивной бинаризации", indent=2)
            self.debug_fmt.debug(
                f"Размер изображения: {image.shape}", indent=3)
            self.debug_fmt.debug(
                f"Размер тайла: {self.tile_size}", indent=3)
            self.debug_fmt.debug(
                f"Перекрытие тайлов: {self.overlap}", indent=3)

        # Получение калибровочных параметров для адаптивной настройки
        calibration = self._get_calibration_parameters(context)
        mean_feature_size = calibration['mean_size']
        complexity_level = calibration['complexity_level']

        if debug_mode:
            self.debug_fmt.debug(
                f"Средний размер объектов: {mean_feature_size:.1f}", indent=3)
            self.debug_fmt.debug(
                f"Уровень сложности: {complexity_level:.3f}", indent=3)

        # 1. ПОДГОТОВКА И АНАЛИЗ ИЗОБРАЖЕНИЯ ДЛЯ ТАЙЛИНГА
        if debug_mode:
            self.debug_fmt.debug(
                "Подготовка к разбиению на тайлы...", indent=3)

        h, w = image.shape[:2]

        # Адаптивный расчет размера тайла на основе размера изображения и объектов
        optimal_tile_size = self._calculate_optimal_tile_size(
            h, w, mean_feature_size)
        tile_h = min(optimal_tile_size, h)
        tile_w = min(optimal_tile_size, w)

        # Расчет перекрытия для избежания артефактов на границах
        overlap = self._calculate_optimal_overlap(mean_feature_size)

        if debug_mode:
            tiling_params = {
                'Размер изображения': f"{w}×{h}",
                'Оптимальный размер тайла': optimal_tile_size,
                'Фактический размер тайла': f"{tile_w}×{tile_h}",
                'Перекрытие тайлов': overlap,
                'Кол-во тайлов по X': f"{(w + tile_w - 1) // tile_w}",
                'Кол-во тайлов по Y': f"{(h + tile_h - 1) // tile_h}",
                'Общее кол-во тайлов': f"{((w + tile_w - 1) // tile_w) * ((h + tile_h - 1) // tile_h)}"
            }
            self.debug_fmt.metrics_table(
                "Параметры разбиения на тайлы", tiling_params, indent=4)

        # 2. ПАРАМЕТРЫ АДАПТИВНОЙ БИНАРИЗАЦИИ ДЛЯ ТАЙЛОВ
        if debug_mode:
            self.debug_fmt.debug(
                "Расчет параметров бинаризации для тайлов...", indent=3)

        # Расчет параметров адаптивной бинаризации для отдельных тайлов
        block_size = max(21, int(np.sqrt(mean_feature_size) * 2.5) | 1)
        C = self._calculate_adaptive_C(complexity_level, mean_feature_size)

        if debug_mode:
            tile_params = {
                'Размер блока': block_size,
                'Константа C': f"{C:.1f}",
                'Метод': 'ADAPTIVE_THRESH_GAUSSIAN_C',
                'Размер ядра Гаусса': 'автоматический'
            }
            self.debug_fmt.metrics_table(
                "Параметры бинаризации тайлов", tile_params, indent=4)

        # 3. ПРОЦЕСС ТАЙЛИНГА И ЛОКАЛЬНОЙ БИНАРИЗАЦИИ
        if debug_mode:
            self.debug_fmt.debug("Начало обработки тайлов...", indent=3)

        binary = np.zeros_like(image)
        total_tiles = ((w + tile_w - 1) // tile_w) * \
            ((h + tile_h - 1) // tile_h)
        processed_tiles = 0

        tile_statistics = {
            'successful': 0,
            'empty': 0,
            'low_contrast': 0
        }

        for y in range(0, h, tile_h - overlap):
            for x in range(0, w, tile_w - overlap):
                y_end = min(y + tile_h, h)
                x_end = min(x + tile_w, w)

                # Корректировка координат с учетом перекрытия
                y_start = max(0, y)
                x_start = max(0, x)

                tile = image[y_start:y_end, x_start:x_end]

                processed_tiles += 1

                if debug_mode:
                    # ВАЖНО: Прогресс-бар должен быть ОТДЕЛЬНО от других сообщений
                    # Выводим прогресс без дополнительных сообщений для чистоты отображения
                    self.debug_fmt.progress(
                        processed_tiles, total_tiles,
                        f"Тайл {processed_tiles}/{total_tiles}",
                        indent=4
                    )

                # Пропуск пустых тайлов
                if tile.size == 0:
                    tile_statistics['empty'] += 1
                    continue

                # Проверка контрастности тайла
                tile_contrast = np.std(tile)
                if tile_contrast < 5:  # Низкоконтрастный тайл
                    tile_statistics['low_contrast'] += 1
                    if debug_mode:
                        self.debug_fmt.debug(
                            f"Пропуск низкоконтрастного тайла (std={tile_contrast:.1f})",
                            indent=5
                        )
                    continue

                # Локальная адаптивная бинаризация тайла
                try:
                    tile_binary = cv2.adaptiveThreshold(
                        tile, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, block_size, C
                    )

                    binary[y_start:y_end, x_start:x_end] = tile_binary
                    tile_statistics['successful'] += 1

                    # if debug_mode:
                    #     self.debug_fmt.debug(
                    #         f"Тайл успешно обработан (контраст: {tile_contrast:.1f})",
                    #         indent=5
                    #     )

                except Exception as e:
                    if debug_mode:
                        self.debug_fmt.warn(
                            f"Ошибка обработки тайла: {e}",
                            indent=5
                        )

        if debug_mode:
            print()  # Завершение прогресс-бара
            self.debug_fmt.debug("Обработка всех тайлов завершена", indent=3)

            # Статистика обработки тайлов
            tile_stats = {
                'Всего тайлов': total_tiles,
                'Успешно обработано': tile_statistics['successful'],
                'Пустых тайлов': tile_statistics['empty'],
                'Низкоконтрастных': tile_statistics['low_contrast'],
                'Процент успеха': f"{(tile_statistics['successful'] / total_tiles * 100):.1f}%"
            }
            self.debug_fmt.metrics_table(
                "Статистика обработки тайлов", tile_stats, indent=4)

        # 4. ПОСТОБРАБОТКА ДЛЯ УСТРАНЕНИЯ АРТЕФАКТОВ ГРАНИЦ
        if debug_mode:
            self.debug_fmt.debug(
                "Постобработка для устранения артефактов...", indent=3)

        # Применение морфологических операций для сглаживания границ
        morph_kernel_size = max(3, int(np.sqrt(mean_feature_size) * 0.5) | 1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))

        binary_smoothed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        if debug_mode:
            morph_stats = {
                'Размер ядра морфологии': morph_kernel_size,
                'Операция': 'MORPH_CLOSE',
                'Назначение': 'устранение артефактов границ'
            }
            self.debug_fmt.metrics_table(
                "Параметры постобработки", morph_stats, indent=4)

        # Параметры для логирования
        binarization_params = {
            'tile_size': optimal_tile_size,
            'overlap': overlap,
            'adaptive_block_size': block_size,
            'adaptive_C': C,
            'total_tiles': total_tiles,
            'successful_tiles': tile_statistics['successful'],
            'empty_tiles': tile_statistics['empty'],
            'low_contrast_tiles': tile_statistics['low_contrast'],
            'morph_kernel_size': morph_kernel_size,
            'mean_feature_size': mean_feature_size,
            'complexity_level': complexity_level
        }

        return binary_smoothed, binarization_params

    def _calculate_optimal_tile_size(self, height: int, width: int, mean_feature_size: float) -> int:
        """
        Расчет оптимального размера тайла на основе характеристик изображения.

        Args:
            height: Высота изображения
            width: Ширина изображения
            mean_feature_size: Средний размер объектов

        Returns:
            int: Оптимальный размер тайла
        """
        # Базовый размер тайла из конфигурации
        base_tile_size = self.tile_size

        # Адаптация под размер изображения
        min_dimension = min(height, width)
        if min_dimension < base_tile_size:
            return min_dimension

        # Адаптация под размер объектов
        if mean_feature_size > 500:  # Крупные объекты - большие тайлы
            return min(base_tile_size * 2, min_dimension)
        elif mean_feature_size < 50:  # Мелкие объекты - меньшие тайлы
            return max(500, base_tile_size // 2)

        return base_tile_size

    def _calculate_optimal_overlap(self, mean_feature_size: float) -> int:
        """
        Расчет оптимального перекрытия тайлов.

        Args:
            mean_feature_size: Средний размер объектов

        Returns:
            int: Оптимальное перекрытие
        """
        base_overlap = self.overlap

        # Адаптация перекрытия под размер объектов
        if mean_feature_size > 200:  # Крупные объекты - большее перекрытие
            return min(base_overlap * 2, 300)
        elif mean_feature_size < 30:  # Мелкие объекты - меньшее перекрытие
            return max(50, base_overlap // 2)

        return base_overlap

    def _calculate_adaptive_C(self, complexity_level: float, mean_feature_size: float) -> float:
        """
        Расчет адаптивной константы C для бинаризации тайлов.

        Args:
            complexity_level: Уровень сложности изображения
            mean_feature_size: Средний размер объектов

        Returns:
            float: Адаптивная константа C
        """
        base_C = 3.0

        # Адаптация по уровню сложности
        if complexity_level > 0.7:  # Сложное изображение
            base_C += 1.0
        elif complexity_level < 0.3:  # Простое изображение
            base_C -= 1.0

        # Адаптация по размеру объектов
        if mean_feature_size < 50:  # Мелкие объекты
            base_C -= 0.5
        elif mean_feature_size > 200:  # Крупные объекты
            base_C += 0.5

        return max(1.0, min(5.0, base_C))

    def _calculate_realistic_metrics(self, contours: list, binary_image: np.ndarray) -> Dict[str, Any]:
        """
        Расчет метрик качества для локальной адаптивной стратегии.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия объясняет расчет своих специфичных метрик качества
        - Учитывает особенности обработки больших изображений

        Args:
            contours: Найденные контуры
            binary_image: Бинаризованное изображение
            calibration: Калибровочные параметры

        Returns:
            Dict[str, Any]: Метрики качества бинаризации
        """
        # Базовые метрики из родительского класса
        base_metrics = super()._calculate_realistic_metrics(contours, binary_image)

        # Дополнительные метрики специфичные для локальной стратегии
        if contours and 'contour_metrics' in base_metrics:
            # Оценка равномерности покрытия по областям изображения
            coverage_quality = self._assess_coverage_uniformity(binary_image)

            # Локальная стратегия должна обеспечивать равномерное покрытие
            base_metrics['contour_metrics']['coverage_uniformity'] = coverage_quality

            # Модифицируем композитный score с учетом равномерности покрытия
            uniformity_bonus = coverage_quality * 0.15
            base_metrics['contour_metrics']['composite_score'] = min(1.0,
                                                                     base_metrics['contour_metrics']['composite_score'] + uniformity_bonus)
            base_metrics['contour_metrics']['local_uniformity'] = coverage_quality

        return base_metrics

    def _assess_coverage_uniformity(self, binary_image: np.ndarray) -> float:
        """
        Оценка равномерности покрытия бинаризации по областям изображения.

        Args:
            binary_image: Бинаризованное изображение

        Returns:
            float: Мера равномерности покрытия (0-1)
        """
        h, w = binary_image.shape
        tile_size = min(500, h // 4, w // 4)  # Меньшие тайлы для анализа

        coverages = []

        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)

                tile = binary_image[y:y_end, x:x_end]
                if tile.size > 0:
                    coverage = np.sum(tile == 255) / tile.size
                    coverages.append(coverage)

        if not coverages:
            return 0.0

        # Равномерность оценивается через коэффициент вариации
        mean_coverage = np.mean(coverages)
        if mean_coverage == 0:
            return 0.0

        cv_coverage = np.std(coverages) / mean_coverage
        uniformity = 1.0 - min(1.0, cv_coverage)

        return uniformity

    def _log_binarization_details(self, original: np.ndarray, binary: np.ndarray,
                                  contours: list, metrics: Dict[str, Any],
                                  binarization_params: Dict[str, Any],
                                  execution_time: float) -> None:
        """
        Детальное логирование процесса локальной адаптивной бинаризации.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия объясняет процесс обработки больших изображений
        - Логирует статистику тайлинга и параметры обработки
        - Показывает преимущества локального подхода

        Args:
            original: Исходное изображение
            binary: Бинаризованное изображение
            contours: Найденные контуры
            metrics: Метрики качества
            binarization_params: Параметры бинаризации
            execution_time: Время выполнения
        """
        self.debug_fmt.debug(
            "Детали локальной адаптивной стратегии:", indent=1)

        # Логирование параметров тайлинга и обработки
        tiling_params = {
            'Размер тайла': binarization_params['tile_size'],
            'Перекрытие тайлов': binarization_params['overlap'],
            'Всего тайлов': binarization_params['total_tiles'],
            'Успешных тайлов': binarization_params['successful_tiles'],
            'Пустых тайлов': binarization_params['empty_tiles'],
            'Низкоконтрастных': binarization_params['low_contrast_tiles'],
            'Размер блока': binarization_params['adaptive_block_size'],
            'Константа C': f"{binarization_params['adaptive_C']:.1f}",
            'Размер морф. ядра': binarization_params['morph_kernel_size']
        }
        self.debug_fmt.metrics_table(
            "Параметры локальной обработки", tiling_params, indent=2)

        # Специфичные метрики локальной стратегии
        if 'contour_metrics' in metrics and 'local_uniformity' in metrics['contour_metrics']:
            local_metrics = {
                'Равномерность покрытия': f"{metrics['contour_metrics']['local_uniformity']:.3f}",
                'Композитный score': f"{metrics['contour_metrics']['composite_score']:.3f}",
                'Время выполнения': f"{execution_time:.3f}с",
                'Время на тайл': f"{(execution_time / binarization_params['total_tiles']):.3f}с"
            }
            self.debug_fmt.metrics_table(
                "Метрики локальной стратегии", local_metrics, indent=2)
