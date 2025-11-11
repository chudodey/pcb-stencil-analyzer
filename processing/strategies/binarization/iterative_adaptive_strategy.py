# processing/strategies/binarization/iterative_adaptive_strategy.py
"""
Стратегия итеративного подбора параметров с единообразным выводом.

СООТВЕТСТВИЕ ИНСТРУКЦИИ:
- Уровень 4 в иерархии логирования (наследник BaseBinarizationStrategy)
- Логирует детали выполнения КОНКРЕТНОЙ итеративной стратегии
- Объясняет логику итеративного подбора параметров и критерии выбора
- Детально описывает процесс тестирования различных значений C
"""

import time
from typing import Any, Dict, List

import cv2
import numpy as np

from .base_binarization import BaseBinarizationStrategy


class IterativeAdaptiveStrategy(BaseBinarizationStrategy):
    """
    Стратегия итеративного подбора параметров адаптивной бинаризации.

    Основная ответственность:
    - Систематический перебор параметров для нахождения оптимальных значений
    - Оценка качества каждой итерации с комплексными метриками
    - Объяснение критериев выбора лучшего результата
    - Детальное логирование процесса итеративного поиска
    """

    def __init__(self, name: str = "IterativeAdaptive", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        self.C_VALUES_TO_TEST = self.config.get('c_values', [
            8, 6, 4, 3, 2.5, 2.0, 1.5, 1.0, 0.5, 0, -0.5, -1, -2
        ])

    def _binarize_image(self, image: np.ndarray, context: Dict[str, Any]) -> tuple:
        """
        Основной метод итеративной бинаризации с подбором параметров.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия объясняет логику итеративного подбора параметров
        - Логирует процесс тестирования различных значений C
        - Объясняет критерии выбора лучшего результата

        Args:
            image: Входное изображение для бинаризации
            context: Контекст выполнения с настройками и анализом

        Returns:
            tuple: (binary_image, parameters_dict)
        """
        debug_mode = context.get('debug_mode', False)

        if debug_mode:
            self.debug_fmt.debug("Начало итеративной стратегии", indent=2)
            self.debug_fmt.debug(
                f"Размер изображения: {image.shape}", indent=3)
            self.debug_fmt.debug(
                f"Тестируемых значений C: {len(self.C_VALUES_TO_TEST)}", indent=3)

        # Получение калибровочных параметров для адаптивной настройки
        calibration = self._get_calibration_parameters(context)
        mean_feature_size = calibration['mean_size']
        complexity_level = calibration['complexity_level']

        if debug_mode:
            self.debug_fmt.debug(
                f"Средний размер объектов: {mean_feature_size:.1f}", indent=3)
            self.debug_fmt.debug(
                f"Уровень сложности: {complexity_level:.3f}", indent=3)

        # 1. РАСЧЕТ БАЗОВЫХ ПАРАМЕТРОВ
        if debug_mode:
            self.debug_fmt.debug("Расчет базовых параметров...", indent=3)

        img_height, img_width = image.shape[:2]
        mean_obj_size = calibration['mean_size']
        obj_diameter = np.sqrt(mean_obj_size) * 2
        block_size = max(11, min(151, int(obj_diameter * 2.5) | 1))

        if debug_mode:
            # КОМПАКТНАЯ ТАБЛИЦА базовых параметров
            base_params = {
                'composite_score': 0.0,  # ← ГАРАНТИРОВАННО в корне
                'Размер изображения': f"{img_width}×{img_height}",
                'Тестируемых C': len(self.C_VALUES_TO_TEST),
                'Средний размер объектов': f"{mean_obj_size:.1f}",
                'Диаметр объекта': f"{obj_diameter:.1f}",
                'Размер блока': block_size
            }
            self.debug_fmt.metrics_table(
                "Базовые параметры итерации", base_params, indent=4)

        # 2. ИТЕРАТИВНЫЙ ПОДБОР ПАРАМЕТРОВ
        if debug_mode:
            self.debug_fmt.debug(
                "Начало итеративного подбора параметров...", indent=3)
            self.debug_fmt.debug(
                f"Тестируемые значения C: {self.C_VALUES_TO_TEST}", indent=4)

        best_result = {
            'score': -1.0,
            'contours': [],
            'binary': None,
            'C': 0,
            'metrics': {},
            'raw_count': 0,
            'filtered_count': 0
        }

        iteration_results = []
        iterations_table_data = []  # Данные для компактной таблицы

        # 3. ВЫПОЛНЕНИЕ ИТЕРАЦИЙ
        for i, C in enumerate(self.C_VALUES_TO_TEST):
            # Бинаризация с текущим C
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, C
            )

            # Поиск и фильтрация контуров
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            raw_count = len(contours)
            contours_list = self._convert_contours_to_list(contours)

            filtered_contours = self._realistic_contour_filtering(
                contours_list, calibration, debug=False
            )

            # Расчет метрик качества
            metrics = self._calculate_realistic_metrics(
                filtered_contours, binary
            )
            score = metrics['composite_score']

            # Определяем лучший результат
            is_best = score > best_result['score']
            if is_best:
                best_result = {
                    'score': score,
                    'contours': filtered_contours,
                    'binary': binary,
                    'C': C,
                    'metrics': metrics,
                    'raw_count': raw_count,
                    'filtered_count': len(filtered_contours)
                }

            # Сохранение результатов итерации
            iteration_result = {
                'C': C,
                'raw_count': raw_count,
                'filtered_count': len(filtered_contours),
                'score': score,
                'component_count': metrics['component_count'],
                'size_uniformity': metrics['size_uniformity'],
                'binary': binary,
                'contours': filtered_contours,
                'metrics': metrics,
                'is_best': is_best
            }
            iteration_results.append(iteration_result)

            # Данные для компактной таблицы
            table_row = {
                'C': C,
                'Сырые': raw_count,
                'Фильтр': len(filtered_contours),
                'Score': score,
                'Компоненты': metrics['component_count'],
                'Равномерность': metrics['size_uniformity'],
                'is_best': is_best
            }
            iterations_table_data.append(table_row)

        # 4. ВЫВОД КОМПАКТНОЙ ТАБЛИЦЫ ВСЕХ ИТЕРАЦИЙ
        if debug_mode and iterations_table_data:
            self.debug_fmt.debug("Результаты итеративного подбора:", indent=3)

            # Используем новую таблицу итераций для компактного вывода
            self.debug_fmt.iteration_table(
                "Итерации подбора параметра C",
                iterations_table_data,
                key_columns=['C', 'Сырые', 'Фильтр',
                             'Score', 'Компоненты', 'Равномерность'],
                indent=4
            )

            # Общая статистика по всем итерациям
            total_stats = {
                # ← ГАРАНТИРОВАННО в корне
                'composite_score': best_result['score'],
                'Всего итераций': len(iterations_table_data),
                'Успешных итераций': len([iter_data for iter_data in iterations_table_data if iter_data['Score'] > 0]),
                'Лучшее C': f"{best_result['C']:.1f}",
                'Макс. Score': f"{best_result['score']:.3f}",
                'Мин. Score': f"{min(iter_data['Score'] for iter_data in iterations_table_data):.3f}",
                'Средний Score': f"{np.mean([iter_data['Score'] for iter_data in iterations_table_data]):.3f}"
            }
            self.debug_fmt.metrics_table(
                "Статистика итераций", total_stats, indent=4)

        # 5. ВЫБОР И ОБОСНОВАНИЕ ЛУЧШЕГО РЕЗУЛЬТАТА
        if debug_mode:
            self.debug_fmt.debug("Выбор лучшего результата...", indent=3)

            best_info = {
                # ← ГАРАНТИРОВАННО в корне
                'composite_score': best_result['score'],
                'Лучшее C': f"{best_result['C']:.1f}",
                'Лучший score': f"{best_result['score']:.3f}",
                'Отфильтрованных контуров': best_result['filtered_count'],
                'Компонентов': best_result['metrics']['component_count'],
                'Равномерность размеров': f"{best_result['metrics']['size_uniformity']:.3f}",
                'Консистентность форм': f"{best_result['metrics']['shape_consistency']:.3f}",
                'Уровень шума': f"{best_result['metrics']['noise_ratio']:.3f}"
            }
            self.debug_fmt.metrics_table(
                "Лучший результат итераций", best_info, indent=4)

        # Параметры для логирования
        binarization_params = {
            'iterative_best_C': best_result['C'],
            'iterative_block_size': block_size,
            'iterations_count': len(self.C_VALUES_TO_TEST),
            'best_score': best_result['score'],
            'filtered_contours_count': best_result['filtered_count'],
            'tested_C_values': self.C_VALUES_TO_TEST,
            'mean_feature_size': mean_feature_size,
            'complexity_level': complexity_level
        }

        return best_result['binary'], binarization_params

    def _calculate_realistic_metrics(self, contours: list, binary_image: np.ndarray) -> Dict[str, Any]:
        """
        Расчет метрик качества для итеративной стратегии.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия объясняет расчет своих специфичных метрик качества
        - Учитывает особенности итеративного подхода в оценке

        Args:
            contours: Найденные контуры
            binary_image: Бинаризованное изображение
            calibration: Калибровочные параметры

        Returns:
            Dict[str, Any]: Метрики качества бинаризации
        """
        # Базовые метрики из родительского класса
        base_metrics = super()._calculate_realistic_metrics(contours, binary_image)

        # Дополнительные метрики специфичные для итеративной стратегии
        if contours and 'contour_metrics' in base_metrics:
            # Оценка стабильности результатов итеративного подхода
            areas = [cv2.contourArea(c)
                     for c in contours if cv2.contourArea(c) > 0]
            if areas:
                area_cv = np.std(areas) / \
                    np.mean(areas) if np.mean(areas) > 0 else 1.0
                stability_score = 1.0 - min(1.0, area_cv)

                # Итеративный подход должен давать более стабильные результаты
                base_metrics['contour_metrics']['size_stability'] = stability_score

                # Модифицируем композитный score с учетом стабильности
                stability_bonus = stability_score * 0.1
                base_metrics['contour_metrics']['composite_score'] = min(1.0,
                                                                         base_metrics['contour_metrics']['composite_score'] + stability_bonus)
                base_metrics['contour_metrics']['iterative_stability'] = stability_score

        return base_metrics

    def _log_binarization_details(self, original: np.ndarray, binary: np.ndarray,
                                  contours: list, metrics: Dict[str, Any],
                                  binarization_params: Dict[str, Any],
                                  execution_time: float) -> None:
        """
        Детальное логирование процесса итеративной бинаризации.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия объясняет процесс итеративного подбора параметров
        - Логирует критерии выбора лучшего результата
        - Показывает преимущества итеративного подхода

        Args:
            original: Исходное изображение
            binary: Бинаризованное изображение
            contours: Найденные контуры
            metrics: Метрики качества
            binarization_params: Параметры бинаризации
            execution_time: Время выполнения
        """
        self.debug_fmt.debug("Детали итеративной стратегии:", indent=1)

        # Логирование параметров итеративного процесса
        iterative_params = {
            'Количество итераций': binarization_params['iterations_count'],
            'Лучшее C': f"{binarization_params['iterative_best_C']:.1f}",
            'Размер блока': binarization_params['iterative_block_size'],
            'Лучший score': f"{binarization_params['best_score']:.3f}",
            'Отфильтрованных контуров': binarization_params['filtered_contours_count'],
            'Ср. размер объектов': f"{binarization_params['mean_feature_size']:.1f}",
            'Уровень сложности': f"{binarization_params['complexity_level']:.3f}"
        }
        self.debug_fmt.metrics_table(
            "Параметры итеративного процесса", iterative_params, indent=2)

        # Специфичные метрики итеративной стратегии
        if 'contour_metrics' in metrics and 'iterative_stability' in metrics['contour_metrics']:
            iterative_metrics = {
                'Стабильность размеров': f"{metrics['contour_metrics']['iterative_stability']:.3f}",
                'Композитный score': f"{metrics['contour_metrics']['composite_score']:.3f}",
                'Время выполнения': f"{execution_time:.3f}с",
                'Время на итерацию': f"{(execution_time / binarization_params['iterations_count']):.3f}с"
            }
            self.debug_fmt.metrics_table(
                "Метрики итеративной стратегии", iterative_metrics, indent=2)

        # Логирование тестируемых значений C
        self.debug_fmt.debug(
            f"Тестируемые значения C: {binarization_params['tested_C_values']}",
            indent=2
        )
