# processing/strategies/binarization/base_binarization.py
"""
Базовый класс для стратегий бинаризации с единообразным выводом.

BaseBinarizationStrategy (Исполнитель бинаризации) отвечает за:
- Адаптивную бинаризации изображений на основе анализа
- Расчет метрик качества бинаризации (контуры, формы, шум)
- Создание "легких" метрик для следующих этапов
- Детальное логирование процесса адаптивной бинаризации

СООТВЕТСТВИЕ ИНСТРУКЦИИ:
- Уровень 4 в иерархии логирования (наследник BinarizationStrategy)
- Логирует детали выполнения КОНКРЕТНОЙ стратегии бинаризации
- Объясняет адаптивные настройки параметров на основе анализа
"""

import time
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from ..base_strategies import BinarizationStrategy, StrategyResult


class BaseBinarizationStrategy(BinarizationStrategy):
    """
    Базовый класс для адаптивных стратегий бинаризации изображений.

    Основная ответственность:
    - Адаптивная настройка параметров бинаризации на основе анализа изображения
    - Расчет метрик качества бинаризации для оценки эффективности
    - Создание "легких" метрик для передачи следующему этапу обработки
    - Детальное логирование процесса адаптивной бинаризации
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация адаптивной стратегии бинаризации.

        Args:
            name: Уникальное имя стратегии для идентификации
            config: Конфигурация стратегии с базовыми параметрами обработки
        """
        super().__init__(name, config or {})
        self.scan_analysis = None
        self.reference_analysis = None
        self.recommendations = None

    def execute(self, input_data: Any, context: Dict[str, Any]) -> StrategyResult:
        """
        Основная логика выполнения адаптивной бинаризации.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия бинаризации логирует свой специфичный процесс
        - Объясняет адаптивные настройки параметров на основе анализа
        - Детально описывает расчет метрик качества бинаризации

        Args:
            input_data: Входное изображение для обработки (numpy array)
            context: Контекст выполнения с global_analysis и настройками

        Returns:
            StrategyResult: Результат бинаризации с метриками качества и контурами
        """
        debug_mode = context.get('debug_mode', False)
        start_time = time.time()

        if debug_mode:
            self.debug_fmt.debug("Начало адаптивной бинаризации", indent=1)
            self.debug_fmt.debug(
                f"Тип входных данных: {type(input_data)}", indent=2)
            if hasattr(input_data, 'shape'):
                self.debug_fmt.debug(
                    f"Размер изображения: {input_data.shape}", indent=2)

        try:
            # 1. КОНФИГУРАЦИЯ НА ОСНОВЕ ГЛОБАЛЬНОГО АНАЛИЗА
            global_analysis = context.get('global_analysis', {})
            self._configure_from_analysis(global_analysis)

            if debug_mode:
                self.debug_fmt.debug(
                    "Конфигурация на основе анализа завершена", indent=2)

            # 2. ИЗВЛЕЧЕНИЕ И ПОДГОТОВКА ИЗОБРАЖЕНИЯ
            image = self._extract_input_image(input_data)

            # 3. ВЫПОЛНЕНИЕ АДАПТИВНОЙ БИНАРИЗАЦИИ
            binary_image, binarization_params = self._binarize_image(
                image, context)

            if debug_mode:
                self.debug_fmt.debug("Бинаризация выполнена успешно", indent=2)

            # 4. ПОИСК И ФИЛЬТРАЦИЯ КОНТУРОВ
            contours = self._find_and_filter_contours(binary_image, context)

            if debug_mode:
                self.debug_fmt.debug(
                    f"Найдено контуров: {len(contours)}", indent=2)

            # 5. РАСЧЕТ МЕТРИК КАЧЕСТВА С ГАРАНТИЕЙ composite_score
            metrics = self._create_metrics_with_guaranteed_composite(
                image, binary_image, contours, context)

            execution_time = time.time() - start_time

            # 6. ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ ПРОЦЕССА БИНАРИЗАЦИИ
            if debug_mode:
                self._log_binarization_details(
                    image, binary_image, contours, metrics,
                    binarization_params, execution_time
                )

            return StrategyResult(
                strategy_name=self.name,
                success=True,
                result_data={
                    'binary_image': binary_image,
                    'contours': contours,
                    'visualization': self._create_contours_visualization(binary_image, contours)
                },
                metrics=metrics,
                processing_time=execution_time,
                error_message=None
            )

        except Exception as error:
            execution_time = time.time() - start_time
            return self._create_error_result(error, execution_time, debug_mode)

    def _configure_from_analysis(self, global_analysis: Dict[str, Any]) -> None:
        """
        Настройка параметров стратегии на основе глобального анализа изображений.

        Args:
            global_analysis: Результаты комплексного анализа скана и эталона из контекста
        """
        self.scan_analysis = global_analysis.get('scan_analysis')
        self.reference_analysis = global_analysis.get('reference_analysis')
        self.recommendations = global_analysis.get('recommendations', {})

    def _binarize_image(self, image: np.ndarray, context: Dict[str, Any]) -> tuple:
        """
        Основной метод бинаризации изображения (должен быть переопределен в дочерних классах).

        Args:
            image: Входное изображение для бинаризации
            context: Контекст выполнения с настройками и анализом

        Returns:
            tuple: (binary_image, parameters_dict)

        Raises:
            NotImplementedError: Если метод не переопределен в дочернем классе
        """
        raise NotImplementedError(
            "Метод _binarize_image должен быть переопределен в дочернем классе")

    def _find_and_filter_contours(self, binary_image: np.ndarray, context: Dict[str, Any]) -> List[np.ndarray]:
        """
        Поиск и фильтрация контуров на бинаризованном изображении.

        Args:
            binary_image: Бинаризованное изображение
            context: Контекст выполнения

        Returns:
            List[np.ndarray]: Отфильтрованные контуры
        """
        debug_mode = context.get('debug_mode', False)

        # Поиск контуров
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = self._convert_contours_to_list(contours)

        if debug_mode:
            self.debug_fmt.debug(
                f"Найдено контуров до фильтрации: {len(contours)}", indent=2)

        # Получение калибровочных параметров
        calibration = self._get_calibration_parameters(context)

        # Фильтрация контуров
        filtered_contours = self._realistic_contour_filtering(
            contours, calibration, debug_mode
        )

        return filtered_contours

    def _get_calibration_parameters(self, context: Dict[str, Any]) -> Dict[str, float]:
        debug_mode = context.get('debug_mode', False)

        if debug_mode:
            self.debug_fmt.debug(
                "Расчет калибровочных параметров...", indent=2)

        global_analysis = context.get('global_analysis', {})
        reference_analysis = global_analysis.get('reference_analysis', {})
        component_sizes = reference_analysis.get('component_sizes', {})
        geometric_features = reference_analysis.get('geometric_features', {})

        # ЗАЩИТА ОТ НЕРЕАЛИСТИЧНЫХ ЗНАЧЕНИЙ
        min_size = max(50.0, component_sizes.get(
            'min_size', 50.0))  # МИНИМУМ 50px
        max_size = max(1000.0, component_sizes.get('max_size', 1000.0))
        mean_size = max(100.0, component_sizes.get(
            'mean_size', 200.0))  # МИНИМУМ 100px

        mean_circularity = max(
            0.3, geometric_features.get('mean_circularity', 0.65))

        # Адаптивные допуски
        size_tolerance = 0.5
        circularity_tolerance = 0.4

        calibration = {
            'min_area': min_size * (1 - size_tolerance),
            'max_area': max_size * (1 + size_tolerance),
            'mean_size': mean_size,
            'min_circularity': max(0.2, mean_circularity * (1 - circularity_tolerance)),
            'expected_circularity': mean_circularity,
            'complexity_level': reference_analysis.get('complexity_score', 0.5)
        }

        if debug_mode:
            calib_data = {
                'Минимальная площадь': f"{calibration['min_area']:.1f}",
                'Максимальная площадь': f"{calibration['max_area']:.1f}",
                'Средний размер': f"{calibration['mean_size']:.1f}",
                'Минимальная циркулярность': f"{calibration['min_circularity']:.3f}",
                'Уровень сложности': f"{calibration['complexity_level']:.3f}"
            }
            self.debug_fmt.metrics_table(
                "Калибровочные параметры", calib_data, indent=3)

        return calibration

    def _realistic_contour_filtering(
        self,
        contours: List[np.ndarray],
        calibration: Dict[str, float],
        debug: bool = False
    ) -> List[np.ndarray]:
        if debug:
            self.debug_fmt.debug(
                f"Фильтрация контуров (входных: {len(contours)})", indent=2)

        min_area = calibration['min_area']
        max_area = calibration['max_area']
        min_circularity = calibration['min_circularity']
        mean_size = calibration.get('mean_size', 100.0)

        # АДАПТИВНЫЕ ПОРОГИ ДЛЯ БОЛЕЕ АГРЕССИВНОЙ ФИЛЬТРАЦИИ
        adaptive_min_area = max(50.0, min_area * 0.5)  # Повысил с 0.1
        adaptive_max_area = max_area * 2.0  # Понизил с 3.0

        # ПОРОГ ДЛЯ МЕЛКОГО ШУМА
        noise_threshold = max(30.0, mean_size * 0.1)

        filtered_contours = []
        statistics = {
            'too_small': 0,
            'too_large': 0,
            'bad_shape': 0,
            'noise': 0,
            'good': 0
        }

        for contour in contours:
            area = cv2.contourArea(contour)

            # БОЛЕЕ АГРЕССИВНАЯ ФИЛЬТРАЦИЯ
            if area < noise_threshold:
                statistics['noise'] += 1
                continue
            elif area < adaptive_min_area:
                statistics['too_small'] += 1
                continue
            elif area > adaptive_max_area:
                statistics['too_large'] += 1
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)

                # АДАПТИВНЫЙ ПОРОГ ЦИРКУЛЯРНОСТИ
                adaptive_circularity = min_circularity
                if area < min_area:
                    adaptive_circularity *= 0.6  # Более строгий
                elif area > max_area * 0.5:
                    adaptive_circularity *= 0.8

                if circularity >= adaptive_circularity:
                    filtered_contours.append(contour)
                    statistics['good'] += 1
                else:
                    statistics['bad_shape'] += 1
            else:
                if area > adaptive_min_area:
                    filtered_contours.append(contour)
                    statistics['good'] += 1

        if debug:
            filter_stats = {
                'Оставлено контуров': statistics['good'],
                'Отфильтровано шума': statistics['noise'],
                'Слишком мелких': statistics['too_small'],
                'Слишком крупных': statistics['too_large'],
                'Отклонено по форме': statistics['bad_shape'],
                'Процент отфильтровано': f"{(1 - statistics['good']/len(contours))*100:.1f}%"
            }
            self.debug_fmt.metrics_table(
                "Статистика фильтрации", filter_stats, indent=3)

        return filtered_contours

    def _create_metrics_with_guaranteed_composite(
        self,
        original: np.ndarray,
        binary: np.ndarray,
        contours: List[np.ndarray],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Создание комплексных метрик качества бинаризации с ГАРАНТИЕЙ composite_score.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия объясняет расчет своих метрик качества
        - Создает "легкие" метрики для передачи контекста следующему этапу
        - ГАРАНТИРУЕТ наличие composite_score в корне метрик

        Args:
            original: Исходное изображение до обработки
            binary: Бинаризованное изображение
            contours: Найденные контуры
            context: Контекст выполнения

        Returns:
            Dict[str, Any]: Словарь с метриками качества бинаризации и composite_score
        """
        debug_mode = context.get('debug_mode', False)

        if debug_mode:
            self.debug_fmt.debug(
                "Расчет метрик качества бинаризации...", indent=2)

        # 1. РАСЧЕТ МЕТРИК КАЧЕСТВА БИНАРИЗАЦИИ
        contour_metrics = self._calculate_realistic_metrics(contours, binary)

        # 2. ВЫПОЛНЕНИЕ "ЛЕГКОГО" АНАЛИЗА ДЛЯ СЛЕДУЮЩЕГО ЭТАПА
        lean_metrics = self._analyze_binary_image_lean(binary, contours)

        # 3. ГАРАНТИЯ: composite_score всегда в корне метрик
        composite_score = contour_metrics.get('composite_score', 0.0)

        # Если composite_score не рассчитан, используем резервный расчет
        if composite_score <= 0:
            composite_score = self._calculate_fallback_composite_score(
                contours, binary, context
            )

        metrics = {
            'composite_score': composite_score,  # ГАРАНТИРОВАННО в корне
            'contour_metrics': contour_metrics,
            'lean_metrics': lean_metrics  # Передача контекста следующему этапу
        }

        if debug_mode:
            self.debug_fmt.debug(
                f"composite_score гарантирован: {composite_score:.3f}", indent=3)
            self.debug_fmt.debug(
                "Метрики качества успешно рассчитаны", indent=3)

        return metrics

    def _calculate_fallback_composite_score(
        self,
        contours: List[np.ndarray],
        binary_image: np.ndarray,
        context: Dict[str, Any]
    ) -> float:
        """
        Резервный расчет composite_score если основной метод не сработал.
        """
        debug_mode = context.get('debug_mode', False)

        if debug_mode:
            self.debug_fmt.debug(
                "Использование резервного расчета composite_score", indent=3)

        try:
            if not contours or binary_image is None:
                return 0.0

            # СУПЕР-ПРОСТАЯ оценка: количество контуров и покрытие
            contour_count = len(contours)
            white_pixels = np.sum(binary_image == 255)
            coverage_ratio = white_pixels / binary_image.size

            # Нормализация количества контуров
            count_score = min(1.0, contour_count / 5000.0)

            # Качество покрытия
            if 0.2 <= coverage_ratio <= 0.6:
                coverage_score = 1.0
            elif 0.15 <= coverage_ratio <= 0.7:
                coverage_score = 0.7
            else:
                coverage_score = 0.3

            fallback_score = 0.6 * count_score + 0.4 * coverage_score

            if debug_mode:
                self.debug_fmt.debug(
                    f"Резервный расчет: count_score={count_score:.3f}, "
                    f"coverage_score={coverage_score:.3f}, final={fallback_score:.3f}",
                    indent=4
                )

            return fallback_score

        except Exception as e:
            if debug_mode:
                self.debug_fmt.debug(
                    f"Ошибка резервного расчета: {e}", indent=4)
            return 0.3

    def _calculate_realistic_metrics(
        self,
        contours: List[np.ndarray],
        binary_image: np.ndarray
    ) -> Dict[str, float]:
        detected_count = len(contours)
        if detected_count == 0:
            return {
                'component_count': 0,
                'size_uniformity': 0.0,
                'shape_consistency': 0.0,
                'noise_ratio': 1.0,
                'coverage_quality': 0.0,
                'composite_score': 0.0  # ГАРАНТИЯ: всегда есть composite_score
            }

        # Анализ размеров
        areas = [cv2.contourArea(c)
                 for c in contours if cv2.contourArea(c) > 0]

        # ДИАГНОСТИКА ДЛЯ ОТЛАДКИ
        debug_mode = getattr(self, 'debug_mode', False)
        if debug_mode and areas:
            self.debug_fmt.debug(f"Диагностика areas:", indent=3)
            self.debug_fmt.debug(f"  Кол-во: {len(areas)}", indent=4)
            self.debug_fmt.debug(f"  Min: {np.min(areas):.1f}", indent=4)
            self.debug_fmt.debug(f"  Max: {np.max(areas):.1f}", indent=4)
            self.debug_fmt.debug(f"  Mean: {np.mean(areas):.1f}", indent=4)
            self.debug_fmt.debug(f"  Std: {np.std(areas):.1f}", indent=4)

        if not areas:
            return {
                'component_count': detected_count,
                'size_uniformity': 0.0,
                'shape_consistency': 0.0,
                'noise_ratio': 1.0,
                'coverage_quality': 0.0,
                'composite_score': 0.0  # ГАРАНТИЯ: всегда есть composite_score
            }

        mean_area = np.mean(areas)
        std_area = np.std(areas)

        # ИСПРАВЛЕННЫЙ РАСЧЕТ SIZE_UNIFORMITY
        if mean_area > 1e-6:
            cv_ratio = std_area / mean_area  # coefficient of variation
            # Преобразуем в оценку [0, 1]: меньше variation = лучше
            if cv_ratio < 0.5:
                size_uniformity = 1.0
            elif cv_ratio < 1.0:
                size_uniformity = 0.7
            elif cv_ratio < 2.0:
                size_uniformity = 0.4
            elif cv_ratio < 5.0:
                size_uniformity = 0.1
            else:
                size_uniformity = 0.05  # Минимальное значение вместо 0
        else:
            size_uniformity = 0.0

        # Анализ форм
        circularities = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0 and area > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                # Ограничиваем сверху
                circularities.append(min(1.0, circularity))

        shape_consistency = 1.0 - \
            np.std(circularities) if circularities else 0.0
        shape_consistency = max(0.0, min(1.0, shape_consistency))

        # Анализ шума - УВЕЛИЧИВАЕМ ПОРОГ
        # Более агрессивная фильтрация
        noise_threshold = max(50.0, mean_area * 0.3)
        noise_count = sum(1 for area in areas if area < noise_threshold)
        noise_ratio = noise_count / len(areas) if areas else 0.0

        # Анализ покрытия
        white_pixels = np.sum(binary_image == 255)
        coverage_ratio = white_pixels / binary_image.size

        if 0.2 <= coverage_ratio <= 0.6:
            coverage_quality = 1.0
        elif 0.1 <= coverage_ratio <= 0.8:
            coverage_quality = 0.7
        else:
            coverage_quality = 0.3

        # Композитная оценка с УСИЛЕННОЙ БОРЬБОЙ С ШУМОМ
        composite_score = (
            0.25 * min(1.0, detected_count / 500) +  # Уменьшил вес количества
            0.25 * size_uniformity +
            0.25 * shape_consistency +
            0.15 * (1.0 - noise_ratio) +  # Увеличил вес борьбы с шумом
            0.10 * coverage_quality
        )

        # ШТРАФ ЗА СЛИШКОМ МНОГО КОНТУРОВ (вероятный шум)
        if detected_count > 10000:
            composite_score *= 0.5  # Сильный штраф
        elif detected_count > 5000:
            composite_score *= 0.7  # Средний штраф
        elif detected_count > 1000:
            composite_score *= 0.9  # Легкий штраф

        return {
            'component_count': detected_count,
            'size_uniformity': max(0.0, min(1.0, size_uniformity)),
            'shape_consistency': shape_consistency,
            'noise_ratio': noise_ratio,
            'coverage_quality': coverage_quality,
            'composite_score': max(0.0, min(1.0, composite_score))  # ГАРАНТИЯ
        }

    def _analyze_binary_image_lean(self, binary_image: np.ndarray, contours: List[np.ndarray]) -> Dict[str, Any]:
        """
        Легкий анализ бинаризованного изображения для следующего этапа.

        Args:
            binary_image: Бинаризованное изображение
            contours: Найденные контуры

        Returns:
            Dict[str, Any]: Легкие метрики для следующего этапа
        """
        return {
            'contour_count': len(contours),
            'coverage_ratio': np.sum(binary_image == 255) / binary_image.size,
            'mean_contour_area': np.mean([cv2.contourArea(c) for c in contours]) if contours else 0.0,
            'binary_image_shape': binary_image.shape
        }

    def _log_binarization_details(
        self,
        original: np.ndarray,
        binary: np.ndarray,
        contours: List[np.ndarray],
        metrics: Dict[str, Any],
        binarization_params: Dict[str, Any],
        execution_time: float
    ) -> None:
        """
        Детальное логирование процесса адаптивной бинаризации.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия объясняет свои адаптивные решения и параметры
        - Логирует метрики качества для понимания эффективности
        - Показывает статистику контуров и результаты фильтрации

        Args:
            original: Исходное изображение до обработки
            binary: Бинаризованное изображение после применения стратегии
            contours: Найденные контуры
            metrics: Рассчитанные метрики качества бинаризации
            binarization_params: Параметры бинаризации
            execution_time: Время выполнения стратегии
        """
        self.debug_fmt.debug("Детали адаптивной бинаризации:", indent=1)

        # Информация об изображениях
        self.debug_fmt.debug(
            f"Исходное: {original.shape} {original.dtype}", indent=2)
        self.debug_fmt.debug(
            f"Бинаризованное: {binary.shape} {binary.dtype}", indent=2)

        # Логирование параметров бинаризации
        if binarization_params:
            self.debug_fmt.debug("Параметры бинаризации:", indent=2)
            for param, value in binarization_params.items():
                self.debug_fmt.debug(f"{param}: {value}", indent=3)

        # Статистика контуров
        contour_stats = {
            'Всего контуров': len(contours),
            'composite_score': f"{metrics.get('composite_score', 0.0):.3f}",
            'Время выполнения': f"{execution_time:.3f}с"
        }
        self.debug_fmt.metrics_table("Статистика", contour_stats, indent=2)

    # ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ====================

    def _extract_input_image(self, input_data: Any) -> np.ndarray:
        """Извлекает изображение из входных данных."""
        if isinstance(input_data, dict):
            image = input_data.get('image', input_data)
        else:
            image = input_data

        if not isinstance(image, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(image)}")

        return image

    def _convert_contours_to_list(self, contours: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Конвертирует Sequence в List."""
        return [np.array(contour) for contour in contours]

    def _create_contours_visualization(
        self,
        binary_image: np.ndarray,
        contours: List[np.ndarray]
    ) -> np.ndarray:
        """Создает визуализацию контуров."""
        if len(binary_image.shape) == 2:
            vis_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = binary_image.copy()

        for i, contour in enumerate(contours):
            color = ((i * 37) % 255, (i * 67) % 255, (i * 97) % 255)
            cv2.drawContours(vis_image, [contour], -1, color, 2)

            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                cv2.putText(vis_image, str(i), (cx-10, cy+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return vis_image

    def _create_error_result(self, error: Exception, execution_time: float,
                             debug_mode: bool = False) -> StrategyResult:
        """
        Создание результата с информацией об ошибке выполнения.

        Args:
            error: Исключение, возникшее при выполнении стратегии
            execution_time: Время выполнения до возникновения ошибки
            debug_mode: Флаг отладочного режима для логирования

        Returns:
            StrategyResult: Результат с информацией об ошибке
        """
        if debug_mode:
            self.debug_fmt.error(
                f"Ошибка выполнения стратегии бинаризации: {error}", indent=1)

        return StrategyResult(
            strategy_name=self.name,
            success=False,
            result_data=None,
            metrics={},
            processing_time=execution_time,
            error_message=str(error)
        )
