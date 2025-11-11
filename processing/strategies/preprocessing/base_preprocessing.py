# processing/strategies/preprocessing/base_preprocessing.py
"""
Базовый класс для умных стратегий предобработки изображений с адаптацией на основе глобального анализа.

BasePreprocessingStrategy (Исполнитель предобработки) отвечает за:
- Адаптивную предобработку изображений на основе анализа
- Расчет метрик качества предобработки (SNR, контраст, границы)
- Создание "легких" метрик для следующих этапов
- Детальное логирование процесса адаптивной обработки

СООТВЕТСТВИЕ ИНСТРУКЦИИ:
- Уровень 4 в иерархии логирования (наследник ProcessingStrategy)
- Логирует детали выполнения КОНКРЕТНОЙ стратегии предобработки
- Объясняет адаптивные настройки параметров на основе анализа
"""

import time
from typing import Any, Dict, Optional, Tuple

# pylint: disable=no-member
import cv2
import numpy as np

from domain.data_models import PreprocessingMetrics
from ..base_strategies import PreprocessingStrategy, StrategyResult
from ..image_analyzer import ImageAnalyzer


class BasePreprocessingStrategy(PreprocessingStrategy):
    """
    Базовый класс для адаптивных стратегий предобработки изображений.

    Основная ответственность:
    - Адаптивная настройка параметров обработки на основе анализа изображения
    - Расчет метрик качества предобработки для оценки эффективности
    - Создание "легких" метрик для передачи следующему этапу обработки
    - Детальное логирование процесса адаптивной обработки
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация адаптивной стратегии предобработки.

        Args:
            name: Уникальное имя стратегии для идентификации
            config: Конфигурация стратегии с базовыми параметрами обработки
        """
        super().__init__(name, config)
        self.scan_analysis = None
        self.reference_analysis = None
        self.recommendations = None

        # Инициализация анализатора для расчета "легких" метрик
        debug_mode = config.get('debug_mode', False) if config else False
        self.image_analyzer = ImageAnalyzer(debug_mode)

    def execute(self, input_data: Any, context: Dict[str, Any]) -> StrategyResult:
        """
        Основная логика выполнения адаптивной предобработки.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия предобработки логирует свой специфичный процесс
        - Объясняет адаптивные настройки параметров на основе анализа
        - Детально описывает расчет метрик качества

        Args:
            input_data: Входное изображение для обработки (numpy array)
            context: Контекст выполнения с global_analysis и настройками

        Returns:
            StrategyResult: Результат предобработки с метриками качества и обработанным изображением
        """
        debug_mode = context.get('debug_mode', False)
        start_time = time.time()

        if debug_mode:
            self.debug_fmt.debug("Начало адаптивной предобработки", indent=1)
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

            # 2. ВЫПОЛНЕНИЕ АДАПТИВНОЙ ПРЕДОБРАБОТКИ
            processed_image = self._process_image(input_data, context)

            # 3. РАСЧЕТ МЕТРИК КАЧЕСТВА С ГАРАНТИЕЙ composite_score
            metrics = self._create_metrics_with_guaranteed_composite(
                input_data, processed_image, context
            )

            execution_time = time.time() - start_time

            # 4. ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ ПРОЦЕССА ПРЕДОБРАБОТКИ
            if debug_mode:
                self._log_processing_details(
                    input_data, processed_image, metrics, execution_time)

            return StrategyResult(
                strategy_name=self.name,
                success=True,
                # Стандартизированный формат
                result_data={'image': processed_image},
                metrics=metrics,  # ТЕПЕРЬ СЛОВАРЬ!
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

    def _process_image(self, image: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """
        Основной метод обработки изображения (должен быть переопределен в дочерних классах).

        Args:
            image: Входное изображение для обработки
            context: Контекст выполнения с настройками и анализом

        Returns:
            np.ndarray: Обработанное изображение

        Raises:
            NotImplementedError: Если метод не переопределен в дочернем классе
        """
        raise NotImplementedError(
            "Метод _process_image должен быть переопределен в дочернем классе")

    def _create_metrics_with_guaranteed_composite(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Создание метрик предобработки с ГАРАНТИЕЙ composite_score.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия объясняет расчет своих метрик качества
        - Создает "легкие" метрики для передачи контекста следующему этапу
        - ГАРАНТИРУЕТ наличие composite_score в корне метрик

        Args:
            original: Исходное изображение до обработки
            processed: Обработанное изображение после применения стратегии
            context: Контекст выполнения

        Returns:
            Dict[str, Any]: Словарь с метриками качества предобработки и composite_score
        """
        debug_mode = context.get('debug_mode', False)

        if debug_mode:
            self.debug_fmt.debug(
                "Расчет метрик качества предобработки...", indent=1)

        # 1. ВЫПОЛНЕНИЕ "ЛЕГКОГО" АНАЛИЗА ДЛЯ СЛЕДУЮЩЕГО ЭТАПА
        lean_metrics = self.image_analyzer.analyze_image_lean(processed)

        # 2. РАСЧЕТ МЕТРИК КАЧЕСТВА ПРЕДОБРАБОТКИ
        preprocessing_metrics = PreprocessingMetrics(
            contrast_improvement=self._calculate_contrast_improvement(
                original, processed),
            snr_improvement=self._calculate_snr_improvement(
                original, processed),
            edge_preservation=self._calculate_edge_preservation(
                original, processed),
            original_std=float(np.std(original.astype(np.float32))),
            processed_std=float(np.std(processed.astype(np.float32)))
        )

        # 3. ГАРАНТИЯ: composite_score всегда в корне метрик
        composite_score = self._calculate_composite_score(
            preprocessing_metrics)

        # Если composite_score не рассчитан, используем резервный расчет
        if composite_score <= 0:
            composite_score = self._calculate_fallback_composite_score(
                original, processed, context
            )

        metrics = {
            'composite_score': composite_score,  # ГАРАНТИРОВАННО в корне
            'preprocessing_metrics': preprocessing_metrics,
            'lean_metrics': lean_metrics  # Передача контекста следующему этапу
        }

        if debug_mode:
            self.debug_fmt.debug(
                f"composite_score гарантирован: {composite_score:.3f}", indent=2)
            self.debug_fmt.debug(
                "Метрики качества успешно рассчитаны", indent=2)

        return metrics

    def _calculate_composite_score(self, preprocessing_metrics: PreprocessingMetrics) -> float:
        """
        Расчет композитного score для предобработки на основе метрик качества.

        Args:
            preprocessing_metrics: Метрики качества предобработки

        Returns:
            float: Композитный score в диапазоне [0, 1]
        """
        try:
            # Взвешенная сумма метрик качества
            weights = {
                'contrast_improvement': 0.4,
                'snr_improvement': 0.3,
                'edge_preservation': 0.3
            }

            # Нормализация значений метрик к [0, 1]
            contrast_norm = max(
                0.0, min(1.0, (preprocessing_metrics.contrast_improvement - 0.5) * 2.0))
            snr_norm = max(
                0.0, min(1.0, (preprocessing_metrics.snr_improvement - 0.5) * 2.0))
            edge_norm = preprocessing_metrics.edge_preservation

            composite_score = (
                weights['contrast_improvement'] * contrast_norm +
                weights['snr_improvement'] * snr_norm +
                weights['edge_preservation'] * edge_norm
            )

            return max(0.0, min(1.0, composite_score))

        except Exception as error:
            self.debug_fmt.debug(
                f"Ошибка расчета composite_score: {error}", indent=2)
            return 0.0

    def _calculate_fallback_composite_score(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        context: Dict[str, Any]
    ) -> float:
        """
        Резервный расчет composite_score для предобработки.

        Args:
            original: Исходное изображение
            processed: Обработанное изображение
            context: Контекст выполнения

        Returns:
            float: Резервный composite_score
        """
        debug_mode = context.get('debug_mode', False)

        try:
            # Простая оценка на основе улучшения контраста
            original_contrast = np.std(original)
            processed_contrast = np.std(processed)

            if original_contrast > 0:
                contrast_improvement = processed_contrast / original_contrast
                # Нормализуем к [0, 1]
                contrast_score = min(1.0, contrast_improvement / 2.0)
            else:
                contrast_score = 0.5

            fallback_score = contrast_score

            if debug_mode:
                self.debug_fmt.debug(
                    f"Резервный расчет: contrast_score={contrast_score:.3f}, "
                    f"final={fallback_score:.3f}",
                    indent=3
                )

            return fallback_score

        except Exception as e:
            if debug_mode:
                self.debug_fmt.debug(
                    f"Ошибка резервного расчета: {e}", indent=3)
            return 0.3

    def _log_processing_details(self, original: np.ndarray, processed: np.ndarray,
                                metrics: Dict[str, Any], execution_time: float) -> None:
        """
        Детальное логирование процесса адаптивной предобработки.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия объясняет свои адаптивные решения и параметры
        - Логирует метрики качества для понимания эффективности
        - Показывает различия между исходным и обработанным изображением

        Args:
            original: Исходное изображение до обработки
            processed: Обработанное изображение после применения стратегии
            metrics: Рассчитанные метрики качества предобработки
            execution_time: Время выполнения стратегии
        """
        self.debug_fmt.debug("Детали адаптивной обработки:", indent=1)

        # Информация об изображениях
        self.debug_fmt.debug(
            f"Исходное: {original.shape} {original.dtype}", indent=2)
        self.debug_fmt.debug(
            f"Обработанное: {processed.shape} {processed.dtype}", indent=2)

        # Логирование адаптивных параметров если они были рассчитаны
        if hasattr(self, '_current_parameters'):
            self.debug_fmt.debug("Адаптивные параметры:", indent=2)
            for param, value in self._current_parameters.items():
                self.debug_fmt.debug(f"{param}: {value}", indent=3)

        # Детальные метрики качества предобработки
        if 'preprocessing_metrics' in metrics:
            prep_metrics = metrics['preprocessing_metrics']
            quality_metrics = {
                'composite_score': f"{metrics.get('composite_score', 0):.3f}",
                'SNR улучшение': f"{getattr(prep_metrics, 'snr_improvement', 0):.3f}",
                'Сохранение границ': f"{getattr(prep_metrics, 'edge_preservation', 0):.3f}",
                'Улучшение контраста': f"{getattr(prep_metrics, 'contrast_improvement', 0):.3f}",
                'Время выполнения': f"{execution_time:.3f}с"
            }
            self.debug_fmt.metrics_table(
                "Метрики качества предобработки", quality_metrics, indent=2)

    # ==================== АДАПТИВНЫЕ МЕТОДЫ РАСЧЕТА ПАРАМЕТРОВ ====================

    def _calculate_adaptive_kernel_size(self, base_size: int = 3) -> int:
        """
        Адаптивный расчет размера ядра фильтра на основе анализа шума и деталей.

        Args:
            base_size: Базовый размер ядра по умолчанию

        Returns:
            int: Адаптивный размер ядра (всегда нечетный для OpenCV)
        """
        if not self.scan_analysis or not self.reference_analysis:
            kernel_size = self.config.get('kernel_size', base_size)
            return kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        kernel_size = base_size

        # Корректировка на основе уровня шума
        if self.scan_analysis.get('high_noise', False):
            kernel_size += 2  # Увеличиваем для подавления сильного шума

        # Корректировка на основе размера деталей
        if self.reference_analysis.get('has_small_features', False):
            # Уменьшаем для сохранения мелких деталей
            kernel_size = max(3, kernel_size - 2)

        # Корректировка на основе контраста
        if self.scan_analysis.get('low_contrast', False):
            kernel_size += 1  # Увеличиваем для улучшения низкого контраста

        # Учет рекомендаций из анализа
        kernel_recommendation = None
        if self.recommendations:
            kernel_recommendation = self.recommendations.get(
                'parameter_adjustments', {}).get('kernel_sizes')

        if kernel_recommendation == 'small':
            kernel_size = max(3, kernel_size - 1)
        elif kernel_recommendation == 'large':
            kernel_size += 1

        # Гарантия нечетности для OpenCV
        return kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    def _calculate_adaptive_sigma(self, base_sigma: float = 1.0) -> float:
        """
        Адаптивный расчет сигмы для гауссовых фильтров на основе текстуры изображения.

        Args:
            base_sigma: Базовая сигма по умолчанию

        Returns:
            float: Адаптивная сигма с округлением до 2 знаков
        """
        if not self.scan_analysis or not self.reference_analysis:
            return base_sigma

        sigma = base_sigma

        # Корректировка на основе уровня шума
        noise_level = self.scan_analysis.get('noise_level', 0)
        if noise_level > 0.15:
            sigma += 1.0  # Увеличиваем для сильного шума

        # Корректировка на основе размера деталей
        mean_size = self.reference_analysis.get(
            'component_sizes', {}).get('mean_size', 0)
        if mean_size < 50:
            sigma = max(0.5, sigma - 0.5)  # Уменьшаем для мелких деталей

        # Корректировка на основе резкости
        sharpness_ratio = self.scan_analysis.get(
            'gradient_analysis', {}).get('sharpness_ratio', 0)
        if sharpness_ratio < 0.1:
            sigma += 0.5  # Увеличиваем для размытых изображений

        return round(sigma, 2)

    def _calculate_adaptive_strength(self, base_strength: float = 1.5) -> float:
        """
        Адаптивный расчет силы эффекта обработки на основе анализа резкости.

        Args:
            base_strength: Базовая сила эффекта по умолчанию

        Returns:
            float: Адаптивная сила эффекта с округлением до 2 знаков
        """
        if not self.scan_analysis:
            return base_strength

        strength = base_strength

        # Корректировка на основе резкости изображения
        sharpness_ratio = self.scan_analysis.get(
            'gradient_analysis', {}).get('sharpness_ratio', 0)
        if sharpness_ratio < 0.1:
            strength += 0.5  # Усиливаем для размытых изображений
        elif sharpness_ratio > 0.5:
            strength -= 0.5  # Ослабляем для резких изображений

        # Корректировка на основе плотности деталей
        if (self.reference_analysis and
            self.reference_analysis.get('is_high_density', False) and
                self.reference_analysis.get('has_small_features', False)):
            # Увеличиваем для высокой плотности
            strength = min(2.5, strength + 0.5)

        return round(strength, 2)

    def _calculate_adaptive_gamma(self) -> float:
        """
        Адаптивный расчет гамма-коррекции на основе анализа яркости.

        Returns:
            float: Адаптивное значение гаммы для коррекции яркости
        """
        if not self.scan_analysis:
            return self.config.get('gamma', 1.2)

        mean_intensity = self.scan_analysis.get('mean_intensity', 128)

        # Определение гаммы на основе яркости
        if self.scan_analysis.get('is_dark', False):
            gamma = 0.8  # Осветляем темные изображения
        elif self.scan_analysis.get('is_bright', False):
            gamma = 1.8  # Затемняем яркие изображения
        elif mean_intensity < 100:
            gamma = 1.0  # Нейтральная коррекция
        elif mean_intensity > 150:
            gamma = 1.4
        else:
            gamma = 1.2

        # Дополнительная корректировка для высокой плотности
        if (self.reference_analysis and
                self.reference_analysis.get('is_high_density', False)):
            gamma = max(0.7, gamma - 0.1)  # Слегка осветляем

        return round(gamma, 2)

    def _calculate_adaptive_clip_limit(self, base_limit: float = 2.0) -> float:
        """
        Адаптивный расчет clip limit для CLAHE на основе контраста и плотности.

        Args:
            base_limit: Базовый clip limit по умолчанию

        Returns:
            float: Адаптивный clip limit с ограничением до 4.0
        """
        if not self.scan_analysis:
            return base_limit

        clip_limit = base_limit

        # Корректировка на основе контраста
        if self.scan_analysis.get('low_contrast', False):
            clip_limit += 1.5  # Увеличиваем для низкого контраста

        # Корректировка на основе шума
        if self.scan_analysis.get('high_noise', False):
            # Уменьшаем чтобы не усиливать шум
            clip_limit = max(1.0, clip_limit - 1.0)

        # Корректировка на основе плотности
        if (self.reference_analysis and
                self.reference_analysis.get('is_high_density', False)):
            clip_limit += 0.5  # Увеличиваем для высокой плотности

        # Ограничение максимального значения
        return min(round(clip_limit, 2), 4.0)

    def _calculate_adaptive_tile_size(self) -> Tuple[int, int]:
        """
        Адаптивный расчет размера тайлов для CLAHE на основе регулярности структуры.

        Returns:
            Tuple[int, int]: Кортеж (rows, cols) с адаптивным размером тайлов
        """
        if (self.reference_analysis and
                self.reference_analysis.get('regularity', {}).get('has_regular_pattern', False)):
            return (4, 4)  # Меньшие тайлы для регулярных структур
        else:
            return (8, 8)  # Стандартные тайлы для нерегулярных структур

    def _calculate_adaptive_blur_size(self, base_size: int = 51) -> int:
        """
        Адаптивный расчет размера размытия на основе анализа текстуры фона.

        Args:
            base_size: Базовый размер размытия по умолчанию

        Returns:
            int: Адаптивный размер размытия (всегда нечетный)
        """
        if not self.scan_analysis:
            return base_size if base_size % 2 == 1 else base_size + 1

        blur_size = base_size

        # Корректировка на основе текстуры фона
        gradient_std = self.scan_analysis.get(
            'gradient_analysis', {}).get('std_gradient', 0)

        if gradient_std > 50:  # Сильно текстурированный фон
            blur_size = 71
        elif gradient_std > 30:
            blur_size = 51
        else:  # Гладкий фон
            blur_size = 31

        # Корректировка на основе размера деталей
        if (self.reference_analysis and
                self.reference_analysis.get('has_small_features', False)):
            # Уменьшаем для мелких features
            blur_size = max(21, blur_size - 20)

        return blur_size if blur_size % 2 == 1 else blur_size + 1

    def _adaptive_normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Адаптивная нормализация гистограммы на основе динамического диапазона.

        Args:
            image: Входное изображение для нормализации

        Returns:
            np.ndarray: Нормализованное изображение в формате uint8
        """
        dst = np.empty_like(image, dtype=np.uint8)

        # Анализ динамического диапазона
        min_val = np.min(image)
        max_val = np.max(image)
        dynamic_range = max_val - min_val

        if dynamic_range < 50:  # Очень низкий динамический диапазон
            # Агрессивная нормализация для расширения диапазона
            return cv2.normalize(image, dst, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            # Сохранение части исходного диапазона для естественности
            normalized = cv2.normalize(
                image, dst, 30, 225, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return normalized

    # ==================== МЕТОДЫ РАСЧЕТА МЕТРИК КАЧЕСТВА ====================

    def _calculate_snr_improvement(self, original: np.ndarray, processed: np.ndarray) -> float:
        """
        Расчет улучшения отношения сигнал/шум после предобработки.

        Args:
            original: Исходное изображение до обработки
            processed: Обработанное изображение после применения стратегии

        Returns:
            float: Коэффициент улучшения SNR в диапазоне [0.5, 2.0]
        """
        try:
            orig_mean, orig_std = cv2.meanStdDev(original)
            proc_mean, proc_std = cv2.meanStdDev(processed)

            orig_snr = orig_mean[0][0] / (orig_std[0][0] + 1e-6)
            proc_snr = proc_mean[0][0] / (proc_std[0][0] + 1e-6)

            improvement = proc_snr / (orig_snr + 1e-6)
            return float(min(2.0, max(0.5, improvement)))
        except Exception as error:
            self.debug_fmt.debug(
                f"Ошибка расчета SNR улучшения: {error}", indent=2)
            return 1.0

    def _calculate_edge_preservation(self, original: np.ndarray, processed: np.ndarray) -> float:
        """
        Расчет степени сохранения граничных деталей после обработки.

        Args:
            original: Исходное изображение до обработки
            processed: Обработанное изображение после применения стратегии

        Returns:
            float: Коэффициент сохранения границ в диапазоне [0.0, 1.0]
        """
        try:
            # Детекция границ на исходном и обработанном изображениях
            orig_edges = cv2.Canny(original, 50, 150)
            proc_edges = cv2.Canny(processed, 50, 150)

            # Расчет пересечения и объединения границ
            intersection = np.logical_and(orig_edges > 0, proc_edges > 0).sum()
            union = np.logical_or(orig_edges > 0, proc_edges > 0).sum()

            preservation = intersection / (union + 1e-6)
            return float(max(0.0, min(1.0, preservation)))
        except Exception as error:
            self.debug_fmt.debug(
                f"Ошибка расчета сохранения границ: {error}", indent=2)
            return 0.5

    def _calculate_contrast_improvement(self, original: np.ndarray, processed: np.ndarray) -> float:
        """
        Расчет улучшения контраста после предобработки.

        Args:
            original: Исходное изображение до обработки
            processed: Обработанное изображение после применения стратегии

        Returns:
            float: Коэффициент улучшения контраста в диапазоне [0.5, 2.0]
        """
        try:
            orig_std = np.std(original.astype(np.float32))
            proc_std = np.std(processed.astype(np.float32))

            improvement = proc_std / (orig_std + 1e-6)
            return float(min(2.0, max(0.5, improvement)))
        except Exception as error:
            self.debug_fmt.debug(
                f"Ошибка расчета улучшения контраста: {error}", indent=2)
            return 1.0

    def _create_metrics(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, Any]:
        """
        Создание комплексных метрик качества предобработки и "легких" метрик для следующих этапов.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия объясняет расчет своих метрик качества
        - Создает "легкие" метрики для передачи контекста следующему этапу

        Args:
            original: Исходное изображение до обработки
            processed: Обработанное изображение после применения стратегии

        Returns:
            Dict[str, Any]: Словарь с метриками качества предобработки и легкими метриками
        """
        debug_mode = getattr(self.debug_fmt, 'debug_mode', False)

        if debug_mode:
            self.debug_fmt.debug(
                "Расчет метрик качества предобработки...", indent=1)

        # 1. ВЫПОЛНЕНИЕ "ЛЕГКОГО" АНАЛИЗА ДЛЯ СЛЕДУЮЩЕГО ЭТАПА
        lean_metrics = self.image_analyzer.analyze_image_lean(processed)

        # 2. РАСЧЕТ МЕТРИК КАЧЕСТВА ПРЕДОБРАБОТКИ
        preprocessing_metrics = PreprocessingMetrics(
            contrast_improvement=self._calculate_contrast_improvement(
                original, processed),
            snr_improvement=self._calculate_snr_improvement(
                original, processed),
            edge_preservation=self._calculate_edge_preservation(
                original, processed),
            original_std=float(np.std(original.astype(np.float32))),
            processed_std=float(np.std(processed.astype(np.float32)))
        )

        if debug_mode:
            self.debug_fmt.debug(
                "Метрики качества успешно рассчитаны", indent=2)

        return {
            'preprocessing_metrics': preprocessing_metrics,
            'lean_metrics': lean_metrics  # Передача контекста следующему этапу
        }

    def _convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Конвертация изображения в градации серого если необходимо.

        Args:
            image: Входное изображение (может быть цветным или grayscale)

        Returns:
            np.ndarray: Одноканальное изображение в градациях серого
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()

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
                f"Ошибка выполнения стратегии предобработки: {error}", indent=1)

        return StrategyResult(
            strategy_name=self.name,
            success=False,
            result_data=None,
            metrics={},
            processing_time=execution_time,
            error_message=str(error)
        )

    def _log_adaptive_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Логирование адаптивных параметров обработки.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Стратегия объясняет какие адаптивные параметры были применены
        - Помогает понять логику настройки параметров на основе анализа

        Args:
            parameters: Словарь с адаптивными параметрами для логирования
        """
        debug_mode = getattr(self.debug_fmt, 'debug_mode', False)
        if not debug_mode:
            return

        self.debug_fmt.debug("Примененные адаптивные параметры:", indent=1)
        for param_name, param_value in parameters.items():
            self.debug_fmt.debug(f"{param_name}: {param_value}", indent=2)

        # Сохранение параметров для использования в детальном логировании
        self._current_parameters = parameters
