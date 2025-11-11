# processing/strategies/preprocessing_strategies/clahe_bilateral_strategy.py
"""
Стратегия умной комбинации CLAHE + билатеральный фильтр с адаптивными параметрами.

CLAHEBilateralPreprocessingStrategy (Конкретный исполнитель) отвечает за:
- Адаптивную комбинацию CLAHE и билатеральной фильтрации
- Интеллектуальную настройку параметров на основе анализа изображения
- Детальное логирование процесса комбинированной обработки

СООТВЕТСТВИЕ ИНСТРУКЦИИ:
- Уровень 4 в иерархии логирования (конкретная реализация стратегии)
- Логирует специфичные детали КОНКРЕТНОЙ комбинации CLAHE + Bilateral
- Объясняет адаптивные настройки параметров для данной комбинации методов
"""

import time
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from .base_preprocessing import BasePreprocessingStrategy, StrategyResult


class CLAHEBilateralPreprocessingStrategy(BasePreprocessingStrategy):
    """
    Адаптивная стратегия комбинированной обработки CLAHE + билатеральный фильтр.

    Особенности реализации:
    - Интеллектуальная настройка параметров CLAHE на основе контраста и плотности
    - Адаптивная билатеральная фильтрация с сохранением границ
    - Детальное логирование процесса комбинированной обработки
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Инициализация стратегии CLAHE + билатеральный фильтр.

        Args:
            name: Уникальное имя стратегии (фиксируется как "CLAHE+Bilateral")
            config: Конфигурация с базовыми параметрами обработки
        """
        super().__init__("CLAHE+Bilateral", config)

        # Базовые параметры из конфигурации
        self.clip_limit = self.config.get('clip_limit', 3.0)
        self.tile_grid_size = tuple(self.config.get('tile_grid_size', [8, 8]))
        self.d = self.config.get('d', 9)
        self.sigma_color = self.config.get('sigma_color', 75)
        self.sigma_space = self.config.get('sigma_space', 75)

    def _process_image(self, image: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """
        Основная логика комбинированной обработки CLAHE + билатеральный фильтр.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Конкретная стратегия детально логирует свой уникальный процесс
        - Объясняет адаптивные настройки параметров для этой комбинации методов
        - Показывает последовательность применения фильтров

        Args:
            image: Входное изображение для обработки
            context: Контекст выполнения с global_analysis

        Returns:
            np.ndarray: Обработанное изображение после применения комбинации фильтров
        """
        debug_mode = context.get('debug_mode', False)

        if debug_mode:
            self.debug_fmt.debug(
                "Начало CLAHE + Bilateral обработки", indent=1)

        # Конвертация в grayscale если необходимо
        gray = self._convert_to_grayscale(image)

        # 1. АДАПТИВНАЯ НАСТРОЙКА ПАРАМЕТРОВ CLAHE
        clip_limit = self._calculate_adaptive_clip_limit()
        tile_grid_size = self._calculate_adaptive_tile_size()

        if debug_mode:
            self.debug_fmt.debug("Адаптивные параметры CLAHE:", indent=1)
            self.debug_fmt.debug(f"Clip limit: {clip_limit}", indent=2)
            self.debug_fmt.debug(f"Tile grid size: {tile_grid_size}", indent=2)

        # 2. ПРИМЕНЕНИЕ CLAHE ДЛЯ УЛУЧШЕНИЯ КОНТРАСТА
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=tile_grid_size)
        clahe_image = clahe.apply(gray)

        if debug_mode:
            self.debug_fmt.debug("CLAHE применен успешно", indent=1)

        # 3. АДАПТИВНАЯ НАСТРОЙКА ПАРАМЕТРОВ БИЛАТЕРАЛЬНОГО ФИЛЬТРА
        d, sigma_color, sigma_space = self._calculate_adaptive_bilateral_params()

        if debug_mode:
            self.debug_fmt.debug("Адаптивные параметры Bilateral:", indent=1)
            self.debug_fmt.debug(f"d: {d}", indent=2)
            self.debug_fmt.debug(f"sigma_color: {sigma_color}", indent=2)
            self.debug_fmt.debug(f"sigma_space: {sigma_space}", indent=2)

        # 4. ПРИМЕНЕНИЕ БИЛАТЕРАЛЬНОГО ФИЛЬТРА ДЛЯ СОХРАНЕНИЯ ГРАНИЦ
        processed_image = cv2.bilateralFilter(
            clahe_image, d, sigma_color, sigma_space)

        if debug_mode:
            self.debug_fmt.debug(
                "Билатеральный фильтр применен успешно", indent=1)
            self.debug_fmt.debug(
                "CLAHE + Bilateral обработка завершена", indent=1)

        return processed_image

    def _calculate_adaptive_bilateral_params(self) -> Tuple[int, float, float]:
        """
        Адаптивный расчет параметров билатерального фильтра на основе анализа изображения.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Конкретная стратегия объясняет логику настройки СВОИХ параметров
        - Показывает как анализ изображения влияет на параметры фильтра

        Returns:
            Tuple[int, float, float]: Адаптивные параметры (d, sigma_color, sigma_space)
        """
        # Базовые параметры из конфигурации
        d = self.d
        sigma_color = self.sigma_color
        sigma_space = self.sigma_space

        if not self.scan_analysis:
            return d, sigma_color, sigma_space

        # ЛОГИКА АДАПТИВНОЙ НАСТРОЙКИ ДЛЯ КОНКРЕТНОЙ СТРАТЕГИИ

        # 1. НАСТРОЙКА ДЛЯ СОХРАНЕНИЯ МЕЛКИХ ДЕТАЛЕЙ
        if (self.reference_analysis and
                self.reference_analysis.get('has_small_features', False)):
            d = 5  # Уменьшаем диаметр для лучшего сохранения мелких деталей
            sigma_color = 50  # Уменьшаем цветовую сигму
            sigma_space = 50  # Уменьшаем пространственную сигму

        # 2. НАСТРОЙКА ДЛЯ СИЛЬНОГО ШУМА
        if self.scan_analysis.get('noise_level', 0) > 0.2:
            sigma_color += 25  # Увеличиваем для лучшего подавления шума
            sigma_space += 25  # Увеличиваем пространственное сглаживание

        # 3. ДОПОЛНИТЕЛЬНАЯ НАСТРОЙКА ДЛЯ НИЗКОГО КОНТРАСТА
        if self.scan_analysis.get('low_contrast', False):
            # Более агрессивная фильтрация для низкоконтрастных изображений
            sigma_color = min(100, sigma_color + 15)
            sigma_space = min(100, sigma_space + 15)

        # Логирование адаптивных параметров
        debug_mode = getattr(self.debug_fmt, 'debug_mode', False)
        if debug_mode:
            self.debug_fmt.debug(
                "Логика настройки Bilateral параметров:", indent=2)
            if (self.reference_analysis and
                    self.reference_analysis.get('has_small_features', False)):
                self.debug_fmt.debug(
                    "→ Мелкие детали: уменьшены d, sigma_color, sigma_space", indent=3)
            if self.scan_analysis.get('noise_level', 0) > 0.2:
                self.debug_fmt.debug(
                    "→ Сильный шум: увеличены sigma_color, sigma_space", indent=3)
            if self.scan_analysis.get('low_contrast', False):
                self.debug_fmt.debug(
                    "→ Низкий контраст: увеличены sigma_color, sigma_space", indent=3)

        return d, sigma_color, sigma_space

    def _calculate_adaptive_clip_limit(self) -> float:
        """
        Адаптивный расчет clip limit для CLAHE с учетом специфики комбинированной обработки.

        Returns:
            float: Адаптивный clip limit для CLAHE
        """
        # Используем базовый метод, но с дополнительной логикой для этой стратегии
        base_clip_limit = super()._calculate_adaptive_clip_limit(self.clip_limit)

        # Дополнительная корректировка для комбинированной обработки
        if (self.scan_analysis and
            self.scan_analysis.get('high_noise', False) and
                self.scan_analysis.get('low_contrast', False)):
            # Для шумных низкоконтрастных изображений используем более агрессивный CLAHE
            base_clip_limit = min(4.0, base_clip_limit + 0.5)

        debug_mode = getattr(self.debug_fmt, 'debug_mode', False)
        if debug_mode:
            self.debug_fmt.debug(
                f"Адаптивный CLAHE clip limit: {base_clip_limit}", indent=2)

        return base_clip_limit

    def _calculate_adaptive_tile_size(self) -> Tuple[int, int]:
        """
        Адаптивный расчет размера тайлов для CLAHE с учетом комбинированной обработки.

        Returns:
            Tuple[int, int]: Адаптивный размер тайлов (rows, cols)
        """
        base_tile_size = super()._calculate_adaptive_tile_size()

        # Дополнительная корректировка для комбинированной стратегии
        if (self.scan_analysis and
                self.scan_analysis.get('has_uneven_lighting', False)):
            # Для неравномерного освещения используем меньшие тайлы
            base_tile_size = (max(4, base_tile_size[0] - 2),
                              max(4, base_tile_size[1] - 2))

        debug_mode = getattr(self.debug_fmt, 'debug_mode', False)
        if debug_mode:
            self.debug_fmt.debug(
                f"Адаптивный CLAHE tile size: {base_tile_size}", indent=2)

        return base_tile_size

    def _log_adaptive_parameters(self) -> None:
        """
        Специфичное логирование адаптивных параметров для комбинированной стратегии.

        СООТВЕТСТВИЕ ИНСТРУКЦИИ:
        - Конкретная стратегия логирует свои уникальные адаптивные параметры
        - Объясняет логику настройки для этой конкретной комбинации методов
        """
        debug_mode = getattr(self.debug_fmt, 'debug_mode', False)
        if not debug_mode:
            return

        # Расчет адаптивных параметров
        clip_limit = self._calculate_adaptive_clip_limit()
        tile_grid_size = self._calculate_adaptive_tile_size()
        d, sigma_color, sigma_space = self._calculate_adaptive_bilateral_params()

        # Логирование параметров комбинированной стратегии
        self.debug_fmt.debug("Адаптивные параметры CLAHE+Bilateral:", indent=1)

        self.debug_fmt.debug("CLAHE параметры:", indent=2)
        self.debug_fmt.debug(f"  Clip limit: {clip_limit}", indent=3)
        self.debug_fmt.debug(f"  Tile grid: {tile_grid_size}", indent=3)

        self.debug_fmt.debug("Bilateral параметры:", indent=2)
        self.debug_fmt.debug(f"  Diameter (d): {d}", indent=3)
        self.debug_fmt.debug(f"  Sigma color: {sigma_color}", indent=3)
        self.debug_fmt.debug(f"  Sigma space: {sigma_space}", indent=3)

        # Сохранение для использования в детальном логировании
        self._current_parameters = {
            'clahe_clip_limit': clip_limit,
            'clahe_tile_grid_size': tile_grid_size,
            'bilateral_d': d,
            'bilateral_sigma_color': sigma_color,
            'bilateral_sigma_space': sigma_space
        }
