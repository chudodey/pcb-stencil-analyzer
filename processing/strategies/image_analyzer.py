# processing/strategies/image_analyzer.py
"""
Анализатор изображений для интеллектуальной предобработки.
Выполняет единоразовый анализ скана и эталона перед пайплайном.

ОПТИМИЗИРОВАННАЯ ВЕРСИЯ:
- Устранены избыточные вычисления
- Заменены O(N^2) алгоритмы на O(N) или O(k^2) (на thumbnail'ах)
- Заменены медленные циклы Python на векторизованные/нативные функции

ОБНОВЛЕННЫЙ ВЫВОД:
- Использует DebugFormatter для отладочных сообщений (серый цвет)
- Ключевые результаты показываются оператору через UIService
- Детальная отладка доступна разработчикам
- Сводный текстовый отчет распределен по методам
"""

import time
from typing import Any, Dict, List, Tuple
import numpy as np

# pylint: disable=no-member
import cv2
from scipy import stats

from infrastructure.debug_formatter import DebugFormatter


class ImageAnalyzer:
    """Анализатор характеристик скана и эталона Gerber."""

    def __init__(self, debug_mode: bool = False):
        """
        Инициализация анализатора изображений.

        Args:
            config_service: Сервис конфигурации
        """
        self.debug_mode = debug_mode
        self.debug = DebugFormatter(debug_mode, __name__)

        # Размеры для thumbnail'ов, используемых в тяжелых операциях
        self.SYMMETRY_DIM = 200
        self.SSIM_DIM = 256

    def analyze_image_lean(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Выполняет "лёгкий" анализ уже обработанного изображения.

        Args:
            image: Обработанное изображение в градациях серого

        Returns:
            Словарь с ключевыми метриками
        """
        # Базовые статистики
        mean_intensity = float(np.mean(image))
        std_intensity = float(np.std(image))

        # Анализ гистограммы
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist.flatten()
        peaks = self._find_histogram_peaks(hist)

        return {
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'histogram_peaks': peaks,
            'is_dark': mean_intensity < 80,
            'is_bright': mean_intensity > 180,
            'low_contrast': std_intensity < 40,
            'is_bimodal': len(peaks) == 2 and peaks[0][1] > 0.2 and peaks[1][1] > 0.2
        }

    def analyze_images(self, scan: np.ndarray, reference: np.ndarray) -> Dict[str, Any]:
        """
        Комплексный анализ скана и эталона.

        Args:
            scan: Изображение скана
            reference: Эталонное Gerber-изображение

        Returns:
            Dict с результатами анализа
        """
        self.debug.info("Начат комплексный анализ изображений...")

        with self.debug.timed_section("Комплексный анализ изображений"):
            scan_analysis = self._analyze_scan(scan)
            reference_analysis = self._analyze_reference(reference)
            comparative_analysis = self._compare_images(scan, reference)

            # Генерация рекомендаций
            recommendations = self._generate_recommendations(
                scan_analysis, reference_analysis)

        # Вывод ключевых результатов для оператора
        self._log_analysis_summary(
            scan_analysis, reference_analysis, recommendations)

        # Детальный отладочный вывод
        self._log_detailed_analysis(scan_analysis, reference_analysis,
                                    comparative_analysis, recommendations)

        return {
            'scan_analysis': scan_analysis,
            'reference_analysis': reference_analysis,
            'comparative_analysis': comparative_analysis,
            'recommendations': recommendations
        }

    def _analyze_scan(self, scan: np.ndarray) -> Dict[str, Any]:
        """Детальный анализ характеристик скана."""
        with self.debug.timed_section("Анализ скана"):
            gray = self._convert_to_grayscale(scan)

            # Базовые статистики
            mean_intensity = float(np.mean(gray))
            std_intensity = float(np.std(gray))
            min_intensity = float(np.min(gray))
            max_intensity = float(np.max(gray))
            dynamic_range = max_intensity - min_intensity

            # Анализ гистограммы
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten()
            peaks = self._find_histogram_peaks(hist)
            entropy = float(stats.entropy(hist + 1e-6))

            # Анализ распределения яркости
            brightness_distribution = self._analyze_brightness_distribution(
                gray)

            # Шум и градиенты
            noise_level = self._estimate_noise_level(gray)
            noise_type = self._analyze_noise_type(noise_level)
            gradient_analysis = self._analyze_gradients(gray)
            texture_analysis = self._analyze_texture(gradient_analysis)

            # Анализ контраста
            contrast_metrics = self._analyze_contrast(gray)

            # Общая оценка качества
            quality_score = self._calculate_quality_score(
                gray, noise_level, gradient_analysis)

            # Отладочный вывод метрик скана
            if self.debug_mode:
                self._log_scan_debug_info(
                    mean_intensity, std_intensity, dynamic_range,
                    noise_level, quality_score, entropy, peaks,
                    brightness_distribution, contrast_metrics
                )

        return {
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'min_intensity': min_intensity,
            'max_intensity': max_intensity,
            'dynamic_range': dynamic_range,
            'histogram_peaks': peaks,
            'entropy': entropy,
            'brightness_distribution': brightness_distribution,
            'noise_level': noise_level,
            'noise_type': noise_type,
            'gradient_analysis': gradient_analysis,
            'texture_analysis': texture_analysis,
            'contrast_metrics': contrast_metrics,
            'low_contrast': std_intensity < 40,
            'is_dark': mean_intensity < 80,
            'is_bright': mean_intensity > 180,
            'is_washed_out': dynamic_range < 100,
            'has_high_dynamic_range': dynamic_range > 150,
            'quality_score': quality_score,
            'high_noise': noise_level > 0.1
        }

    def _analyze_reference(self, reference: np.ndarray) -> Dict[str, Any]:
        """Детальный анализ характеристик эталона Gerber."""
        with self.debug.timed_section("Анализ эталона"):
            gray = self._convert_to_grayscale(reference)

            # Статистики пикселей
            unique, counts = np.unique(gray, return_counts=True)
            black_pixels = counts[unique == 0][0] if 0 in unique else 0
            white_pixels = counts[unique == 255][0] if 255 in unique else 0
            total_pixels = gray.size

            hole_ratio = black_pixels / total_pixels
            metal_ratio = white_pixels / total_pixels

            # Анализ компонентов
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
                binary, connectivity=8)

            components_analysis = self._analyze_component_sizes(
                stats, num_labels)
            spatial_distribution = self._analyze_spatial_distribution(
                centroids, num_labels, gray.shape)
            regularity = self._analyze_regularity(gray)
            symmetry_analysis = self._analyze_symmetry(gray)
            geometric_features = self._analyze_geometric_features(binary)
            boundary_analysis = self._analyze_boundaries(gray)

            # Оценка сложности
            complexity_score = self._calculate_complexity_score(
                components_analysis, regularity)

            # Отладочный вывод метрик эталона
            if self.debug_mode:
                self._log_reference_debug_info(
                    hole_ratio, metal_ratio, components_analysis,
                    spatial_distribution, regularity, symmetry_analysis,
                    geometric_features, complexity_score
                )

        return {
            'hole_ratio': hole_ratio,
            'metal_ratio': metal_ratio,
            'density_category': self._categorize_density(hole_ratio),
            'fill_factor': metal_ratio / (hole_ratio + 1e-6),
            'component_sizes': components_analysis,
            'spatial_distribution': spatial_distribution,
            'regularity': regularity,
            'symmetry_analysis': symmetry_analysis,
            'geometric_features': geometric_features,
            'boundary_analysis': boundary_analysis,
            'is_high_density': hole_ratio > 0.3,
            'is_low_density': hole_ratio < 0.1,
            'is_balanced': 0.15 <= hole_ratio <= 0.25,
            'has_small_features': components_analysis['mean_size'] < 100,
            'has_large_features': components_analysis['mean_size'] > 500,
            'has_varied_sizes': components_analysis['std_size'] > components_analysis['mean_size'] * 0.5,
            'complexity_score': complexity_score
        }

    def _compare_images(self, scan: np.ndarray, reference: np.ndarray) -> Dict[str, Any]:
        """Сравнительный анализ скана и эталона."""
        with self.debug.timed_section("Сравнительный анализ"):
            scan_gray = self._convert_to_grayscale(scan)
            ref_gray = self._convert_to_grayscale(reference)

            # Статистики сравнения
            scan_mean = float(np.mean(scan_gray))
            ref_mean = float(np.mean(ref_gray))
            scan_std = float(np.std(scan_gray))
            ref_std = float(np.std(ref_gray))

            # Сравнительные метрики
            hist_correlation = self._compare_histograms(scan_gray, ref_gray)
            structural_similarity = self._calculate_structural_similarity(
                scan_gray, ref_gray)

            # Отладочный вывод сравнения
            if self.debug_mode:
                self._log_comparison_debug_info(
                    scan_mean, ref_mean, scan_std, ref_std,
                    hist_correlation, structural_similarity
                )

        return {
            'brightness_shift': scan_mean - ref_mean,
            'contrast_difference': scan_std - ref_std,
            'requires_brightness_correction': abs(scan_mean - ref_mean) > 20,
            'requires_contrast_correction': abs(scan_std - ref_std) > 15,
            'histogram_correlation': hist_correlation,
            'structural_similarity': structural_similarity,
            'estimated_correction_gamma': self._estimate_gamma_correction(scan_mean, ref_mean),
            'estimated_contrast_boost': max(1.0, ref_std / (scan_std + 1e-6))
        }

    def _log_analysis_summary(self, scan_analysis: Dict, reference_analysis: Dict,
                              recommendations: Dict) -> None:
        """Логирует ключевые результаты анализа для оператора."""

        # Ключевые метрики скана
        scan_items = {
            "Яркость": f"{scan_analysis['mean_intensity']:.1f}",
            "Контраст": f"{scan_analysis['std_intensity']:.1f}",
            "Уровень шума": f"{scan_analysis['noise_level']:.3f}",
            "Оценка качества": f"{scan_analysis['quality_score']:.2f}/1.0"
        }
        self.debug.metrics_table("ХАРАКТЕРИСТИКИ СКАНА:", scan_items)

        # Ключевые метрики эталона
        ref_items = {
            "Плотность отверстий": f"{reference_analysis['hole_ratio']:.3f}",
            "Компоненты": str(
                reference_analysis['component_sizes']['num_components']),
            "Сложность структуры":
            f"{reference_analysis['complexity_score']:.2f}/1.0"
        }
        self.debug.metrics_table("ХАРАКТЕРИСТИКИ ЭТАЛОНА:", ref_items)

        # Рекомендации
        # СТАЛО (правильно):
        self.debug.info("РЕКОМЕНДАЦИИ ПО ПРЕДОБРАБОТКЕ:")
        # А затем отдельно вывести содержимое списка
        for step in recommendations['preprocessing_steps']:
            self.debug.info(f"  • {step}", indent=1)

        if recommendations['warnings']:
            for warning in recommendations['warnings']:
                self.debug.warn(warning)

    def _log_detailed_analysis(self, scan_analysis: Dict, reference_analysis: Dict,
                               comparative_analysis: Dict, recommendations: Dict) -> None:
        """Детальный отладочный вывод анализа."""
        if not self.debug_mode:
            return

        self.debug.header("ДЕТАЛЬНЫЙ АНАЛИЗ ИЗОБРАЖЕНИЙ")

        # Сводка рекомендаций
        rec_info = {
            'Шаги предобработки': ', '.join(recommendations['preprocessing_steps']),
            'Приоритет стратегий': ', '.join(recommendations['strategy_priority'][:3]),
            'Корректировки параметров': str(recommendations['parameter_adjustments'])
        }
        self.debug.box("РЕКОМЕНДАЦИИ СИСТЕМЫ", rec_info, indent=1)

        # Сравнительный анализ
        comp_info = {
            'Требуется коррекция яркости': comparative_analysis['requires_brightness_correction'],
            'Требуется коррекция контраста': comparative_analysis['requires_contrast_correction'],
            'Рекомендуемая гамма': f"{comparative_analysis['estimated_correction_gamma']:.2f}",
            'Усиление контраста': f"{comparative_analysis['estimated_contrast_boost']:.2f}x"
        }
        self.debug.box("СРАВНИТЕЛЬНЫЙ АНАЛИЗ", comp_info, indent=1)

    def _log_scan_debug_info(self, mean_intensity: float, std_intensity: float,
                             dynamic_range: float, noise_level: float, quality_score: float,
                             entropy: float, peaks: List[Tuple[int, float]],
                             brightness_distribution: Dict, contrast_metrics: Dict) -> None:
        """Отладочный вывод информации о скане."""
        scan_metrics = {
            'Яркость': f"{mean_intensity:.1f}",
            'Контраст': f"{std_intensity:.1f}",
            'Динамический диапазон': f"{dynamic_range:.1f}",
            'Уровень шума': f"{noise_level:.3f}",
            'Энтропия': f"{entropy:.3f}",
            'Оценка качества': f"{quality_score:.2f}/1.0",
            'Равномерность освещения': f"{brightness_distribution['uniformity_score']:.3f}",
            'Локальный контраст': f"{contrast_metrics['local_contrast']:.1f}"
        }
        self.debug.metrics_table("МЕТРИКИ СКАНА", scan_metrics, indent=2)

        # Пики гистограммы
        if peaks:
            peak_info = {f'Пик {i} (уровень {pos})': f"{prominence:.3f}"
                         for i, (pos, prominence) in enumerate(peaks[:2])}
            self.debug.metrics_table("ПИКИ ГИСТОГРАММЫ", peak_info, indent=3)

    def _log_reference_debug_info(self, hole_ratio: float, metal_ratio: float,
                                  components_analysis: Dict, spatial_distribution: Dict,
                                  regularity: Dict, symmetry_analysis: Dict,
                                  geometric_features: Dict, complexity_score: float) -> None:
        """Отладочный вывод информации о эталоне."""
        ref_metrics = {
            'Плотность отверстий': f"{hole_ratio:.3f}",
            'Коэффициент заполнения': f"{metal_ratio / (hole_ratio + 1e-6):.2f}",
            'Компоненты': str(components_analysis['num_components']),
            'Средний размер': f"{components_analysis['mean_size']:.1f} px",
            'Равномерность распределения': f"{spatial_distribution['uniformity']:.3f}",
            'Регулярность паттерна': f"{regularity['regularity_score']:.3f}",
            'Общая симметрия': f"{symmetry_analysis['overall_symmetry']:.3f}",
            'Контуры': str(geometric_features['total_contours']),
            'Сложность структуры': f"{complexity_score:.2f}/1.0"
        }
        self.debug.metrics_table("МЕТРИКИ ЭТАЛОНА", ref_metrics, indent=2)

        # Дополнительная информация о компонентах
        if components_analysis['num_components'] > 0:
            comp_info = {
                'Мин. размер': f"{components_analysis['min_size']} px",
                'Макс. размер': f"{components_analysis['max_size']} px",
                'Разброс размеров': f"{components_analysis['std_size']:.1f} px"
            }
            self.debug.metrics_table(
                "ХАРАКТЕРИСТИКИ КОМПОНЕНТОВ", comp_info, indent=3)

    def _log_comparison_debug_info(self, scan_mean: float, ref_mean: float,
                                   scan_std: float, ref_std: float,
                                   hist_correlation: float, structural_similarity: float) -> None:
        """Отладочный вывод сравнительной информации."""
        comp_metrics = {
            'Сдвиг яркости': f"{scan_mean - ref_mean:+.1f}",
            'Разница контраста': f"{scan_std - ref_std:+.1f}",
            'Корреляция гистограмм': f"{hist_correlation:.3f}",
            'Структурное сходство': f"{structural_similarity:.3f}",
            'Относительная яркость': f"{scan_mean/ref_mean:.2f}",
            'Относительный контраст': f"{scan_std/ref_std:.2f}"
        }
        self.debug.metrics_table(
            "СРАВНИТЕЛЬНЫЕ МЕТРИКИ", comp_metrics, indent=2)

    # ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ АНАЛИЗА ====================

    def _analyze_brightness_distribution(self, image: np.ndarray) -> Dict[str, Any]:
        """Анализ распределения яркости по квадрантам."""
        height, width = image.shape
        quadrants = [
            image[:height//2, :width//2],  # верхний левый
            image[:height//2, width//2:],  # верхний правый
            image[height//2:, :width//2],  # нижний левый
            image[height//2:, width//2:]   # нижний правый
        ]

        quadrant_means = [float(np.mean(q)) for q in quadrants]
        mean_of_means = float(np.mean(quadrant_means))
        std_of_means = float(np.std(quadrant_means))

        uniformity = 1 - (std_of_means / (mean_of_means + 1e-6))

        return {
            'quadrant_means': quadrant_means,
            'uniformity_score': uniformity,
            'has_uneven_lighting': uniformity < 0.8
        }

    def _analyze_noise_type(self, noise_level: float) -> str:
        """Определение типа шума."""
        if noise_level < 0.05:
            return "low"
        elif noise_level < 0.1:
            return "gaussian"
        elif noise_level < 0.2:
            return "mixed"
        else:
            return "high"

    def _analyze_texture(self, gradient_analysis: Dict) -> Dict[str, float]:
        """Анализ текстуры изображения."""
        gradient_variance = gradient_analysis['std_gradient'] ** 2

        return {
            'texture_variance': gradient_variance,
            'is_textured': gradient_variance > 1000,
            'is_smooth': gradient_variance < 100
        }

    def _analyze_contrast(self, image: np.ndarray) -> Dict[str, float]:
        """Детальный анализ контраста."""
        blurred = cv2.GaussianBlur(image, (15, 15), 0)
        local_contrast = float(
            np.std((image.astype(np.float32) - blurred.astype(np.float32))))

        return {
            'global_contrast': float(np.std(image)),
            'local_contrast': local_contrast,
            'contrast_ratio': float(np.std(image) / (np.mean(image) + 1e-6))
        }

    def _analyze_spatial_distribution(self, centroids: np.ndarray, num_labels: int,
                                      image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Анализ пространственного распределения компонентов."""
        if num_labels <= 2:  # 1 компонент + фон
            return {'uniformity': 1.0, 'clustering': 0.0, 'is_uniform': True, 'is_clustered': False}

        valid_centroids = centroids[1:]  # убираем фон (индекс 0)

        if valid_centroids.shape[0] < 2:
            return {'uniformity': 1.0, 'clustering': 0.0, 'is_uniform': True, 'is_clustered': False}

        # O(N) анализ разброса координат
        std_dev_xy = np.std(valid_centroids, axis=0)
        height, width = image_shape

        # Нормализуем разброс относительно размеров изображения
        norm_std_x = std_dev_xy[0] / (width + 1e-6)
        norm_std_y = std_dev_xy[1] / (height + 1e-6)

        # Средний нормированный разброс
        mean_norm_std = (norm_std_x + norm_std_y) / 2.0

        # Преобразуем разброс в "равномерность"
        uniformity = np.clip(1.0 - mean_norm_std * 3.0, 0.0, 1.0)
        clustering = 1.0 - uniformity

        return {
            'uniformity': float(uniformity),
            'clustering': float(clustering),
            'is_uniform': uniformity > 0.7,
            'is_clustered': uniformity < 0.3
        }

    def _analyze_symmetry(self, image: np.ndarray) -> Dict[str, float]:
        """Анализирует симметрию изображения на уменьшенной копии."""
        try:
            # Уменьшаем изображение до фиксированного размера
            small_image = cv2.resize(
                image, (self.SYMMETRY_DIM, self.SYMMETRY_DIM),
                interpolation=cv2.INTER_AREA
            )

            height, width = small_image.shape

            # Горизонтальная симметрия (левая/правая половины)
            mid_x = width // 2
            left_half = small_image[:, :mid_x]
            right_half = small_image[:, mid_x:]

            # Обеспечиваем одинаковые размеры
            min_height = min(left_half.shape[0], right_half.shape[0])
            min_width = min(left_half.shape[1], right_half.shape[1])

            left_half = left_half[:min_height, :min_width]
            right_half_flipped = np.fliplr(right_half[:min_height, :min_width])

            horizontal_symmetry = np.corrcoef(
                left_half.flatten(),
                right_half_flipped.flatten()
            )[0, 1] if min_height > 0 and min_width > 0 else 0.0

            # Вертикальная симметрия (верхняя/нижняя половины)
            mid_y = height // 2
            top_half = small_image[:mid_y, :]
            bottom_half = small_image[mid_y:, :]

            # Обеспечиваем одинаковые размеры
            min_height_v = min(top_half.shape[0], bottom_half.shape[0])
            min_width_v = min(top_half.shape[1], bottom_half.shape[1])

            top_half = top_half[:min_height_v, :min_width_v]
            bottom_half_flipped = np.flipud(
                bottom_half[:min_height_v, :min_width_v])

            vertical_symmetry = np.corrcoef(
                top_half.flatten(),
                bottom_half_flipped.flatten()
            )[0, 1] if min_height_v > 0 and min_width_v > 0 else 0.0

            # Центральная симметрия
            center_symmetry = np.corrcoef(
                small_image.flatten(),
                np.flipud(np.fliplr(small_image)).flatten()
            )[0, 1]

            overall_symmetry = (horizontal_symmetry +
                                vertical_symmetry + center_symmetry) / 3

            return {
                'horizontal_symmetry': float(horizontal_symmetry),
                'vertical_symmetry': float(vertical_symmetry),
                'center_symmetry': float(center_symmetry),
                'overall_symmetry': float(overall_symmetry)
            }

        except Exception as e:
            self.debug.warn(f"Ошибка анализа симметрии: {e}")
            return {
                'horizontal_symmetry': 0.0,
                'vertical_symmetry': 0.0,
                'center_symmetry': 0.0,
                'overall_symmetry': 0.0
            }

    def _analyze_geometric_features(self, binary_image: np.ndarray) -> Dict[str, Any]:
        """Анализ геометрических характеристик."""
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {'total_contours': 0, 'mean_circularity': 0.0, 'std_circularity': 0.0}

        circularities = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                circularities.append(circularity)

        return {
            'total_contours': len(contours),
            'mean_circularity': float(np.mean(circularities)) if circularities else 0.0,
            'std_circularity': float(np.std(circularities)) if circularities else 0.0
        }

    def _analyze_boundaries(self, image: np.ndarray) -> Dict[str, float]:
        """Анализ границ и краев."""
        edges = cv2.Canny(image, 50, 150)
        edge_density = float(np.sum(edges > 0) / edges.size)

        return {
            'edge_density': edge_density,
            'has_strong_edges': edge_density > 0.1,
            'has_weak_edges': edge_density < 0.01
        }

    def _compare_histograms(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Сравнение гистограмм двух изображений."""
        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

        # Нормализация
        hist1 /= (np.sum(hist1) + 1e-6)
        hist2 /= (np.sum(hist2) + 1e-6)

        # Корреляция
        return float(np.corrcoef(hist1.flatten(), hist2.flatten())[0, 1])

    def _calculate_structural_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Упрощенный расчет структурного сходства на уменьшенной копии."""
        # Уменьшаем изображения до SSIM_DIM
        if img1.shape != (self.SSIM_DIM, self.SSIM_DIM):
            img1_small = cv2.resize(img1, (self.SSIM_DIM, self.SSIM_DIM),
                                    interpolation=cv2.INTER_AREA)
        else:
            img1_small = img1

        if img2.shape != (self.SSIM_DIM, self.SSIM_DIM):
            img2_small = cv2.resize(img2, (self.SSIM_DIM, self.SSIM_DIM),
                                    interpolation=cv2.INTER_AREA)
        else:
            img2_small = img2

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        # Все вычисления на small-версиях
        mu1 = float(np.mean(img1_small))
        mu2 = float(np.mean(img2_small))
        sigma1 = float(np.std(img1_small))
        sigma2 = float(np.std(img2_small))

        sigma12 = float(np.cov(img1_small.flatten(),
                        img2_small.flatten())[0, 1])

        numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        denominator = ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))

        ssim = numerator / denominator if denominator != 0 else 0.0

        return max(0.0, ssim)

    def _estimate_gamma_correction(self, scan_mean: float, ref_mean: float) -> float:
        """Оценивает параметр гамма-коррекции."""
        try:
            if scan_mean <= 0 or ref_mean <= 0:
                return 1.0

            # Нормализуем
            norm_scan_mean = scan_mean / 255.0
            norm_ref_mean = ref_mean / 255.0

            # (scan^gamma) = ref  =>  gamma * log(scan) = log(ref)
            gamma = np.log(norm_ref_mean) / (np.log(norm_scan_mean) + 1e-6)

            # Ограничиваем гамма разумными пределами
            return float(np.clip(gamma, 0.5, 2.0))

        except (ValueError, ZeroDivisionError, OverflowError):
            return 1.0

    def _calculate_quality_score(self, image: np.ndarray, noise_level: float,
                                 gradients: Dict) -> float:
        """Общая оценка качества изображения."""
        score = 0.0

        # Контраст (30%)
        contrast = float(np.std(image)) / 80.0
        score += min(1.0, contrast) * 0.3

        # Яркость (20%)
        brightness = 1 - abs(float(np.mean(image)) - 128) / 128.0
        score += brightness * 0.2

        # Шум (30%)
        noise = 1 - min(1.0, noise_level * 10)
        score += noise * 0.3

        # Резкость (20%)
        sharpness = min(1.0, gradients['mean_gradient'] / 50.0)
        score += sharpness * 0.2

        return score

    def _calculate_complexity_score(self, components: Dict, regularity: Dict) -> float:
        """Оценка сложности структуры эталона."""
        if components['num_components'] == 0:
            return 0.0

        # Сложность растет с количеством компонентов и нерегулярностью
        component_complexity = min(1.0, components['num_components'] / 1000.0)
        irregularity = 1 - regularity['regularity_score']

        return float(component_complexity * 0.6 + irregularity * 0.4)

    # ==================== БАЗОВЫЕ МЕТОДЫ ====================

    def _convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Конвертирует изображение в оттенки серого."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()

    def _find_histogram_peaks(self, hist: np.ndarray,
                              min_prominence: float = 0.1) -> List[Tuple[int, float]]:
        """Находит пики в гистограмме."""
        peaks = []
        total = np.sum(hist)
        if total == 0:
            return []

        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                prominence = hist[i] / total
                if prominence > min_prominence:
                    peaks.append((i, float(prominence)))
        return sorted(peaks, key=lambda x: x[1], reverse=True)[:3]

    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """Оценивает уровень шума в изображении."""
        # Дисперсия Лапласиана (быстро)
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        variance = float(np.var(laplacian))

        return float(variance / 10000.0 if variance > 1e-6 else 0.0)

    def _analyze_gradients(self, image: np.ndarray) -> Dict[str, float]:
        """Анализирует градиенты изображения."""
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        return {
            'mean_gradient': float(np.mean(gradient_magnitude)),
            'std_gradient': float(np.std(gradient_magnitude)),
            'sharpness_ratio': float(np.mean(gradient_magnitude) / (np.std(image) + 1e-6))
        }

    def _analyze_component_sizes(self, stats: np.ndarray, num_labels: int) -> Dict[str, float]:
        """Анализирует размеры компонентов."""
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]  # stats[0] - это фон

            # Проверка, что у нас действительно есть компоненты
            if areas.size > 0:
                return {
                    'num_components': num_labels - 1,
                    'mean_size': float(np.mean(areas)),
                    'std_size': float(np.std(areas)),
                    'min_size': float(np.min(areas)),
                    'max_size': float(np.max(areas))
                }

        # Если num_labels <= 1 или areas.size == 0
        return {
            'num_components': 0,
            'mean_size': 0.0,
            'std_size': 0.0,
            'min_size': 0.0,
            'max_size': 0.0
        }

    def _analyze_regularity(self, image: np.ndarray) -> Dict[str, float]:
        """Анализирует регулярность паттернов (FFT)."""
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)

        height, width = magnitude_spectrum.shape
        center_y, center_x = height // 2, width // 2

        # Энергия в низкочастотной области (центр спектра)
        low_freq_size = min(height, width) // 20  # Берем 5%
        low_freq_energy = np.sum(
            magnitude_spectrum[
                center_y-low_freq_size: center_y+low_freq_size,
                center_x-low_freq_size: center_x+low_freq_size
            ]
        )
        total_energy = np.sum(magnitude_spectrum)

        regularity_score = low_freq_energy / (total_energy + 1e-6)

        return {
            'regularity_score': float(regularity_score),
            'has_regular_pattern': regularity_score > 0.3
        }

    def _categorize_density(self, hole_ratio: float) -> str:
        """Категоризирует плотность отверстий."""
        if hole_ratio < 0.05:
            return "very_low"
        elif hole_ratio < 0.15:
            return "low"
        elif hole_ratio < 0.25:
            return "medium"
        elif hole_ratio < 0.35:
            return "high"
        else:
            return "very_high"

    def _generate_recommendations(self, scan_analysis: Dict,
                                  reference_analysis: Dict) -> Dict[str, Any]:
        """Генерация детальных рекомендаций для стратегий предобработки."""
        recommendations = {
            'preprocessing_steps': [],
            'parameter_adjustments': {},
            'strategy_priority': [],
            'warnings': []
        }

        # Рекомендации на основе скана
        if scan_analysis['is_dark']:
            recommendations['preprocessing_steps'].append(
                'brightness_correction')
            recommendations['parameter_adjustments']['brightness'] = 'increase'
            recommendations['strategy_priority'].extend(
                ['GammaCorrection', 'CLAHE'])

        elif scan_analysis['is_bright']:
            recommendations['preprocessing_steps'].append(
                'brightness_correction')
            recommendations['parameter_adjustments']['brightness'] = 'decrease'
            recommendations['strategy_priority'].extend(['GammaCorrection'])

        if scan_analysis['low_contrast']:
            recommendations['preprocessing_steps'].append(
                'contrast_enhancement')
            recommendations['parameter_adjustments']['contrast'] = 'aggressive'
            recommendations['strategy_priority'].extend(
                ['CLAHE+Bilateral', 'UnsharpMask'])

        if scan_analysis['high_noise']:
            recommendations['preprocessing_steps'].append('noise_reduction')
            recommendations['parameter_adjustments']['denoising'] = 'strong'
            recommendations['strategy_priority'].extend(
                ['GaussianBlur', 'MedianBlur'])

        if scan_analysis['is_washed_out']:
            recommendations['preprocessing_steps'].append(
                'dynamic_range_compression')
            recommendations['warnings'].append(
                'Изображение имеет низкий динамический диапазон')

        # Рекомендации на основе эталона
        if reference_analysis['is_high_density']:
            recommendations['preprocessing_steps'].append(
                'delicate_processing')
            recommendations['parameter_adjustments']['kernel_sizes'] = 'small'
            recommendations['parameter_adjustments']['filter_strength'] = 'light'
            recommendations['warnings'].append(
                'Высокая плотность отверстий - требуется деликатная обработка')

        elif reference_analysis['has_small_features']:
            recommendations['preprocessing_steps'].append(
                'feature_preservation')
            recommendations['parameter_adjustments']['kernel_sizes'] = 'small'
            recommendations['strategy_priority'].extend(
                ['BilateralFilter', 'CLAHE'])

        if reference_analysis['has_varied_sizes']:
            recommendations['preprocessing_steps'].append(
                'multi_scale_processing')
            recommendations['warnings'].append(
                'Различный размер features - требуется многоуровневая обработка')

        if reference_analysis['regularity']['has_regular_pattern']:
            recommendations['preprocessing_steps'].append(
                'pattern_aware_processing')
            recommendations['parameter_adjustments']['tile_sizes'] = 'small'

        # Приоритизация стратегий
        if not recommendations['strategy_priority']:
            recommendations['strategy_priority'] = [
                'CLAHE+Bilateral', 'GaussianBlur', 'BackgroundSubtraction',
                'UnsharpMask', 'GammaCorrection', 'MedianBlur'
            ]

        return recommendations
