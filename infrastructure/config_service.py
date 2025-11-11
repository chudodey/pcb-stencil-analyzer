# infrastructure/config_service.py
"""
Сервис конфигурации.
Загружает, хранит и валидирует все параметры приложения.
"""

import configparser
from pathlib import Path
import random
from typing import Dict, Any  # 🔧 Добавлен typing


class ConfigService:
    """Централизованное управление конфигурацией приложения."""

    def __init__(self, config_file: str = 'config.ini', debug_mode: bool = False) -> None:
        self.config_file = Path(config_file)
        self.debug_mode = debug_mode
        self._config = configparser.ConfigParser()
        self._validated = False

        self._load_config()

    def _load_config(self) -> None:
        """Загружает конфигурацию из существующего файла."""
        if not self.config_file.exists():
            raise FileNotFoundError(
                f"Файл конфигурации '{self.config_file.name}' не найден. "
                "Скачайте его из репозитория и положите рядом с main.py."
            )

        self._config.read(self.config_file, encoding='utf-8')

        # Переопределение режима отладки из параметров запуска
        if self.debug_mode:
            self._config.set('GENERAL', 'debug_mode', 'true')

        # --- Секция GENERAL ---
        self.default_operator_name = self._config.get(
            'GENERAL', 'default_operator_name', fallback='Иванов И.И.')
        self.order_number_digits = self._config.getint(
            'GENERAL', 'order_number_digits', fallback=6)
        self.debug_mode = self._config.getboolean(
            'GENERAL', 'debug_mode', fallback=False)

        # --- Секция PATHS ---
        self.gerber_folder = self._get_path('PATHS', 'gerber_folder')
        self.scan_folder = self._get_path('PATHS', 'scan_folder')
        self.output_folder = self._get_path('PATHS', 'output_folder')
        self.create_order_subfolder = self._config.getboolean(
            'PATHS', 'create_order_subfolder', fallback=True)

        # --- Секция SCANNING ---
        self.scan_instruction = self._config.get(
            'SCANNING', 'scan_instruction', fallback='')
        self.file_check_interval = self._config.getfloat(
            'SCANNING', 'file_check_interval', fallback=3.0)
        self.default_dpi = self._config.getint(
            'SCANNING', 'default_dpi', fallback=600)
        self.dpi_priority = self._config.get(
            'SCANNING', 'dpi_priority', fallback='metadata')
        self.supported_image_formats = [fmt.strip() for fmt in self._config.get(
            'SCANNING', 'supported_image_formats', fallback='.png,.jpg,.jpeg,.tiff,.tif,.bmp').split(',')]

        # --- Секция GERBER_SEARCH ---
        self.multiple_files_rule = self._config.get(
            'GERBER_SEARCH', 'multiple_files_rule', fallback='alphabetic_first')

        # --- Секция GERBER_PROCESSING ---
        self.gerber_margin_mm = self._config.getfloat(
            'GERBER_PROCESSING', 'margin_mm', fallback=2.0)

        # --- Секция IMAGE_PREPROCESSING ---
        self.size_tolerance_percent = self._config.getfloat(
            'IMAGE_PREPROCESSING', 'size_tolerance_percent', fallback=5.0)
        self.partial_preprocessing_action = self._config.get(
            'IMAGE_PREPROCESSING', 'partial_preprocessing_action', fallback='ignore')

        # --- Секция EVALUATOR (новая) ---
        self.boost_factor = self._config.getfloat(
            'EVALUATOR', 'boost_factor', fallback=1.3)
        self.penalty_factor = self._config.getfloat(
            'EVALUATOR', 'penalty_factor', fallback=0.7)
        self.min_boundary_match_threshold = self._config.getfloat(
            'EVALUATOR', 'min_boundary_match_threshold', fallback=0.3)

        # --- Секция IMAGE_COMPARISON ---
        self.consider_reflection = self._config.getboolean(
            'IMAGE_COMPARISON', 'consider_reflection', fallback=False)
        self.rotation_angles = [int(angle.strip()) for angle in self._config.get(
            'IMAGE_COMPARISON', 'rotation_angles', fallback='0,90,180,270').split(',')]
        self.min_contour_coefficient = self._config.getfloat(
            'IMAGE_COMPARISON', 'min_contour_coefficient', fallback=2.0)
        self.max_contour_coefficient = self._config.getfloat(
            'IMAGE_COMPARISON', 'max_contour_coefficient', fallback=2.0)
        self.ransac_reprojection_threshold = self._config.getfloat(
            'IMAGE_COMPARISON', 'ransac_reprojection_threshold', fallback=3.0)
        self.max_iterations = self._config.getint(
            'IMAGE_COMPARISON', 'max_iterations', fallback=2000)
        self.confidence = self._config.getfloat(
            'IMAGE_COMPARISON', 'confidence', fallback=0.99)
        self.refine_iterations = self._config.getint(
            'IMAGE_COMPARISON', 'refine_iterations', fallback=10)
        self.low_correlation_threshold = self._config.getfloat(
            'IMAGE_COMPARISON', 'low_correlation_threshold', fallback=0.2)
        self.medium_correlation_threshold = self._config.getfloat(
            'IMAGE_COMPARISON', 'medium_correlation_threshold', fallback=0.4)
        self.high_correlation_threshold = self._config.getfloat(
            'IMAGE_COMPARISON', 'high_correlation_threshold', fallback=0.8)

        # --- Секция VISUALIZATION ---
        self.reference_color = (self._config.getint('VISUALIZATION', 'reference_color_r', fallback=255), self._config.getint(
            'VISUALIZATION', 'reference_color_g', fallback=0), self._config.getint('VISUALIZATION', 'reference_color_b', fallback=0))
        self.scan_color = (self._config.getint('VISUALIZATION', 'scan_color_r', fallback=0), self._config.getint(
            'VISUALIZATION', 'scan_color_g', fallback=255), self._config.getint('VISUALIZATION', 'scan_color_b', fallback=255))
        self.intersection_color = (self._config.getint('VISUALIZATION', 'intersection_color_r', fallback=255), self._config.getint(
            'VISUALIZATION', 'intersection_color_g', fallback=255), self._config.getint('VISUALIZATION', 'intersection_color_b', fallback=255))
        self.info_font_size = self._config.getint(
            'VISUALIZATION', 'info_font_size', fallback=20)
        self.info_text_color = (self._config.getint('VISUALIZATION', 'info_text_color_r', fallback=255), self._config.getint(
            'VISUALIZATION', 'info_text_color_g', fallback=255), self._config.getint('VISUALIZATION', 'info_text_color_b', fallback=255))
        self.info_background_color = (self._config.getint('VISUALIZATION', 'info_background_color_r', fallback=0), self._config.getint(
            'VISUALIZATION', 'info_background_color_g', fallback=0), self._config.getint('VISUALIZATION', 'info_background_color_b', fallback=0))

        # --- Секция OUTPUT ---
        self.save_intermediate_images = self._config.getboolean(
            'OUTPUT', 'save_intermediate_images', fallback=False) or self.debug_mode
        self.save_final_image = self._config.getboolean(
            'OUTPUT', 'save_final_image', fallback=True)
        self.existing_files_action = self._config.get(
            'OUTPUT', 'existing_files_action', fallback='increment')
        self.gerber_image_filename = self._config.get(
            'OUTPUT', 'gerber_image_filename', fallback='{order_number}_1_gerber.png')
        self.original_scan_filename = self._config.get(
            'OUTPUT', 'original_scan_filename', fallback='{order_number}_2_scan.png')
        self.processed_scan_filename = self._config.get(
            'OUTPUT', 'processed_scan_filename', fallback='{order_number}_3_scan_prep.png')
        self.comparison_result_filename = self._config.get(
            'OUTPUT', 'comparison_result_filename', fallback='{order_number}_4_compared.png')
        self.detailed_log_filename = self._config.get(
            'OUTPUT', 'detailed_log_filename', fallback='{order_number}_5_log_detailed.txt')
        self.short_log_filename = self._config.get(
            'OUTPUT', 'short_log_filename', fallback='{order_number}_6_log_short.txt')

        # --- Секция LOGGING ---
        self.log_level = self._config.get(
            'LOGGING', 'log_level', fallback='short')
        self.datetime_format = self._config.get(
            'LOGGING', 'datetime_format', fallback='%Y-%m-%d %H:%M:%S')
        self.short_log_template = self._config.get(
            'LOGGING', 'short_log_template', fallback='Дата и время: {datetime}\nОператор: {operator_name}\nНомер заказа: {order_number}\nРезультат корреляции: {correlation_result:.3f}\nСтатус совмещения: {alignment_status}')
        self.include_debug_info = self._config.getboolean(
            'LOGGING', 'include_debug_info', fallback=False) or self.debug_mode

        # --- Секция SYSTEM ---
        self.max_scan_file_size = self._config.getint(
            'SYSTEM', 'max_scan_file_size', fallback=100)
        self.scan_wait_timeout = self._config.getint(
            'SYSTEM', 'scan_wait_timeout', fallback=300)

        # --- Секция PYTHON_REQUIREMENTS ---
        self.min_python_version = self._config.get(
            'PYTHON_REQUIREMENTS', 'min_python_version', fallback='3.8')
        self.required_modules = [mod.strip() for mod in self._config.get(
            'PYTHON_REQUIREMENTS', 'required_modules', fallback='').split(',') if mod.strip()]
        self.install_command = self._config.get(
            'PYTHON_REQUIREMENTS', 'install_command', fallback='pip install opencv-python numpy Pillow matplotlib scipy')

        self._validate_config()

    def _get_path(self, section: str, option: str) -> Path:
        """Получает путь из конфигурации."""
        return Path(self._config.get(section, option))

    def _validate_config(self) -> None:
        """Валидация параметров конфигурации."""
        if self.order_number_digits <= 0:
            raise ValueError(
                "Количество цифр в номере заказа должно быть положительным")
        if self.file_check_interval <= 0:
            raise ValueError(
                "Интервал проверки файлов должен быть положительным")
        if not 0 <= self.low_correlation_threshold <= self.medium_correlation_threshold <= self.high_correlation_threshold <= 1:
            raise ValueError(
                "Пороги корреляции должны быть в диапазоне [0,1] и возрастающими")

    def generate_example_order_number(self) -> str:
        """Генерирует пример номера заказа для подсказки пользователю."""
        return ''.join([str(random.randint(0, 9)) for _ in range(self.order_number_digits)])

    def get_filename(self, template: str, order_number: str, workspace: Path) -> Path:
        """Генерирует имя файла на основе шаблона и обрабатывает конфликты имен."""
        filename = template.format(order_number=order_number)
        filepath = workspace / filename

        if self.existing_files_action == 'increment' and filepath.exists():
            counter = 1
            name, ext = filepath.stem, filepath.suffix
            while True:
                new_name = f"{name}_{counter}{ext}"
                new_filepath = workspace / new_name
                if not new_filepath.exists():
                    return new_filepath
                counter += 1
        return filepath

    # 🔧 Новый метод для evaluator (решает ошибку в StageRunner)
    def get_evaluator_config(self) -> Dict[str, Any]:
        """
        Возвращает конфиг для StrategyEvaluator.
        """
        return {
            'quality_thresholds': {
                'preprocessing': 0.5,  # Можно взять из INI, если добавить
                'binarization': 0.4,
                'roi_extraction': 0.5,
                'alignment': 0.5
            },
            'boost_factor': self.boost_factor,
            'penalty_factor': self.penalty_factor,
            'min_boundary_match_threshold': self.min_boundary_match_threshold
        }

    # 🔧 Новый метод для пайплайна/StageRunner
    def get_pipeline_config(self) -> Dict[str, Any]:
        """
        Возвращает конфиг для пайплайна (e.g., для StageRunner).
        """
        return {
            'debug_mode': self.debug_mode,
            # Если добавите в INI
            'pipeline_strategy': getattr(self, 'pipeline_strategy', 'optimized'),
            # Другие: e.g., 'file_check_interval': self.file_check_interval
        }
