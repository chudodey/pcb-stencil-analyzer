# infrastructure/ui_service.py
"""
Сервис для взаимодействия с пользователем через консольный интерфейс (UI).

ОТВЕТСТВЕННОСТЬ:
- Рабочий вывод для оператора (информация, успехи, ошибки, предупреждения)
- Единообразное форматирование всех сообщений для пользователя
- Профессиональный визуальный стиль
- Четкая иерархия информации
- НЕ используется для отладочного вывода (для этого есть DebugFormatter)
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# pylint: disable=no-member
import cv2

import numpy as np
from screeninfo import get_monitors

from domain.data_models import ProcessedScan, calculate_alignment_status

from .config_service import ConfigService
from .file_manager import FileManager
from .logging_service import LoggingService


class UIService:
    """
    Профессиональный текстовый интерфейс для оператора.

    Используется ТОЛЬКО для рабочего вывода информации пользователю.
    Для отладочного вывода используйте DebugFormatter.
    """

    # Константы для типов сообщений (только рабочие, без отладки)
    MSG_INFO = "info"
    MSG_SUCCESS = "success"
    MSG_WARNING = "warning"
    MSG_ERROR = "error"
    MSG_ACTION = "action"  # Для запросов действий от пользователя

    def __init__(self, config_service: ConfigService):
        """
        Инициализирует UI сервис.

        Args:
            config_service: Сервис конфигурации.
        """
        self.config_service = config_service
        self.logger = LoggingService.get_logger(__name__)

        # Шаблоны БЕЗ префиксов (для заголовков)
        self._templates = {
            "header": "СИСТЕМА КОНТРОЛЯ КАЧЕСТВА ТРАФАРЕТОВ v2.0",
            "stage_divider": "─" * 40,
            "result_divider": "─" * 50,
            "main_menu_header": "─" * 30
        }

        # Префиксы для типов сообщений
        self._prefixes = {
            self.MSG_INFO: "[INFO]",
            self.MSG_SUCCESS: "[OK]",
            self.MSG_WARNING: "[WARN]",
            self.MSG_ERROR: "[ERROR]",
            self.MSG_ACTION: "[ACT]",
        }

        # Выравниваем префиксы по максимальной длине
        self._align_prefixes()

        # Extra-данные для логгера (БЕЗ is_debug_formatter - значит белый цвет)
        self._log_extra = {"className": self.__class__.__name__}

        # Счетчик этапов
        self._current_stage = 0
        self._total_stages = 6

    def _align_prefixes(self):
        """Выравнивает префиксы по максимальной длине."""
        max_length = max(len(prefix) for prefix in self._prefixes.values())
        self._prefixes = {
            msg_type: f"{prefix:<{max_length}}"
            for msg_type, prefix in self._prefixes.items()
        }

    # ========================================================================
    # БАЗОВЫЕ МЕТОДЫ ВЫВОДА
    # ========================================================================

    def show_message(self, text: str, msg_type: str = MSG_INFO,
                     add_newline: bool = False, indent: int = 0) -> None:
        """
        Централизованный вывод сообщения с префиксом в колонке 0.

        Args:
            text: Текст сообщения
            msg_type: Тип сообщения
            add_newline: Добавить пустую строку перед сообщением
            indent: Уровень отступа для ТЕКСТА (0 = нет, 1 = 2 пробела, 2 = 4 пробела)
        """
        prefix = self._prefixes.get(msg_type, self._prefixes[self.MSG_INFO])

        # ВАЖНО: Отступ применяется ТОЛЬКО к тексту, НЕ к префиксу
        indent_str = " " * (indent * 2)
        message = f"{prefix} {indent_str}{text}"

        if add_newline:
            message = f"\n{message}"

        # Логируем БЕЗ is_debug_formatter (белый цвет по умолчанию)
        log_extra = self._log_extra

        # Выбираем метод логгера в зависимости от типа
        if msg_type == self.MSG_ERROR:
            self.logger.error(message, extra=log_extra)
        elif msg_type == self.MSG_WARNING:
            self.logger.warning(message, extra=log_extra)
        else:
            self.logger.info(message, extra=log_extra)

    def show_success(self, text: str, add_newline: bool = False, indent: int = 0) -> None:
        """Отображает сообщение об успехе."""
        self.show_message(text, self.MSG_SUCCESS, add_newline, indent)

    def show_warning(self, text: str, add_newline: bool = False, indent: int = 0) -> None:
        """Отображает предупреждение."""
        self.show_message(text, self.MSG_WARNING, add_newline, indent)

    def show_error(self, text: str, add_newline: bool = False, indent: int = 0) -> None:
        """Отображает сообщение об ошибке."""
        self.show_message(text, self.MSG_ERROR, add_newline, indent)

    def show_action(self, text: str, add_newline: bool = False, indent: int = 0) -> None:
        """Отображает запрос действия."""
        self.show_message(text, self.MSG_ACTION, add_newline, indent)

    # ========================================================================
    # СТРУКТУРНЫЕ ЭЛЕМЕНТЫ (БЕЗ ПРЕФИКСОВ)
    # ========================================================================

    def _log_without_prefix(self, text: str) -> None:
        """
        Логирует текст БЕЗ префикса (для заголовков, разделителей).

        Используется для элементов, которые не должны иметь префиксов:
        - Заголовки
        - Разделители
        - Рамки
        """
        # Используем is_preformatted чтобы избежать добавления префикса
        log_extra = {
            **self._log_extra,
            "is_preformatted": True
        }
        self.logger.info(text, extra=log_extra)

    def show_header(self) -> None:
        """Отображает профессиональный заголовок программы БЕЗ префикса."""
        self._log_without_prefix(f"\n{self._templates['header']}")
        if self.config_service.debug_mode:
            # Режим отладки - это обычное сообщение, с префиксом
            self.show_message("Режим отладки включен", self.MSG_INFO)
        self._log_without_prefix("")

    def show_main_stage(self, stage_title: str) -> None:
        """
        Отображает заголовок основного этапа БЕЗ префикса.
        """
        self._current_stage += 1
        stage_header = f"[{self._current_stage}/{self._total_stages}] {stage_title.upper()}"

        # Заголовок и разделитель БЕЗ префиксов
        self._log_without_prefix(f"\n{stage_header}")
        self._log_without_prefix(self._templates["stage_divider"])

    def separator(self, length: int = 40) -> None:
        """Отображает разделитель БЕЗ префикса."""
        self._log_without_prefix("─" * length)

    # ========================================================================
    # ДРЕВОВИДНАЯ СТРУКТУРА
    # ========================================================================

    def show_tree_item(self, text: str, level: int = 1, is_last: bool = False) -> None:
        """
        Отображает элемент древовидной структуры.

        Древовидные элементы - это обычные сообщения с префиксом,
        но текст включает символы дерева и имеет отступ.

        Args:
            text: Текст элемента
            level: Уровень вложенности (1, 2, 3)
            is_last: Является ли последним элементом
        """
        prefixes = {
            1: {"middle": "├─", "last": "└─"},
            2: {"middle": "│  ├─", "last": "│  └─"},
            3: {"middle": "│     ├─", "last": "│     └─"}
        }

        level_data = prefixes.get(level, prefixes[1])
        tree_symbol = level_data["last"] if is_last else level_data["middle"]

        # Вызываем show_message с отступом 0 (так как символ дерева уже содержит отступ)
        self.show_message(f"{tree_symbol} {text}", indent=0)

    # ========================================================================
    # БЛОКИ С ДАННЫМИ
    # ========================================================================

    def show_compact_block(self, title: str, items: List[Tuple[str, str]]) -> None:
        """
        Отображает компактный блок информации в формате ключ-значение.

        Args:
            title: Заголовок блока (обычное сообщение с префиксом)
            items: Список кортежей (ключ, значение) - как элементы дерева
        """
        # Заголовок - обычное сообщение
        self.show_message(title, self.MSG_INFO, add_newline=True)

        # Элементы - как дерево с правильным завершением
        for i, (key, value) in enumerate(items):
            is_last = (i == len(items) - 1)
            self.show_tree_item(f"{key}: {value}", is_last=is_last)

    def show_recommendations(self, title: str, items: List[str]) -> None:
        """
        Отображает блок рекомендаций с древовидной структурой.

        Args:
            title: Заголовок блока
            items: Список элементов
        """
        self.show_message(title, self.MSG_INFO, add_newline=False)

        for i, item in enumerate(items):
            is_last = (i == len(items) - 1)
            # indent=1 для вложенности рекомендаций
            prefix = "└─" if is_last else "├─"
            self.show_message(f"{prefix} {item}", self.MSG_INFO, indent=1)

    # ========================================================================
    # СПЕЦИАЛИЗИРОВАННЫЕ МЕТОДЫ ВЫВОДА
    # ========================================================================

    def show_processing_step(self, text: str, status: str = MSG_INFO, indent: int = 0) -> None:
        """
        Отображает шаг обработки с отступом для лучшей читаемости.
        """
        self.show_message(text, status, indent=indent)

    def show_environment_check(self, issues: List[str]) -> bool:
        """
        Показывает результаты проверки окружения в компактном формате.
        """
        if not issues:
            self.show_success("Проверка окружения пройдена успешно")
            return True

        self.show_error("Обнаружены проблемы с окружением:")
        for i, issue in enumerate(issues):
            is_last = (i == len(issues) - 1)
            self.show_tree_item(issue, is_last=is_last)
        return False

    def show_gerber_search_result(self, files: List[Path], order_number: str) -> None:
        """Отображает результаты поиска Gerber-файлов."""
        if not files:
            self.show_error(
                f"Gerber-файл для заказа '{order_number}' не найден")
            self.show_message("Проверьте папку Gerber", self.MSG_INFO)
            return

        if len(files) == 1:
            self.show_success(f"Найден файл: {files[0].name}")
        else:
            rule_map = {
                "alphabetic_first": "первый по алфавиту",
                "newest": "самый новый",
            }
            rule_text = rule_map.get(
                self.config_service.multiple_files_rule, "первый")
            self.show_warning(
                f"Найдено несколько файлов. Выбран {rule_text}: {files[0].name}"
            )

    def show_parsing_stats(self, metrics: Dict) -> None:
        """Отображает статистику парсинга в компактном формате."""
        self.show_success("Gerber-файл успешно проанализирован")

        stats_items = []
        if 'board_width_mm' in metrics and metrics['board_width_mm'] != 'N/A':
            stats_items.append(
                ("Ширина платы", f"{metrics['board_width_mm']:.2f} мм"))
        if 'board_height_mm' in metrics and metrics['board_height_mm'] != 'N/A':
            stats_items.append(
                ("Высота платы", f"{metrics['board_height_mm']:.2f} мм"))
        if 'contour_count' in metrics and metrics['contour_count'] != 'N/A':
            stats_items.append(("Контуры", str(metrics['contour_count'])))
        if 'processing_time' in metrics:
            stats_items.append(
                ("Время обработки", f"{metrics['processing_time']:.2f} сек"))

        if stats_items:
            self.show_compact_block("Параметры платы:", stats_items)

    def show_scanning_instructions(self) -> None:
        """Отображает инструкции по сканированию."""
        instructions = [self.config_service.scan_instruction]
        self.show_recommendations("ИНСТРУКЦИЯ ПО СКАНИРОВАНИЮ:", instructions)

    def show_scan_info(self, scan_path: Path, dpi: float, size_pixels: tuple, size_mm: tuple, file_size: int) -> None:
        """Отображает информацию о скане в компактном формате."""
        dpi_source = "метаданные" if self.config_service.dpi_priority == "metadata" else "конфигурация"

        scan_items = [
            ("Файл", scan_path.name),
            ("DPI", f"{dpi} ({dpi_source})"),
            ("Размер",
             f"{size_pixels[0]}×{size_pixels[1]} px ({size_mm[0]:.1f}×{size_mm[1]:.1f} мм)"),
            ("Размер файла", FileManager.format_file_size(file_size)),
        ]
        self.show_compact_block("ИНФОРМАЦИЯ О СКАНЕ:", scan_items)

    def show_alignment_results(self, metrics: Dict, indent: int = 0) -> None:
        """Отображает результаты совмещения в профессиональном формате."""
        correlation = metrics.get("correlation", 0.0)
        status_enum, status_text = calculate_alignment_status(
            correlation,
            {
                "high": self.config_service.high_correlation_threshold,
                "medium": self.config_service.medium_correlation_threshold,
            },
        )

        # Разделитель БЕЗ префикса
        self._log_without_prefix(f"\n{self._templates['result_divider']}")

        result_items = [
            ("Корреляция",
             f"{correlation:.3f} (требуется > {self.config_service.medium_correlation_threshold})"),
            ("Статус", status_text)
        ]

        self.show_compact_block("РЕЗУЛЬТАТЫ СОВМЕЩЕНИЯ:", result_items)

        # Рекомендации только при ошибках
        if status_enum.value == "failed":
            recommendations = [
                "Проверьте разрешение скана (1200 DPI)",
                "Убедитесь в правильной ориентации трафарета",
                "Проверьте чистоту маркеров совмещения"
            ]
            self.show_recommendations("Рекомендации:", recommendations)

        if self.config_service.debug_mode:
            debug_items = [
                ("Поворот", f"{metrics.get('rotation', 0):+.2f}°"),
                ("Сдвиг X", f"{metrics.get('offset_x_px', 0):+.1f} px"),
                ("Сдвиг Y", f"{metrics.get('offset_y_px', 0):+.1f} px")
            ]
            self.show_compact_block("Отладочная информация:", debug_items)

        # Разделитель БЕЗ префикса
        self._log_without_prefix(self._templates["result_divider"])

    def show_files_saved(self, saved_files: List[Path]) -> None:
        """
        Отображает информацию о сохраненных файлах.

        Args:
            saved_files: Список путей к сохраненным файлам.
        """
        if not saved_files:
            return

        self.show_success("Сохранены файлы результатов:", add_newline=True)
        for file_path in saved_files:
            self.show_message(f"  {file_path.resolve()}", self.MSG_INFO)

    def show_combined_image(self, image: np.ndarray, title: str = "Результат сравнения") -> None:
        """
        Отображает совмещенное изображение с масштабированием под экран.

        Args:
            image: Изображение для отображения.
            title: Заголовок окна.
        """
        try:
            monitor = get_monitors()[0]
            screen_w, screen_h = monitor.width, monitor.height
            h, w = image.shape[:2]

            # Рассчитываем масштаб для отображения
            scale = min((screen_w * 0.8) / w, (screen_h * 0.8) /
                        h) if w > 0 and h > 0 else 1.0

            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                display_image = cv2.resize(
                    image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                display_image = image
                new_w, new_h = w, h

            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, new_w, new_h)
            cv2.imshow(title, display_image)
            cv2.waitKey(0)

        except Exception as e:
            self.show_error(f"Не удалось отобразить изображение: {e}")
        finally:
            cv2.destroyAllWindows()

    # ========================================================================
    # Методы ввода данных
    # ========================================================================

    def get_operator_name(self) -> str:
        """
        Запрашивает ФИО оператора с валидацией.

        Returns:
            Введенное ФИО или значение по умолчанию.
        """
        default = self.config_service.default_operator_name
        prompt = f'Введите ваше ФИО (Enter для "{default}"): '

        name = input(prompt).strip()
        return name if name else default

    def get_order_number(self) -> Optional[str]:
        """Запрашивает номер заказа с валидацией."""
        digits = self.config_service.order_number_digits
        example = self.config_service.generate_example_order_number()
        prompt = (
            f"\nВведите {digits}-значный номер заказа (пример: {example}) "
            "или 'exit' для выхода: "
        )
        while True:
            value = input(prompt).strip()
            if value.lower() == "exit":
                return None
            if re.fullmatch(rf"\d{{{digits}}}", value):
                return value
            self.show_error(f"Номер должен содержать ровно {digits} цифр.")

    def ask_user_confirmation(self, question: str) -> bool:
        """
        Задает простой вопрос да/нет.

        Args:
            question: Текст вопроса.

        Returns:
            True если пользователь подтвердил, False если отказался.
        """
        while True:
            response = input(f"{question} (y/n): ").strip().lower()
            if response in ["y", "yes", "да", "д"]:
                return True
            if response in ["n", "no", "нет", "н"]:
                return False
            self.show_error("Некорректный ввод. Введите 'y' или 'n'.")

    # ========================================================================
    # Анимации и интерактивные запросы
    # ========================================================================

    def show_scan_waiting_animation(self, timeout: float, elapsed: float) -> None:
        """
        Показывает анимацию ожидания скана в той же строке консоли.

        Примечание: использует print() вместо логгера для использования
        спецсимвола '\r' (возврат каретки), что невозможно в стандартном логгере.
        """
        if elapsed < 0.2:  # Логируем начало ожидания только один раз
            self.logger.debug(
                "Начато ожидание нового файла скана...", extra=self._log_extra)

        animation_chars = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
        anim_char = animation_chars[int(elapsed * 10) % len(animation_chars)]

        # Исправляем отрицательное время
        remaining = max(0, int(timeout - elapsed)) if timeout > 0 else "∞"

        # Прямой вывод в stdout для интерактивной анимации
        print(
            f"\r{anim_char} Ожидание скана... (осталось: {remaining} сек) ",
            end="",
            flush=True,
        )

    def ask_choice(self, title: str, options: Dict[str, str]) -> str:
        """
        Универсальный метод для запроса выбора у пользователя.

        Args:
            title: Вопрос или заголовок меню.
            options: Словарь, где ключ - номер опции, а значение - её описание.

        Returns:
            Выбранный пользователем ключ из словаря options.
        """
        self.show_message(title, self.MSG_INFO, add_newline=True)
        for key, value in options.items():
            self.show_message(f"  {key} — {value}", self.MSG_INFO)

        while True:
            choice = input("Ваш выбор: ").strip()
            if choice in options:
                # Очищаем строку после анимации или предыдущего вывода
                print("\r" + " " * 80 + "\r", end="")
                return choice
            self.show_error(
                "Некорректный ввод. Пожалуйста, выберите один из вариантов.")

    def show_main_menu(self) -> str:
        """
        Отображает главное меню программы.

        Returns:
            Выбранный пользователем пункт меню.
        """
        self._log_without_prefix(f"\n{self._templates['main_menu_header']}")
        self._log_without_prefix("ГЛАВНОЕ МЕНЮ")
        self._log_without_prefix(self._templates['main_menu_header'])

        options = {
            "1": "Начать новый заказ",
            "2": "Выйти из программы"
        }
        return self.ask_choice("", options)

    def ask_scan_timeout_action(self, last_existing_file: Optional[Path]) -> str:
        """Запрашивает действие при таймауте сканирования."""
        self.show_warning("Время ожидания скана истекло.", add_newline=True)
        options = {"1": "Продолжить ожидание"}
        if last_existing_file:
            options["2"] = f"Использовать последний файл: {last_existing_file.name}"
        options["3"] = "Ввести новый номер заказа"
        options["4"] = "Выйти из программы"
        return self.ask_choice("Выберите действие:", options)

    def ask_parsing_failed_action(self, error_msg: str) -> str:
        """Запрос действия при ошибке парсинга Gerber."""
        self.show_error(f"Ошибка анализа Gerber-файла: {error_msg}")
        self.show_message("Рекомендации:", self.MSG_INFO)
        self.show_message("  - Проверьте формат Gerber-файла", self.MSG_INFO)
        self.show_message(
            "  - Убедитесь, что файл не поврежден", self.MSG_INFO)

        options = {"1": "Ввести новый номер заказа", "2": "Выйти"}
        return self.ask_choice("Выберите действие:", options)

    def ask_alignment_failed_action(self, correlation: float) -> str:
        """
        Запрос действия при неудачном совмещении.

        Args:
            correlation: Значение корреляции для детализации сообщения.
        """
        if correlation < self.config_service.low_correlation_threshold:
            self.show_error("Совмещение не удалось (очень низкая корреляция).")
        else:
            self.show_warning("Совмещение сомнительно (низкая корреляция).")

        options = {
            "1": "Повторить сканирование",
            "2": "Ввести новый номер заказа",
            "3": "Выйти",
        }
        return self.ask_choice("Выберите действие:", options)

    def confirm_exit(self) -> bool:
        """Подтверждение выхода из программы."""
        return self.ask_user_confirmation("\nВы действительно хотите выйти?")

    def show_processing_error(self, error_type: str, error_details: str = "") -> None:
        """
        Показывает информативное сообщение об ошибке обработки.

        Args:
            error_type: Тип ошибки ('alignment', 'io', 'technical')
            error_details: Детали ошибки для отладки
        """
        if error_type == 'alignment':
            self.show_error_block(
                error_title="Ошибка совмещения изображений",
                causes=[
                    "Низкое качество скана",
                    "Несоответствие Gerber-файлу",
                    "Проблемы с маркерами совмещения"
                ],
                debug_details=error_details
            )
        elif error_type == 'io':
            self.show_error_block(
                error_title="Ошибка чтения/записи файлов",
                checks=[
                    "Доступность файлов",
                    "Права доступа к папкам"
                ],
                debug_details=error_details
            )
        else:  # technical
            self.show_error_block(
                error_title="Техническая ошибка обработки",
                debug_details=error_details
            )

    # ========================================================================
    # НОВЫЕ МЕТОДЫ ДЛЯ МИГРАЦИИ ИЗ COORDINATOR
    # ========================================================================

    def show_scan_details(self,
                          scan_path: Path,
                          dpi: int,
                          size_pixels: Tuple[int, int],
                          size_mm: Tuple[float, float],
                          file_size: int) -> None:
        """
        Показывает детальную информацию о скане.

        Этот метод объединяет вызовы show_scan_info и дополнительную логику
        форматирования, которая ранее была разбросана по координатору.

        Args:
            scan_path: Путь к файлу скана
            dpi: Разрешение в точках на дюйм
            size_pixels: Размер в пикселях (ширина, высота)
            size_mm: Размер в миллиметрах (ширина, высота)
            file_size: Размер файла в байтах
        """
        # Используем существующий метод show_scan_info
        self.show_scan_info(scan_path, dpi, size_pixels, size_mm, file_size)

        # Дополнительная информация в debug режиме
        if self.config_service.debug_mode:
            debug_items = [
                ("Полный путь", str(scan_path.resolve())),
                ("Соотношение сторон",
                 f"{size_pixels[0]/size_pixels[1]:.2f}:1" if size_pixels[1] > 0 else "N/A")
            ]
            self.show_compact_block("Дополнительно:", debug_items)

    def display_alignment_summary(self, processed_scan: ProcessedScan) -> None:
        """
        Отображает детальную сводку результатов совмещения.
        (Исправлено: теперь использует вложенный объект AlignmentMetrics)
        """
        alignment = processed_scan.alignment
        # Получаем вложенный объект с координатами
        metrics_obj = alignment.alignment_metrics

        # Собираем словарь метрик, используя новые имена атрибутов из AlignmentMetrics
        # и старые имена ключей для вывода в show_alignment_results.
        metrics = {
            'correlation': alignment.alignment_score,

            # ИСПРАВЛЕНО: Доступ к координатам через metrics_obj и использование shift_x_px
            'offset_x_px': metrics_obj.shift_x_px,
            'offset_y_px': metrics_obj.shift_y_px,

            # ИСПРАВЛЕНО: Доступ к углу через metrics_obj и использование rotation_angle
            'rotation_deg': metrics_obj.rotation_angle,

            # Предполагаем, что strategy_name и processing_time остались в ProcessedScan
            # или их нужно получить откуда-то еще (используем безопасный доступ).
            'strategy': getattr(alignment, 'strategy_name', "N/A"),
            'processing_time': getattr(alignment, 'processing_time', None),
        }

        # Вывод статуса
        if alignment.is_aligned:
            self.show_success(
                f"Совмещение успешно завершено: {alignment.alignment_status.value.upper()}")
        else:
            self.show_error(
                f"Совмещение НЕ пройдено: {alignment.alignment_status.value.upper()}")

        # Детальный вывод метрик
        self.show_alignment_results(metrics, indent=1)

        # Дополнительная информация для отладки
        if self.config_service.debug_mode:
            # Эти поля доступны, так как мы используем ProcessedScan
            self.show_message(
                f"Путь скана: {processed_scan.scan_info.scan_path}", self.MSG_INFO, indent=1)
            self.show_message(
                f"Ссылочный Гербер: {processed_scan.stencil_reference.gerber_filename}", self.MSG_INFO, indent=1)

    def show_processing_progress(self, stage: str, step: str, current: int = 0, total: int = 0) -> None:
        """
        Показывает прогресс обработки с детализацией этапа и шага.

        Args:
            stage: Название этапа (например, "Предобработка")
            step: Название шага (например, "Применение фильтра Гаусса")
            current: Текущий номер шага (опционально)
            total: Общее количество шагов (опционально)
        """
        if total > 0:
            progress_text = f"[{current}/{total}] {stage}: {step}"
        else:
            progress_text = f"{stage}: {step}"

        self.show_processing_step(progress_text)

    def show_workspace_info(self, workspace: Path) -> None:
        """
        Показывает информацию о рабочей директории заказа.

        Args:
            workspace: Путь к рабочей директории
        """
        self.show_success(f"Рабочая папка: {workspace}")

        if self.config_service.debug_mode:
            # Показываем дополнительную информацию о директории
            try:
                file_count = len(list(workspace.glob("*")))
                self.logger.debug(
                    f"Файлов в папке: {file_count}", extra=self._log_extra)
            except Exception:
                pass

    def show_artifact_saved(self, artifact_name: str, file_path: Path) -> None:
        """
        Показывает информацию о сохранённом артефакте.

        Args:
            artifact_name: Название артефакта (например, "Финальное изображение")
            file_path: Путь к сохранённому файлу
        """
        if self.config_service.debug_mode:
            self.logger.debug(
                f"Сохранён {artifact_name}: {file_path.name}", extra=self._log_extra)

    def show_pipeline_stage(self, stage_name: str, strategy_count: int = 0) -> None:
        """
        Показывает начало этапа пайплайна обработки.

        Args:
            stage_name: Название этапа (например, "BINARIZATION")
            strategy_count: Количество зарегистрированных стратегий
        """
        if strategy_count > 0:
            self.show_processing_step(
                f"Этап: {stage_name} ({strategy_count} стратегий)"
            )
        else:
            self.show_processing_step(f"Этап: {stage_name}")

    def show_strategy_result(self, strategy_name: str, success: bool, score: float = 0.0) -> None:
        """
        Показывает результат применения стратегии обработки.

        Args:
            strategy_name: Название стратегии
            success: Успешность применения
            score: Оценка качества результата (опционально)
        """
        if not self.config_service.debug_mode:
            return

        status = "✓" if success else "✗"
        if score > 0:
            self.logger.debug(
                f"  {status} {strategy_name}: {score:.3f}", extra=self._log_extra)
        else:
            self.logger.debug(
                f"  {status} {strategy_name}", extra=self._log_extra)

    def show_order_summary(self, order_number: str, scan_count: int, success: bool) -> None:
        """
        Показывает краткую сводку по обработанному заказу.

        Args:
            order_number: Номер заказа
            scan_count: Количество обработанных сканов
            success: Общий статус успешности
        """
        self._log_without_prefix(f"\n{self._templates['result_divider']}")
        self._log_without_prefix("ИТОГИ ОБРАБОТКИ ЗАКАЗА")

        summary_items = [
            ("Номер заказа", order_number),
            ("Обработано сканов", str(scan_count)),
            ("Статус", "Успешно" if success else "С ошибками")
        ]

        for i, item in enumerate(summary_items):
            is_last = (i == len(summary_items) - 1)
            self.show_tree_item(f"{item[0]}: {item[1]}", is_last=is_last)

        self._log_without_prefix(self._templates['result_divider'])

    # ========================================================================
    # УЛУЧШЕННЫЕ МЕТОДЫ ДЛЯ ОБРАБОТКИ ОШИБОК
    # ========================================================================

    def show_gerber_error(self, order_number: str, error_details: str) -> None:
        """
        Показывает информативное сообщение об ошибке обработки Gerber.

        Args:
            order_number: Номер заказа
            error_details: Детали ошибки
        """
        self.show_error_block(
            error_title=f"Ошибка обработки Gerber для заказа {order_number}",
            causes=[
                "Файл повреждён или имеет некорректный формат",
                "Отсутствуют необходимые данные в файле",
                "Проблемы с кодировкой файла"
            ],
            checks=[
                "Проверьте целостность Gerber-файла",
                "Убедитесь, что файл соответствует стандарту RS-274X"
            ],
            debug_details=error_details
        )

    def show_scan_error(self, scan_path: Path, error_details: str) -> None:
        """
        Показывает информативное сообщение об ошибке обработки скана.

        Args:
            scan_path: Путь к файлу скана
            error_details: Детали ошибки
        """
        self.show_error_block(
            error_title=f"Ошибка обработки скана: {scan_path.name}",
            causes=[
                "Файл повреждён или имеет некорректный формат",
                "Не поддерживаемое разрешение или цветовая схема",
                "Недостаточный размер изображения"
            ],
            checks=[
                "Проверьте, что файл не повреждён",
                "Убедитесь в поддержке формата (PNG, JPG, TIFF)",
                "Проверьте разрешение (рекомендуется 1200 DPI)"
            ],
            debug_details=error_details
        )

    def show_error_block(self,
                         error_title: str,
                         causes: List[str] = None,
                         checks: List[str] = None,
                         debug_details: str = "") -> None:
        """
        Универсальный метод для отображения структурированного блока ошибки.

        Args:
            error_title: Заголовок ошибки
            causes: Список возможных причин
            checks: Список рекомендаций по проверке
            debug_details: Детали для отладки
        """
        self.show_error(error_title, add_newline=True)

        if causes:
            self.show_message("Возможные причины:", self.MSG_INFO)
            for cause in causes:
                self.show_tree_item(cause)

        if checks:
            self.show_message("Рекомендуется проверить:", self.MSG_INFO)
            for check in checks:
                self.show_tree_item(check)

        if debug_details and self.config_service.debug_mode:
            self.show_message("Детали ошибки:", self.MSG_INFO)
            self.show_tree_item(debug_details)

    def show_processing_summary(self, processed_scans: List[ProcessedScan]) -> None:
        """
        Показывает сводку по всем обработанным сканам.

        Args:
            processed_scans: Список обработанных сканов
        """
        if not processed_scans:
            return

        successful_scans = [
            scan for scan in processed_scans if scan.is_aligned]
        failed_scans = [
            scan for scan in processed_scans if not scan.is_aligned]

        self._log_without_prefix(f"\n{self._templates['result_divider']}")
        self._log_without_prefix("СВОДКА ПО ВСЕМ СКАНАМ")

        summary_items = [
            ("Всего сканов", str(len(processed_scans))),
            ("Успешно", str(len(successful_scans))),
            ("С ошибками", str(len(failed_scans))),
            ("Успешность",
             f"{(len(successful_scans) / len(processed_scans) * 100):.1f}%")
        ]

        for i, item in enumerate(summary_items):
            is_last = (i == len(summary_items) - 1)
            self.show_tree_item(f"{item[0]}: {item[1]}", is_last=is_last)

        if failed_scans:
            self.show_warning("Сканы с ошибками:", add_newline=True)
            for scan in failed_scans:
                self.show_tree_item(
                    f"{scan.scan_info.scan_path.name}: {scan.alignment.alignment_status.value}")

        self._log_without_prefix(self._templates['result_divider'])
