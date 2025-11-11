# infrastructure/debug_formatter.py
"""
Форматтер для структурированного отладочного вывода с поддержкой вложенности.

Ключевые принципы:
1. Префикс ВСЕГДА в колонке 0 (без отступов)
2. Отступы применяются ТОЛЬКО к тексту сообщения
3. Структурные элементы (заголовки, таблицы) БЕЗ префиксов
4. Серый цвет для ВСЕХ сообщений debug_formatter
5. Единая система отступов без магических чисел
"""

import time
from contextlib import contextmanager
from enum import Enum
from typing import Any, Dict, List, Optional

from .logging_service import LoggingService


class LogLevel(Enum):
    """Уровни логирования."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    SUCCESS = "OK"


class DebugFormatter:
    """
    Централизованный форматтер для единообразного отладочного вывода.

    Все сообщения выводятся СЕРЫМ цветом через флаг is_debug_formatter.
    Поддерживает многоуровневые вложенные структуры с правильными отступами.

    Attributes:
        PREFIX_WIDTH (int): Ширина префикса в символах
        DEFAULT_INDENT_SIZE (int): Размер отступа по умолчанию
        DEFAULT_TABLE_WIDTH (int): Ширина таблицы по умолчанию
        DEFAULT_HEADER_WIDTH (int): Ширина заголовка по умолчанию
        MAX_STRING_LENGTH (int): Максимальная длина строки
        MAX_LIST_PREVIEW (int): Максимальное количество элементов в превью списка
    """

    # Константы для отступов
    PREFIX_WIDTH = 8  # "[DEBUG] "
    DEFAULT_INDENT_SIZE = 2
    DEFAULT_TABLE_WIDTH = 60
    DEFAULT_HEADER_WIDTH = 80
    MAX_STRING_LENGTH = 50
    MAX_LIST_PREVIEW = 3

    # Символы для визуального оформления
    _SYMBOLS = {
        'header': '═',
        'subheader': '─',
        'box_tl': '┌',
        'box_tr': '┐',
        'box_bl': '└',
        'box_br': '┘',
        'box_v': '│',
        'box_h': '─',
        'bullet': '•',
        'arrow': '→',
        'success': '✓',
        'error': '✗',
        'warning': '⚠',
        'info': 'i',
        'progress': '█',
        'progress_empty': '░',
    }

    def __init__(self,
                 debug_mode: bool,
                 module_name: str = __name__,
                 indent_size: int = DEFAULT_INDENT_SIZE):
        """
        Инициализация форматтера.

        Args:
            debug_mode: Режим отладки (включить/выключить вывод)
            module_name: Имя модуля для логирования
            indent_size: Размер отступа в пробелах
        """
        self.debug_mode = debug_mode
        self.logger = LoggingService.get_logger(module_name)
        self.indent_size = indent_size
        self.indent_level = 0

        # ВАЖНО: Всегда устанавливаем is_debug_formatter для СЕРОГО цвета
        self._log_extra = {
            "className": self.__class__.__name__,
            "is_debug_formatter": True  # Маркер для серого цвета
        }

        # Конфигурация таблицы
        self.table_config = {
            'columns': [
                {'name': 'Стратегия', 'width': 32, 'align': '<'},
                {'name': 'Оценка', 'width': 10, 'align': '>'},
                {'name': 'Время (с)', 'width': 12, 'align': '>'}
            ],
            'spacing': {
                'between_columns': 1,
                'left_padding': 1,
                'right_padding': 1,
                'borders': True
            },
            'status_width': 2
        }
        self.table_config['total_width'] = self._calculate_table_width()

        # Вычисляем уровень отступа для префикса
        self.prefix_indent_level = self.PREFIX_WIDTH // self.indent_size

    def _calculate_table_width(self) -> int:
        """Вычисляет общую ширину таблицы без учета граничных символов."""
        spacing = self.table_config['spacing']
        columns_total = sum(col['width']
                            for col in self.table_config['columns'])
        between_cols = (
            len(self.table_config['columns']) - 1) * spacing['between_columns']
        side_padding = spacing['left_padding'] + spacing['right_padding']
        return columns_total + between_cols + side_padding

    def _calculate_content_indent(self, base_indent: int) -> int:
        """
        Вычисляет абсолютный отступ для контента на основе базового отступа.

        Args:
            base_indent: Базовый уровень отступа

        Returns:
            Абсолютный уровень отступа для контента
        """
        return self.prefix_indent_level + base_indent + 1

    # ========================================================================
    # БАЗОВЫЕ МЕТОДЫ ВЫВОДА (С ПРЕФИКСАМИ)
    # ========================================================================

    def _get_indent(self, level: int) -> str:
        """Получить строку пробелов для АБСОЛЮТНОГО уровня отступа."""
        if level <= 0:
            return ""
        return ' ' * (level * self.indent_size)

    def _format_prefix(self, level: LogLevel) -> str:
        """Получить выровненный префикс для уровня логирования."""
        prefixes = {
            LogLevel.DEBUG: "[DEBUG]",
            LogLevel.INFO: "[INFO] ",
            LogLevel.WARN: "[WARN] ",
            LogLevel.ERROR: "[ERROR]",
            LogLevel.SUCCESS: "[OK]   ",
        }
        return prefixes.get(level, "[INFO] ")

    def log(self, message: str, level: LogLevel = LogLevel.INFO,
            indent: Optional[int] = None) -> None:
        """
        Основной метод логирования с префиксом в колонке 0.

        Args:
            message: Текст сообщения
            level: Уровень логирования
            indent: Абсолютный уровень отступа. 
                    Если None, используется self.indent_level.
        """
        if level == LogLevel.DEBUG and not self.debug_mode:
            return

        # Определяем финальный уровень отступа
        final_indent_level = indent if indent is not None else self.indent_level
        indent_str = self._get_indent(final_indent_level)

        prefix = self._format_prefix(level)
        formatted_message = f"{prefix} {indent_str}{message}"
        extra_data = self._log_extra

        # Логируем в зависимости от уровня
        log_methods = {
            LogLevel.ERROR: self.logger.error,
            LogLevel.WARN: self.logger.warning,
            LogLevel.DEBUG: self.logger.debug,
        }
        log_method = log_methods.get(level, self.logger.info)
        log_method(formatted_message, extra=extra_data)

    def debug(self, message: str, indent: Optional[int] = None) -> None:
        """Отладочное сообщение (серое)."""
        self.log(message, LogLevel.DEBUG, indent)

    def info(self, message: str, indent: Optional[int] = None) -> None:
        """Информационное сообщение (серое)."""
        self.log(message, LogLevel.INFO, indent)

    def warn(self, message: str, indent: Optional[int] = None) -> None:
        """Предупреждение (серое)."""
        self.log(message, LogLevel.WARN, indent)

    def error(self, message: str, indent: Optional[int] = None) -> None:
        """Ошибка (серое)."""
        self.log(message, LogLevel.ERROR, indent)

    def success(self, message: str, indent: Optional[int] = None) -> None:
        """Успех (серое)."""
        self.log(message, LogLevel.SUCCESS, indent)

    # ========================================================================
    # СТРУКТУРНЫЕ ЭЛЕМЕНТЫ (БЕЗ ПРЕФИКСОВ, НО СЕРЫЕ)
    # ========================================================================

    def _log_formatted_line(self, formatted_line: str, indent: int = 0) -> None:
        """
        Логирует предварительно отформатированную строку БЕЗ префикса.

        Args:
            formatted_line: Отформатированная строка
            indent: Абсолютный уровень отступа
        """
        indent_str = self._get_indent(indent)
        formatted_message = f"{indent_str}{formatted_line}"

        log_extra = {
            **self._log_extra,
            "is_preformatted": True
        }
        self.logger.info(formatted_message, extra=log_extra)

    def _log_framed_text(self,
                         text: str,
                         char: str,
                         width: int,
                         text_indent: int = 0,
                         text_align: str = '<',
                         add_newline: bool = False) -> None:
        """
        Внутренний helper для рисования текста в рамке.

        Args:
            text: Текст для отображения
            char: Символ для рамки
            width: Ширина рамки
            text_indent: Отступ для текста
            text_align: Выравнивание текста ('<', '^', '>')
            add_newline: Добавить пустую строку перед рамкой
        """
        # Верхняя рамка (всегда indent 0)
        nl = "\n" if add_newline else ""
        self._log_formatted_line(f"{nl}{char * width}", indent=0)

        # Текст (с выравниванием и заданным отступом)
        if text_align == '^':
            formatted_text = f"{text:^{width}}"
        else:
            formatted_text = text

        self._log_formatted_line(formatted_text, indent=text_indent)

        # Нижняя рамка (всегда indent 0)
        self._log_formatted_line(f"{char * width}", indent=0)

    def header(self, text: str, char: str = None,
               width: int = DEFAULT_HEADER_WIDTH, indent: Optional[int] = None) -> None:
        """Основной заголовок БЕЗ префикса, но СЕРЫЙ."""
        text_indent = indent if indent is not None else 0
        char = char or self._SYMBOLS['header']

        self._log_framed_text(
            text=text,
            char=char,
            width=width,
            text_indent=text_indent,
            text_align='^',
            add_newline=False
        )

    def subheader(self, text: str, width: int = DEFAULT_TABLE_WIDTH,
                  indent: Optional[int] = None) -> None:
        """Подзаголовок БЕЗ префикса, но СЕРЫЙ."""
        final_indent_level = indent if indent is not None else self.indent_level
        char = self._SYMBOLS['subheader']

        self._log_framed_text(
            text=text,
            char=char,
            width=width,
            text_indent=final_indent_level,
            text_align='<',
            add_newline=True
        )

    def section(self, title: str, phase: Optional[str] = None,
                width: int = DEFAULT_TABLE_WIDTH, indent: Optional[int] = None) -> None:
        """Начало секции БЕЗ префикса, но СЕРЫЙ."""
        indent_level = indent if indent is not None else self.indent_level
        indent_str = self._get_indent(indent_level)

        if phase:
            header = f"[{phase}] {title.upper()}"
        else:
            header = f"{self._SYMBOLS['arrow']} {title}"

        lines = [
            f"\n{indent_str}{header}",
            f"{indent_str}{self._SYMBOLS['subheader'] * width}"
        ]

        for line in lines:
            self._log_formatted_line(line)

    def separator(self, char: str = None, width: int = DEFAULT_TABLE_WIDTH,
                  indent: Optional[int] = None) -> None:
        """Разделитель БЕЗ префикса, но СЕРЫЙ."""
        indent_level = indent if indent is not None else self.indent_level
        indent_str = self._get_indent(indent_level)
        char = char or self._SYMBOLS['subheader']

        self._log_formatted_line(f"{indent_str}{char * width}")

    # ========================================================================
    # ТАБЛИЦЫ И МЕТРИКИ С ПОДДЕРЖКОЙ ВЛОЖЕННОСТИ
    # ========================================================================

    def metrics_table(self, title: str, data: Dict[str, Any], indent: Optional[int] = None) -> None:
        """
        Таблица метрик с поддержкой вложенных структур.

        Args:
            title: Заголовок таблицы
            data: Данные для отображения (поддерживает вложенные словари и объекты)
            indent: Уровень отступа
        """
        if not data:
            return

        # Определяем отступ для заголовка
        final_indent_level = indent if indent is not None else self.indent_level

        # Выводим заголовок с префиксом
        self.debug(f"{title}:", indent=final_indent_level)

        # Вычисляем отступ для контента
        content_indent_level = self._calculate_content_indent(
            final_indent_level)

        # Форматируем данные с поддержкой вложенности
        self._format_nested_metrics(data, content_indent_level)

    def _format_nested_metrics(self, data: Dict[str, Any], current_indent: int) -> None:
        """
        Рекурсивно форматирует вложенные метрики.

        Args:
            data: Данные для форматирования
            current_indent: Текущий уровень отступа
        """
        if not data:
            return

        # Находим максимальную длину ключей на текущем уровне
        max_key_len = max(len(str(k)) for k in data.keys()) if data else 0

        for key, value in data.items():
            if isinstance(value, dict) and value:
                # Для вложенного словаря - выводим ключ и рекурсивно обрабатываем значение
                line = f"{key}:"
                self._log_formatted_line(line, indent=current_indent)
                self._format_nested_metrics(value, current_indent + 1)
            elif hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool)):
                # Для объектов с атрибутами - выводим как вложенную структуру
                line = f"{key}:"
                self._log_formatted_line(line, indent=current_indent)
                obj_dict = {k: v for k, v in value.__dict__.items()
                            if not k.startswith('_')}
                self._format_nested_metrics(obj_dict, current_indent + 1)
            else:
                # Для простых значений - выводим как пару ключ-значение
                formatted_value = self._format_value(value)
                line = f"{key:<{max_key_len}} : {formatted_value}"
                self._log_formatted_line(line, indent=current_indent)

    def metrics_summary(self, title: str,
                        metrics: Dict[str, Dict[str, float]],
                        indent: Optional[int] = None) -> None:
        """Вывод сводки метрик (min, max, mean, std)."""
        indent_level = indent if indent is not None else 0
        self.debug(f"{title}:", indent_level)

        for metric_name, values in metrics.items():
            parts = []
            if 'min' in values:
                parts.append(f"min={values['min']:.3f}")
            if 'max' in values:
                parts.append(f"max={values['max']:.3f}")
            if 'mean' in values:
                parts.append(f"mean={values['mean']:.3f}")
            if 'std' in values:
                parts.append(f"std={values['std']:.3f}")

            content_indent = self._calculate_content_indent(indent_level)
            self._log_formatted_line(
                f"{metric_name}: {', '.join(parts)}",
                indent=content_indent
            )

    def box(self, title: str, content: Dict[str, Any],
            indent: Optional[int] = None, width: int = DEFAULT_TABLE_WIDTH) -> None:
        """Рамка с данными - только рамка с правильным отступом, без заголовка с префиксом."""
        if not content:
            return

        # Определяем базовый отступ
        final_indent_level = indent if indent is not None else self.indent_level

        # Вычисляем отступ для контента
        content_indent_level = self._calculate_content_indent(
            final_indent_level)

        # Создаем рамку
        lines = [
            f"{self._SYMBOLS['box_tl']}{self._SYMBOLS['box_h'] * (width - 2)}{self._SYMBOLS['box_tr']}",
            f"{self._SYMBOLS['box_v']} {title:<{width - 4}} {self._SYMBOLS['box_v']}",
            f"{self._SYMBOLS['box_v']}{self._SYMBOLS['box_h'] * (width - 2)}{self._SYMBOLS['box_v']}"
        ]

        # Форматируем содержимое с выравниванием
        max_key_len = max(len(str(k)) for k in content.keys())

        for key, value in content.items():
            formatted_value = self._format_value(value)
            line = f"{key:<{max_key_len}} : {formatted_value}"

            if len(line) > width - 4:
                line = line[:width - 7] + "..."

            lines.append(
                f"{self._SYMBOLS['box_v']} {line:<{width - 4}} {self._SYMBOLS['box_v']}")

        lines.append(
            f"{self._SYMBOLS['box_bl']}{self._SYMBOLS['box_h'] * (width - 2)}{self._SYMBOLS['box_br']}")

        # Выводим всю рамку с вычисленным отступом
        for line in lines:
            self._log_formatted_line(line, indent=content_indent_level)

    # ========================================================================
    # ТАБЛИЦА СТРАТЕГИЙ (БЕЗ ПРЕФИКСОВ, НО СЕРАЯ)
    # ========================================================================

    def strategy_table_start(self, stage: str, count: int, indent: Optional[int] = None) -> None:
        """Начало таблицы стратегий."""
        indent_level = indent if indent is not None else 0

        # Заголовок с префиксом
        self.info(f"Оценка стратегий для этапа: {stage}", indent_level)
        self.debug(f"Доступно стратегий: {count}", indent_level + 1)

        base_indent = self._get_indent(indent_level + 1)
        content_width = self.table_config['total_width']
        borders = self.table_config['spacing']['borders']

        # Верхняя граница БЕЗ префикса
        if borders:
            line = f"{self._SYMBOLS['box_tl']}{self._SYMBOLS['box_h'] * content_width}{self._SYMBOLS['box_tr']}"
            self._log_formatted_line(line, indent=indent_level + 1)

        # Заголовки колонок БЕЗ префикса
        header_cells = [col['name'] for col in self.table_config['columns']]
        header_row = self._format_table_row(header_cells, borders)
        self._log_formatted_line(header_row, indent=indent_level + 1)

        # Разделитель БЕЗ префикса
        if borders:
            separator = f"{self._SYMBOLS['box_v']}{self._SYMBOLS['box_h'] * content_width}{self._SYMBOLS['box_v']}"
            self._log_formatted_line(separator, indent=indent_level + 1)

    def strategy_table_row(self, name: str, score: float, time_s: float,
                           passed: bool, indent: Optional[int] = None) -> None:
        """Строка таблицы стратегий БЕЗ префикса."""
        indent_level = indent if indent is not None else 0
        status = self._SYMBOLS['success'] if passed else self._SYMBOLS['error']
        borders = self.table_config['spacing']['borders']

        # Форматируем ячейки
        strategy_cell = f"{status} {name:<{self.table_config['columns'][0]['width'] - self.table_config['status_width']}}"
        score_cell = f"{score:>{self.table_config['columns'][1]['width'] - 2}.3f}"
        time_cell = f"{time_s:>{self.table_config['columns'][2]['width'] - 2}.2f}s"

        cells = [strategy_cell, score_cell, time_cell]
        row = self._format_table_row(cells, borders)

        self._log_formatted_line(row, indent=indent_level + 1)

    def strategy_table_end(self, indent: Optional[int] = None) -> None:
        """Завершение таблицы стратегий БЕЗ префикса."""
        indent_level = indent if indent is not None else 0
        content_width = self.table_config['total_width']

        if self.table_config['spacing']['borders']:
            line = f"{self._SYMBOLS['box_bl']}{self._SYMBOLS['box_h'] * content_width}{self._SYMBOLS['box_br']}"
            self._log_formatted_line(line, indent=indent_level + 1)

    def _format_table_row(self, cells: List[str], borders: bool = True) -> str:
        """Форматирует строку таблицы."""
        spacing = self.table_config['spacing']
        columns = self.table_config['columns']

        formatted_cells = []
        for i, cell in enumerate(cells):
            col_config = columns[i]
            formatted_cells.append(
                f"{cell:{col_config['align']}{col_config['width']}}")

        row_content = (" " * spacing['between_columns']).join(formatted_cells)
        padded_row = (" " * spacing['left_padding']) + \
            row_content + (" " * spacing['right_padding'])

        if borders and spacing['borders']:
            return f"{self._SYMBOLS['box_v']}{padded_row}{self._SYMBOLS['box_v']}"
        return padded_row

    def iteration_table(self, title: str, iterations_data: List[Dict[str, Any]],
                        key_columns: List[str] = None, indent: Optional[int] = None) -> None:
        """
        Компактная таблица для отображения результатов итеративного процесса.

        Args:
            title: Заголовок таблицы
            iterations_data: Список словарей с данными итераций
            key_columns: Ключевые колонки для отображения (если None - все колонки)
            indent: Уровень отступа
        """
        if not iterations_data:
            return

        final_indent_level = indent if indent is not None else self.indent_level

        # Выводим заголовок с префиксом
        self.debug(f"{title}:", indent=final_indent_level)

        # Определяем колонки для отображения
        if key_columns is None:
            key_columns = list(iterations_data[0].keys())

        # Вычисляем ширины колонок
        col_widths = {}
        for col in key_columns:
            # Ширина = максимум(название колонки, максимальное значение в колонке)
            header_width = len(str(col))
            max_data_width = max(len(str(iter_data.get(col, "")))
                                 for iter_data in iterations_data)
            col_widths[col] = max(header_width, max_data_width) + 2

        # Заголовок таблицы БЕЗ префикса
        header_parts = []
        for col in key_columns:
            width = col_widths[col]
            header_parts.append(f"{col:^{width}}")
        header_line = " ".join(header_parts)
        self._log_formatted_line(header_line, indent=final_indent_level + 1)

        # Разделитель БЕЗ префикса
        separator_line = " ".join("─" * col_widths[col] for col in key_columns)
        self._log_formatted_line(separator_line, indent=final_indent_level + 1)

        # Данные БЕЗ префикса
        for i, iter_data in enumerate(iterations_data):
            row_parts = []
            for col in key_columns:
                width = col_widths[col]
                value = iter_data.get(col, "")

                # Специальное форматирование для числовых значений
                if isinstance(value, float):
                    formatted_value = f"{value:.3f}"
                else:
                    formatted_value = str(value)

                # Выравнивание: числа - по правому краю, текст - по левому
                align = '>' if isinstance(value, (int, float)) else '<'
                row_parts.append(f"{formatted_value:{align}{width}}")

            row_line = " ".join(row_parts)

            # Добавляем маркер лучшей итерации если есть
            if iter_data.get('is_best', False):
                row_line += "  🚀"

            self._log_formatted_line(row_line, indent=final_indent_level + 1)

    # ========================================================================
    # СПЕЦИАЛИЗИРОВАННЫЕ МЕТОДЫ
    # ========================================================================

    def contour_info(self, contours: List, indent: Optional[int] = None,
                     max_display: int = MAX_LIST_PREVIEW) -> None:
        """Информация о контурах с примерами."""
        indent_level = indent if indent is not None else 0
        self.debug(f"Контуры: {len(contours)} шт", indent_level)

        if self.debug_mode and contours:
            content_indent = self._calculate_content_indent(indent_level)
            self._log_formatted_line(
                f"Примеры контуров (первые {max_display}):",
                indent=content_indent
            )

            for i, contour in enumerate(contours[:max_display]):
                if hasattr(contour, 'shape'):
                    contour_info = f"[{i}] shape={contour.shape}, points={len(contour)}"
                else:
                    contour_info = f"[{i}] points={len(contour)}"
                self._log_formatted_line(
                    contour_info, indent=content_indent + 1)

    def progress(self, current: int, total: int,
                 message: str = "", indent: Optional[int] = None,
                 bar_length: int = 40) -> None:
        """
        Динамический прогресс-бар с перезаписью строки.

        ВАЖНО: Использует только print с ANSI escape codes, не смешивается с logger.
        """
        if total <= 0:
            return

        indent_level = indent if indent is not None else self.indent_level
        percentage = (current / total * 100)
        filled = int(bar_length * current / total)
        bar = (self._SYMBOLS['progress'] * filled +
               self._SYMBOLS['progress_empty'] * (bar_length - filled))

        # Форматируем прогресс-бар
        max_number_width = len(str(total))
        progress_text = f"[{current:>{max_number_width}}/{total}]   {percentage:>5.1f}% {bar}"

        if message:
            status = f"{progress_text} {message}"
        else:
            status = progress_text

        # Добавляем отступ
        indent_str = self._get_indent(indent_level)
        full_status = f"{indent_str}{status}"

        # ДИНАМИЧЕСКАЯ ПЕРЕЗАПИСЬ СТРОКИ
        if current == 0:
            # Первый вывод
            print(full_status, end='', flush=True)
        elif current < total:
            # Перезаписываем строку: \r возврат каретки, \033[K очистка до конца строки
            print(f"\r\033[K{full_status}", end='', flush=True)
        else:
            # Финальный вывод с переводом строки
            print(f"\r\033[K{full_status}", flush=True)

    # ========================================================================
    # КОНТЕКСТНЫЕ МЕНЕДЖЕРЫ
    # ========================================================================

    @contextmanager
    def indent(self, levels: int = 1):
        """Контекстный менеджер для временного увеличения отступа."""
        self.indent_level += levels
        try:
            yield
        finally:
            self.indent_level -= levels

    @contextmanager
    def timed_section(self, title: str, level: LogLevel = LogLevel.INFO):
        """Контекстный менеджер для измерения времени."""
        start = time.time()
        self.log(f"{title}...", level)

        with self.indent():
            yield

        elapsed = time.time() - start
        self.success(f"{title} завершено за {elapsed:.2f}с")

    # ========================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ========================================================================

    def _format_value(self, value: Any) -> str:
        """Форматирование значения для вывода."""
        if value is None:
            return "None"
        elif isinstance(value, float):
            return f"{value:.6g}"
        elif isinstance(value, int):
            return str(value)
        elif isinstance(value, str):
            if len(value) > self.MAX_STRING_LENGTH:
                return value[:self.MAX_STRING_LENGTH - 3] + "..."
            return value
        elif isinstance(value, (list, tuple)):
            if len(value) > self.MAX_LIST_PREVIEW:
                preview = ', '.join(str(x)
                                    for x in value[:self.MAX_LIST_PREVIEW])
                return f"{type(value).__name__}({preview}...), len={len(value)}"
            return f"{type(value).__name__}({', '.join(str(x) for x in value)})"
        else:
            str_value = str(value)
            if len(str_value) > self.MAX_STRING_LENGTH:
                return str_value[:self.MAX_STRING_LENGTH - 3] + "..."
            return str_value
