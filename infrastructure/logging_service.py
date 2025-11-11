# infrastructure/logging_service.py
"""
Централизованный сервис логирования с разделением вывода для UI и файла.

ОТВЕТСТВЕННОСТЬ:
- Базовая инфраструктура логирования (настройка, форматеры, обработчики)
- Разделение вывода: консоль (чистый UI) и файл (детальный лог)
- Получение модульных логгеров для использования в проекте
- Декораторы для автоматического логирования исключений

РАЗДЕЛЕНИЕ ОТВЕТСТВЕННОСТИ:
- LoggingService: базовая инфраструктура (конфигурация, обработчики, форматеры)
- UIService: использует LoggingService для рабочего вывода оператору
- DebugFormatter: использует LoggingService для отладочного вывода разработчику
"""

import logging
import sys
import traceback
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

# Отключаем логирование PIL перед любыми импортами
logging.getLogger('PIL').setLevel(logging.WARNING)


# --- Форматтеры ---

class _SourceAwareFormatter(logging.Formatter):
    """
    Детальный форматтер для лог-файла.

    Добавляет в запись информацию об источнике вызова:
    'source': {модуль}.{класс}.{функция}
    'file': {имя_файла}
    """

    def format(self, record: logging.LogRecord) -> str:
        """Форматирует запись лога, обогащая её контекстом."""
        module_name = record.module
        file_name = Path(record.pathname).name
        func_name = record.funcName
        class_name = getattr(record, "className", None)

        if class_name:
            source = f"{module_name}.{class_name}.{func_name}"
        else:
            source = f"{module_name}.{func_name}"

        record.source = source
        record.file = file_name

        return super().format(record)


class _UIFormatter(logging.Formatter):
    """
    Минималистичный форматтер для консольного вывода (UI).

    Логика форматирования:
    1. Для предформатированных сообщений (is_preformatted=True):
       - Применяется только цвет (серый для debug, белый для остальных)
       - Текст выводится как есть (уже содержит префиксы и отступы)

    2. Для обычных сообщений:
       - Применяется стандартное форматирование
       - Префикс находится в колонке 0 (без отступов)
       - Отступы применяются к тексту сообщения после префикса
    """

    # Коды цветов ANSI
    COLORS = {
        'white': '\033[97m',
        'gray': '\033[90m',
        'reset': '\033[0m'
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Форматирует запись лога с учетом типа сообщения и отступов.
        """
        # Определяем цвет на основе источника
        is_debug_formatter = getattr(record, 'is_debug_formatter', False)
        color = self.COLORS['gray'] if is_debug_formatter else self.COLORS['white']

        # Получаем отформатированное сообщение
        message = record.getMessage()

        # Если сообщение уже предформатировано - просто добавляем цвет
        if getattr(record, 'is_preformatted', False):
            return f"{color}{message}{self.COLORS['reset']}"

        # Для обычных сообщений применяем стандартное форматирование
        # Сообщение уже содержит префикс и отступы от UIService/DebugFormatter
        return f"{color}{message}{self.COLORS['reset']}"


# --- Основной сервис ---

class LoggingService:
    """Фасад для настройки и получения логгеров проекта."""

    _configured: bool = False
    _level_map = {
        "none": logging.CRITICAL + 10,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }

    # Форматы для разных обработчиков
    _UI_FORMAT = "%(message)s"
    _FILE_FORMAT = (
        "%(asctime)s | %(levelname)-7s | %(source)s | file=%(file)s | %(message)s"
    )

    @classmethod
    def configure(
        cls,
        *,
        log_level: str = "info",
        debug: bool = False,
        log_file: Optional[Path] = None,
        datetime_format: str = "%Y-%m-%d %H:%M:%S",
    ) -> None:
        """
        Настраивает корневой логгер для всего приложения.

        Создает два обработчика:
        1. Для консоли (stdout) с минималистичным форматом.
        2. Для файла (если указан) с подробным форматом.

        Args:
            log_level: Минимальный уровень лога для вывода в консоль ('info', 'debug', etc.).
            debug: Если True, принудительно устанавливает уровень консоли в DEBUG.
            log_file: Путь к файлу для детального логирования.
            datetime_format: Формат даты и времени для файлового лога.
        """
        if cls._configured:
            logging.warning("LoggingService.configure() вызван повторно.")
            return

        console_level = cls._level_map.get(log_level.lower(), logging.INFO)
        if debug:
            console_level = logging.DEBUG

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # 1. Консольный обработчик для чистого UI
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(console_level)
        console_formatter = _UIFormatter(cls._UI_FORMAT)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # 2. Файловый обработчик для детальных логов (опционально)
        if log_file:
            try:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_file, encoding="utf-8")
                file_handler.setLevel(logging.DEBUG)
                file_formatter = _SourceAwareFormatter(
                    cls._FILE_FORMAT, datefmt=datetime_format
                )
                file_handler.setFormatter(file_formatter)
                root_logger.addHandler(file_handler)
            except (IOError, OSError) as e:
                logging.critical(f"Не удалось создать файловый логгер: {e}")

        cls._configured = True

    @staticmethod
    def get_logger(name: Optional[str] = None) -> logging.Logger:
        """
        Возвращает экземпляр логгера с указанным именем.

        Args:
            name: Имя логгера. Обычно `__name__` вызывающего модуля.

        Returns:
            Экземпляр logging.Logger.
        """
        return logging.getLogger(name or __name__)

    @staticmethod
    def log_exceptions(logger: logging.Logger) -> Callable:
        """
        Декоратор для логирования необработанных исключений в функции.

        Args:
            logger: Экземпляр логгера для записи исключения.

        Returns:
            Декоратор.
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    class_name: Optional[str] = None
                    if args:
                        instance = args[0]
                        if hasattr(instance, "__class__"):
                            class_name = instance.__class__.__name__

                    extra = {"className": class_name} if class_name else {}
                    tb = "".join(
                        traceback.format_exception(
                            type(exc), exc, exc.__traceback__)
                    )
                    logger.critical(
                        f"Unhandled exception in {func.__name__}: {exc}\n{tb}",
                        extra=extra,
                    )
                    raise

            return wrapper

        return decorator
