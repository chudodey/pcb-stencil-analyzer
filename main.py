#!/usr/bin/env python3
"""
Модуль для автоматического сравнения Gerber-файлов с отсканированными изображениями трафаретов.
"""

import argparse
import logging
import sys
from pathlib import Path
import traceback

from app import StencilAnalyzerApplication
from infrastructure import (ConfigService, EnvironmentValidator, FileManager,
                            LoggingService, UIService, DebugFormatter)

# Добавляем путь к модулям в PYTHONPATH
sys.path.append(str(Path(__file__).parent))


def safe_setup_logging(config_service, ui_service=None, debug_formatter=None) -> bool:
    """
    Безопасная настройка логирования с обработкой ошибок.
    """
    try:
        log_file = config_service.output_folder / 'application.log'

        # ПРОСТОЙ ФИКС: Всегда используем строковый уровень
        if config_service.debug_mode:
            log_level_str = "debug"
        else:
            log_level_str = "info"  # Или другое значение по умолчанию

        if ui_service:
            ui_service.show_message(
                f"Настраиваем логирование: level='{log_level_str}'")
        elif debug_formatter:
            debug_formatter.info(
                f"Настраиваем логирование: level='{log_level_str}'")
        else:
            # Фоллбэк до инициализации UI сервиса
            logger = logging.getLogger(__name__)
            logger.info(f"Настраиваем логирование: level='{log_level_str}'")

        LoggingService.configure(
            log_level=log_level_str,  # Всегда строка
            debug=config_service.debug_mode,
            log_file=log_file,
            datetime_format=getattr(
                config_service, 'datetime_format', '%Y-%m-%d %H:%M:%S'),
        )

        return True

    except Exception as e:
        error_msg = f"Ошибка настройки логирования: {e}"
        if ui_service:
            ui_service.show_error(error_msg)
        elif debug_formatter:
            debug_formatter.error(error_msg)
        else:
            logger = logging.getLogger(__name__)
            logger.error(f"{error_msg}")
            logging.basicConfig(level=logging.INFO)
        return False


def parse_command_line_arguments() -> argparse.Namespace:
    """
    Обработка аргументов командной строки
    """
    parser = argparse.ArgumentParser(
        description='Анализатор трафаретов - автоматическое сравнение Gerber с реальными сканами',
        epilog='Пример использования: python main.py --debug'
    )

    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Режим отладки с подробным выводом'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.ini',
        help='Путь к файлу конфигурации (по умолчанию: config.ini)'
    )

    parser.add_argument(
        '--version', '-v',
        action='version',
        version='Stencil Analyzer v2.1.0'
    )

    return parser.parse_args()


def initialize_services(args: argparse.Namespace) -> tuple:
    """
    Инициализация всех сервисов приложения с улучшенной обработкой ошибок.
    """
    config_service, ui_service, file_manager, debug_formatter = None, None, None, None

    try:
        # Для вывода до инициализации UI используем временный logger
        logger = logging.getLogger(__name__)

        logger.info("Инициализация конфигурации...")
        config_service = ConfigService(
            config_file=args.config,
            debug_mode=args.debug
        )

        logger.info("Инициализация сервисов инфраструктуры...")

        min_python_version = getattr(
            config_service, 'min_python_version', (3, 8))
        if isinstance(min_python_version, tuple):
            min_python_version = '.'.join(map(str, min_python_version))

        # ПРОСТАЯ ПРОВЕРКА ОКРУЖЕНИЯ ДО СОЗДАНИЯ СЛОЖНЫХ СЕРВИСОВ
        environment_validator = EnvironmentValidator(
            min_python_version=min_python_version
        )
        env_issues = environment_validator.check_python_environment()
        if env_issues:
            print("❌ Обнаружены проблемы с окружением:")
            for issue in env_issues:
                print(f"   {issue}")
            print("\nУстраните проблемы и перезапустите приложение.")
            return None, None, None, None

        # Создаем FileManager
        file_manager = FileManager(config_service)

        # Инициализация UI сервиса
        ui_service = UIService(config_service)

        # Инициализация DebugFormatter для отладочного вывода
        debug_formatter = DebugFormatter(config_service.debug_mode, __name__)

        # Создание рабочих директорий через FileManager
        debug_formatter.info("Создание рабочих директорий...")
        required_dirs = [
            getattr(config_service, 'gerber_folder', Path('gerber')),
            getattr(config_service, 'scan_folder', Path('scans')),
            getattr(config_service, 'output_folder', Path('output'))
        ]

        for directory in required_dirs:
            try:
                file_manager.dirs.ensure_dir(directory, as_dir=True)
                ui_service.show_success(f"{directory}")
            except OSError as e:
                error_msg = f"Не удалось создать директорию {directory}: {e}"
                ui_service.show_error(error_msg)
                return None, None, None, None

        # Настройка логирования
        debug_formatter.info("Настройка логирования...")
        if not safe_setup_logging(config_service, ui_service, debug_formatter):
            ui_service.show_warning("Логирование настроено в базовом режиме")

        # Логируем запуск приложения
        ui_service.show_header()
        debug_formatter.info(f"Файл конфигурации: {args.config}")

        ui_service.show_success("Все сервисы успешно инициализированы")
        return config_service, ui_service, file_manager, debug_formatter

    except FileNotFoundError as e:
        error_msg = f"Файл конфигурации не найден: {e}"
        if ui_service:
            ui_service.show_error(error_msg)
        elif debug_formatter:
            debug_formatter.error(error_msg)
        else:
            logger = logging.getLogger(__name__)
            logger.error(f"{error_msg}")
        return None, None, None, None
    except Exception as e:
        error_msg = f"Критическая ошибка инициализации: {e}"
        if ui_service:
            ui_service.show_error(error_msg)
        elif debug_formatter:
            debug_formatter.error(error_msg)
        else:
            logger = logging.getLogger(__name__)
            logger.error(f"{error_msg}")
        return None, None, None, None


def main() -> int:
    """
    Главная функция приложения.
    """
    # Используем временный logger для вывода до инициализации
    logger = logging.getLogger(__name__)
    logger.info("Анализатор трафаретов - запуск...")

    # Парсинг аргументов командной строки
    args = parse_command_line_arguments()

    app_instance = None

    try:
        # Инициализация основных сервисов
        config_service, ui_service, file_manager, debug_formatter = initialize_services(
            args)

        if not config_service or not ui_service or not file_manager:
            debug_formatter.error(
                "Не удалось инициализировать сервисы. Программа завершена.")
            return 1

        # Создание главного приложения
        debug_formatter.info("Создание главного приложения...")
        app_instance = StencilAnalyzerApplication(
            config_service=config_service,
            ui_service=ui_service,
            file_manager=file_manager
        )

        # Инициализация приложения
        debug_formatter.info("Инициализация приложения...")
        if not app_instance.initialize():
            return 1

        ui_service.show_success("Все системы запущены. Начало работы...")

        # Запуск основного цикла приложения
        app_instance.run()

        # Нормальное завершение
        ui_service.show_success("Программа успешно завершена")
        return 0

    except KeyboardInterrupt:
        ui_service.show_warning(
            "Программа прервана пользователем", add_newline=True)
        return 0
    except Exception as e:
        error_msg = f"Критическая ошибка: {e}"
        debug_formatter.error(error_msg)
        if app_instance and hasattr(app_instance, 'ui_service'):
            app_instance.ui_service.show_error(error_msg)
        traceback.print_exc()
        return 1
    finally:
        # Корректное завершение работы
        if app_instance is not None:
            try:
                app_instance.shutdown()
            except Exception as e:
                ui_service.show_error(f"Ошибка при завершении приложения: {e}")


if __name__ == "__main__":
    EXIT_CODE = main()
    sys.exit(EXIT_CODE)
