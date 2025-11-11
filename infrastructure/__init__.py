# infrastructure/__init__.py
"""
Инфраструктурный слой - технические сервисы

АРХИТЕКТУРА ВЫВОДА ИНФОРМАЦИИ:
- LoggingService: Базовая инфраструктура логирования (конфигурация, обработчики, форматеры)
  - Настройка корневого логгера с разделением вывода: консоль (чистый UI) и файл (детальный лог)
  - Получение модульных логгеров для использования в проекте
  
- UIService: Рабочий вывод для оператора (информация, успехи, ошибки, предупреждения)
  - Единообразное форматирование всех сообщений для пользователя
  - Профессиональный визуальный стиль
  - НЕ используется для отладочного вывода
  
- DebugFormatter: Отладочный вывод для разработчика (детали, метрики, диагностика)
  - Структурированное форматирование с отступами и символами
  - Вывод только при включенном debug_mode
  - Использует LoggingService для вывода

РАЗДЕЛЕНИЕ ОТВЕТСТВЕННОСТИ:
- Для рабочего вывода оператору → используйте UIService
- Для отладочного вывода → используйте DebugFormatter
- Все выводы идут через LoggingService (консоль + файл)

ДРУГИЕ СЕРВИСЫ:
- ConfigService: Конфигурация приложения
- FileManager: Работа с файловой системой
- EnvironmentValidator: Проверка окружения
"""

from .config_service import ConfigService
from .debug_formatter import DebugFormatter
from .environment_validator import EnvironmentValidator
from .file_manager import FileManager
from .logging_service import LoggingService
from .ui_service import UIService

__all__ = [
    'ConfigService',
    'DebugFormatter',
    'EnvironmentValidator',
    'FileManager',
    'LoggingService',
    'UIService',
]