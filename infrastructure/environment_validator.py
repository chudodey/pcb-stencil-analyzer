# infrastructure/environment_validator.py
import sys
import importlib.util
import os
from typing import List, Dict


class EnvironmentValidator:
    """Валидатор окружения Python"""

    def __init__(self, min_python_version: str):
        self.min_python_version = min_python_version
        self.required_modules = self._read_requirements_file()

        # Маппинг имен pip пакетов на имена Python модулей
        self.package_to_module = {
            'opencv_python': 'cv2',
            'opencv-python': 'cv2',
            'Pillow': 'PIL',
            'PyYAML': 'yaml',
            'scikit-learn': 'sklearn',
            'beautifulsoup4': 'bs4',
            'lxml': 'lxml',
            'screeninfo': 'screeninfo',
            'matplotlib': 'matplotlib',
            'numpy': 'numpy',
            'scipy': 'scipy'
        }

    def _read_requirements_file(self) -> List[str]:
        """Читает файл requirements.txt и возвращает список пакетов"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            requirements_file = os.path.join(
                current_dir, "../requirements.txt")

            if not os.path.exists(requirements_file):
                return []

            with open(requirements_file, 'r', encoding='utf-8') as f:
                packages = []
                for line in f:
                    line = line.strip()
                    # Пропускаем пустые строки и комментарии
                    if not line or line.startswith('#') or line.startswith('-'):
                        continue

                    # Извлекаем имя пакета (убираем версии и дополнительные флаги)
                    package_name = line.split('>=')[0].split('==')[
                        0].split('[')[0].strip()
                    if package_name:
                        packages.append(package_name)

                return packages

        except Exception:
            return []

    def _is_module_available(self, package_name: str) -> bool:
        """Проверяет, доступен ли модуль для импорта"""
        try:
            # Получаем имя модуля из маппинга
            module_name = self.package_to_module.get(
                package_name, package_name)

            # Пробуем импортировать модуль
            return importlib.util.find_spec(module_name) is not None

        except Exception:
            return False

    def check_python_environment(self) -> List[str]:
        """Проверяет версию Python и наличие необходимых модулей"""
        issues = []

        # Проверка версии Python
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        required_version_tuple = tuple(
            map(int, self.min_python_version.split('.')))
        current_version_tuple = (
            sys.version_info.major, sys.version_info.minor, sys.version_info.micro)

        if current_version_tuple < required_version_tuple:
            issues.append(
                f"Требуется Python {self.min_python_version}+, установлен {current_version}")

        # Проверка модулей
        missing_modules = []

        for package in self.required_modules:
            if not self._is_module_available(package):
                missing_modules.append(package)

        if missing_modules:
            issues.append(
                f"Отсутствуют модули: {', '.join(sorted(missing_modules))}")
            issues.append("Установите: pip install -r requirements.txt")

        return issues

    def get_environment_info(self) -> dict:
        """Возвращает информацию о текущем окружении"""
        requirements_file = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "../requirements.txt")

        return {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "required_python_version": self.min_python_version,
            "required_modules_count": len(self.required_modules),
            "required_modules": self.required_modules,
            "requirements_file_exists": os.path.exists(requirements_file),
            "requirements_file_path": requirements_file
        }
