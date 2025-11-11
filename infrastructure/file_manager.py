# infrastructure/file_manager.py
"""
Универсальный менеджер файловой системы с внутренним разделением ответственности.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# pylint: disable=no-member
import cv2
import numpy as np
from PIL import Image

from infrastructure.config_service import ConfigService


@dataclass
class EnsureDirResult:
    """Результат проверки/создания директории"""
    created: bool
    dir_path: Path


class FileManager:
    """
    Универсальный менеджер файловой системы.

    Внутренняя структура:
    - DirectoryOperations: операции с директориями
    - FileOperations: операции с файлами  
    - SearchOperations: поисковые операции
    - ConfigOperations: операции, зависящие от конфигурации
    """

    class DirectoryOperations:
        """Операции с директориями (статические)"""

        @staticmethod
        def ensure_dir(path: Union[Path, str], as_dir: Optional[bool] = None) -> EnsureDirResult:
            """Гарантирует существование директории, связанной с указанным путём."""
            p = Path(path)
            dir_path = p if (as_dir is True or (
                as_dir is None and p.suffix == '')) else p.parent

            created = not dir_path.exists()
            if created:
                dir_path.mkdir(parents=True, exist_ok=True)
            return EnsureDirResult(created=created, dir_path=dir_path)

        @staticmethod
        def create_timestamped_workspace(base_path: Path, prefix: str = "") -> Path:
            """Создает директорию с временной меткой"""

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dir_name = f"{prefix}_{timestamp}" if prefix else timestamp
            workspace = base_path / dir_name
            FileManager.DirectoryOperations.ensure_dir(workspace, as_dir=True)
            return workspace

        @staticmethod
        def get_directory_size(path: Path) -> int:
            """Возвращает размер директории в байтах"""
            return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())

        @staticmethod
        def list_files(path: Path, pattern: str = "*") -> List[Path]:
            """Возвращает список файлов в директории"""
            if not path.exists():
                return []
            return list(path.glob(pattern))

    class FileOperations:
        """Операции с файлами (статические)"""

        @staticmethod
        def read_text(path: Union[Path, str], encoding: str = 'utf-8') -> str:
            """Читает текстовые данные из файла."""
            FileManager.DirectoryOperations.ensure_dir(path, as_dir=False)
            return Path(path).read_text(encoding=encoding)

        @staticmethod
        def write_text(path: Union[Path, str], content: str, encoding: str = 'utf-8') -> None:
            """Записывает текстовые данные в файл, создавая директорию при необходимости."""
            FileManager.DirectoryOperations.ensure_dir(path, as_dir=False)
            Path(path).write_text(content, encoding=encoding)

        @staticmethod
        def read_image(path: Path) -> np.ndarray:
            """Читает изображение в numpy array"""
            file_bytes = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"Не удалось прочитать изображение: {path}")
            return img

        @staticmethod
        def write_image(path: Union[Path, str], img: np.ndarray, quality: int = 95) -> None:
            """Сохраняет изображение с оптимизацией качества."""
            FileManager.DirectoryOperations.ensure_dir(path, as_dir=False)
            ext = Path(path).suffix.lower()
            params = []
            if ext in ['.jpg', '.jpeg']:
                params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif ext == '.png':
                params = [cv2.IMWRITE_PNG_COMPRESSION, 6]

            success, encoded_image = cv2.imencode(ext, img, params)
            if not success:
                raise IOError(f"Ошибка кодирования изображения: {path}")

            Path(path).write_bytes(encoded_image)

        @staticmethod
        def get_file_size(path: Path) -> int:
            """Возвращает размер файла в байтах"""
            return path.stat().st_size if path.exists() else 0

        @staticmethod
        def format_file_size(size_bytes: int) -> str:
            """Форматирует размер файла в читаемом виде (Б, КБ, МБ, ГБ)."""
            if size_bytes == 0:
                return "0 Б"

            size_names = ["Б", "КБ", "МБ", "ГБ"]
            i = 0
            size = float(size_bytes)

            while size >= 1024 and i < len(size_names) - 1:
                size /= 1024
                i += 1

            return f"{size:.1f} {size_names[i]}"

    class SearchOperations:
        """Поисковые операции (зависят от конфигурации)"""

        def __init__(self, config_service: 'ConfigService'):
            self.config = config_service

        def find_gerber_files(self, order_number: str) -> List[Path]:
            """Ищет Gerber-файлы с указанным номером заказа."""
            search_dir = self.config.gerber_folder
            rule = self.config.multiple_files_rule

            if not search_dir.exists():
                return []

            pattern = f"*{order_number}*.gbr"
            found_files = list(search_dir.glob(pattern))

            if not found_files:
                return []

            if rule == 'alphabetic_first':
                found_files.sort(key=lambda x: x.name)
            elif rule == 'newest':
                found_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            elif rule == 'oldest':
                found_files.sort(key=lambda x: x.stat().st_mtime)

            return found_files

        def wait_for_scan(self, callback: Optional[Callable[[float, float], None]] = None) -> Optional[Path]:
            """Ожидает появление нового файла скана."""
            start_time = time.time()
            existing_files = self._get_existing_files()

            while True:
                current_time = time.time()
                elapsed = current_time - start_time

                if callback:
                    callback(self.config.scan_wait_timeout, elapsed)

                if self.config.scan_wait_timeout > 0 and elapsed > self.config.scan_wait_timeout:
                    return None

                new_files = self._find_new_files(existing_files)
                if new_files:
                    return self.get_newest_file(new_files)

                time.sleep(self.config.file_check_interval)

        def get_last_existing_scan(self) -> Optional[Path]:
            """Возвращает последний существующий файл скана."""
            scan_folder = self.config.scan_folder
            supported_formats = self.config.supported_image_formats

            all_files = []
            for ext in supported_formats:
                all_files.extend(scan_folder.glob(f"*{ext}"))

            if not all_files:
                return None

            files = [f for f in all_files if f.is_file()]
            if not files:
                return None

            return max(files, key=lambda x: x.stat().st_mtime)

        def _get_existing_files(self) -> Set[Path]:
            """Возвращает множество существующих файлов в папке сканирования."""
            scan_folder = self.config.scan_folder
            supported_formats = self.config.supported_image_formats
            return {f for ext in supported_formats for f in scan_folder.glob(f"*{ext}") if f.is_file()}

        def _find_new_files(self, existing_files: Set[Path]) -> List[Path]:
            """Находит новые, полностью записанные файлы."""
            scan_folder = self.config.scan_folder
            supported_formats = self.config.supported_image_formats
            new_files = []
            for ext in supported_formats:
                for file_path in scan_folder.glob(f"*{ext}"):
                    if file_path.is_file() and file_path not in existing_files:
                        if self._is_file_fully_written(file_path):
                            new_files.append(file_path)
            return new_files

        @staticmethod
        def get_newest_file(files: List[Path]) -> Path:
            """Возвращает самый новый файл по дате изменения."""
            return max(files, key=lambda f: f.stat().st_mtime)

        @staticmethod
        def _is_file_fully_written(file_path: Path) -> bool:
            """Проверяет, что файл полностью записан, сравнивая размер."""
            try:
                initial_size = file_path.stat().st_size
                time.sleep(0.1)
                return initial_size == file_path.stat().st_size and initial_size > 0
            except (OSError, FileNotFoundError):
                return False

    class ConfigOperations:
        """Операции, зависящие от конфигурации"""

        def __init__(self, config_service: 'ConfigService'):
            self.config = config_service

        def create_order_workspace(self, order_number: str) -> Path:
            """Создаёт рабочую директорию для заказа согласно настройкам конфигурации."""
            if self.config.create_order_subfolder:
                workspace = self.config.output_folder / order_number
            else:
                workspace = self.config.output_folder

            result = FileManager.DirectoryOperations.ensure_dir(
                workspace, as_dir=True)
            return result.dir_path

        def get_output_filename(self, template: str, order_number: str, workspace: Path) -> Path:
            """Генерирует имя выходного файла на основе шаблона и обрабатывает конфликты имён."""
            filename = template.format(order_number=order_number)
            filepath = workspace / filename

            if self.config.existing_files_action == 'increment' and filepath.exists():
                counter = 1
                name, ext = filepath.stem, filepath.suffix

                while True:
                    new_name = f"{name}_{counter}{ext}"
                    new_filepath = workspace / new_name
                    if not new_filepath.exists():
                        return new_filepath
                    counter += 1

            return filepath

        def validate_image_file(self, file_path: Path) -> Tuple[bool, str]:
            """Валидирует файл изображения, используя max_size из конфига."""
            max_size_mb = self.config.max_scan_file_size
            if not file_path.exists():
                return False, "Файл не существует"

            file_size_mb = FileManager.FileOperations.get_file_size(
                file_path) / (1024 * 1024)
            if max_size_mb > 0 and file_size_mb > max_size_mb:
                return False, f"Размер файла ({file_size_mb:.1f} МБ) превышает лимит ({max_size_mb} МБ)"

            try:
                with Image.open(file_path) as img:
                    img.verify()
                return True, "OK"
            except Exception as e:
                return False, f"Некорректный файл изображения: {str(e)}"

        def get_image_dpi(self, file_path: Path) -> int:
            """Получение DPI изображения с учетом настроек приоритета из конфига."""
            try:
                with Image.open(file_path) as img:
                    dpi_info = img.info.get('dpi')
                    if self.config.dpi_priority == 'metadata' and dpi_info:
                        return int(dpi_info[0] if isinstance(dpi_info, tuple) else dpi_info)
            except Exception:
                pass
            return self.config.default_dpi

        def get_image_info(self, file_path: Path) -> Dict[str, Any]:
            """Получает полную информацию об изображении."""
            try:
                dpi = self.get_image_dpi(file_path)
                size_pixels = self.get_image_size(file_path)
                size_mm = self.pixels_to_mm(size_pixels, dpi)
                # size_mm = FileManager.FileOperations.pixels_to_mm(size_pixels, dpi)
                file_size = FileManager.FileOperations.get_file_size(file_path)

                return {
                    'dpi': dpi,
                    'size_pixels': size_pixels,
                    'size_mm': size_mm,
                    'file_size_bytes': file_size,
                    'format': file_path.suffix.lstrip('.')
                }
            except Exception as e:
                return {
                    'dpi': 0,
                    'size_pixels': (0, 0),
                    'size_mm': (0.0, 0.0),
                    'file_size_bytes': 0,
                    'format': 'unknown',
                    'error': str(e)
                }

        def get_image_size(self, file_path: Path) -> Tuple[int, int]:
            """Получение размеров изображения в пикселях."""
            try:
                file_bytes = np.fromfile(file_path, dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
                if img is None:
                    return 0, 0
                return img.shape[1], img.shape[0]
            except Exception:
                return 0, 0

        def pixels_to_mm(self, size_pixels: Tuple[int, int], dpi: int) -> Tuple[float, float]:
            """Перевод размеров из пикселей в миллиметры."""
            if dpi <= 0:
                return 0.0, 0.0
            mm_per_pixel = 25.4 / dpi
            return size_pixels[0] * mm_per_pixel, size_pixels[1] * mm_per_pixel

        def mm2_to_pixels(self, area_mm2: float, dpi: int) -> float:
            """Преобразует площадь из мм² в пиксели, гарантируя минимальный размер."""
            if dpi <= 0:
                return max(area_mm2, 10.0)
            pixels_per_mm_sq = (dpi / 25.4) ** 2
            return max(area_mm2 * pixels_per_mm_sq, 10.0)

    # Основной интерфейс FileManager
    def __init__(self, config_service: 'ConfigService'):
        self.config = config_service
        self.dirs = self.DirectoryOperations()
        self.files = self.FileOperations()
        self.search = self.SearchOperations(config_service)
        self.config_ops = self.ConfigOperations(config_service)

    # Делегирующие методы для обратной совместимости
    def ensure_dir(self, path: Union[Path, str], as_dir: Optional[bool] = None) -> EnsureDirResult:
        return self.dirs.ensure_dir(path, as_dir)

    def create_order_workspace(self, order_number: str) -> Path:
        return self.config_ops.create_order_workspace(order_number)

    def get_output_filename(self, template: str, order_number: str, workspace: Path) -> Path:
        return self.config_ops.get_output_filename(template, order_number, workspace)

    def find_gerber_files(self, order_number: str) -> List[Path]:
        return self.search.find_gerber_files(order_number)

    def wait_for_scan(self, callback: Optional[Callable[[float, float], None]] = None) -> Optional[Path]:
        return self.search.wait_for_scan(callback)

    def get_last_existing_scan(self) -> Optional[Path]:
        return self.search.get_last_existing_scan()

    def validate_image_file(self, file_path: Path) -> Tuple[bool, str]:
        return self.config_ops.validate_image_file(file_path)

    def get_image_dpi(self, file_path: Path) -> int:
        return self.config_ops.get_image_dpi(file_path)

    def get_image_info(self, file_path: Path) -> Dict[str, Any]:
        return self.config_ops.get_image_info(file_path)

    # Статические методы для удобства
    @staticmethod
    def read_text(path: Union[Path, str], encoding: str = 'utf-8') -> str:
        return FileManager.FileOperations.read_text(path, encoding)

    @staticmethod
    def write_text(path: Union[Path, str], content: str, encoding: str = 'utf-8') -> None:
        FileManager.FileOperations.write_text(path, content, encoding)

    @staticmethod
    def write_image(path: Union[Path, str], img: np.ndarray, quality: int = 95) -> None:
        FileManager.FileOperations.write_image(path, img, quality)

    @staticmethod
    def get_image_size(file_path: Path) -> Tuple[int, int]:
        # Для обратной совместимости, но лучше использовать config_ops.get_image_size
        try:
            file_bytes = np.fromfile(file_path, dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            if img is None:
                return 0, 0
            return img.shape[1], img.shape[0]
        except Exception:
            return 0, 0

    @staticmethod
    def pixels_to_mm(size_pixels: Tuple[int, int], dpi: int) -> Tuple[float, float]:
        if dpi <= 0:
            return 0.0, 0.0
        mm_per_pixel = 25.4 / dpi
        return size_pixels[0] * mm_per_pixel, size_pixels[1] * mm_per_pixel

    @staticmethod
    def mm2_to_pixels(area_mm2: float, dpi: int) -> float:
        if dpi <= 0:
            return max(area_mm2, 10.0)
        pixels_per_mm_sq = (dpi / 25.4) ** 2
        return max(area_mm2 * pixels_per_mm_sq, 10.0)

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        return FileManager.FileOperations.format_file_size(size_bytes)
