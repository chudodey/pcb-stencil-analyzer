# processing/scan_service.py
"""
Сервис для инкапсуляции логики загрузки сканов.
"""
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from domain.data_models import ScanImage
from infrastructure import FileManager
from infrastructure.debug_formatter import DebugFormatter
from infrastructure.config_service import ConfigService


class ScanService:
    """Инкапсулирует загрузку и подготовку метаданных скана."""

    def __init__(self, file_manager: FileManager, config_service: ConfigService):
        self.file_manager = file_manager
        self.config = config_service
        self.formatter = DebugFormatter(config_service.debug_mode, __name__)

    def load_scan(self, scan_path: Path, order_number: str) -> Tuple[ScanImage, np.ndarray]:
        """
        Подготавливает данные скана: метаинформацию и изображение.
        (Объединяет _prepare_scan_data и _load_scan_image из ProcessingEngine)
        """
        self.formatter.debug(f"Загрузка скана: {scan_path.name}")

        # 1. Метаинформация (из _prepare_scan_info)
        dpi = self.file_manager.get_image_dpi(scan_path)
        image_size = self.file_manager.get_image_size(scan_path)

        scan_info = ScanImage(
            order_number=order_number,
            filename=scan_path.name,
            scan_path=scan_path,
            dpi=dpi,
            image_size_px=image_size
        )

        # 2. Изображение (из _load_scan_image)
        file_bytes = np.fromfile(str(scan_path), dtype=np.uint8)
        scan_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if scan_image is None:
            raise RuntimeError(
                f"Не удалось прочитать изображение: {scan_path}")

        self.formatter.debug(
            f"Загружен скан: {scan_info.filename}, DPI: {scan_info.dpi}, Размер: {scan_info.image_size_px}"
        )

        return scan_info, scan_image
