# processing/artifact_saver.py
"""
Сервис сохранения артефактов для обратной совместимости.
Сохраняет только базовые изображения: гербер, исходный скан и финальный результат.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from domain.data_models import ScanImage, StencilReference
from infrastructure import ConfigService, FileManager
from infrastructure.logging_service import LoggingService


class ArtifactSaver:
    """Сохраняет базовые артефакты обработки."""

    def __init__(self, config_service: ConfigService, file_manager: FileManager):
        self.config_service = config_service
        self.file_manager = file_manager
        self.logger = LoggingService.get_logger(__name__)

    def save_compatibility_artifacts(
        self,
        scan_info: ScanImage,
        stencil_ref: StencilReference,
        original_scan: np.ndarray,
        pipeline_result: Any,
    ) -> Dict[str, Any]:
        """
        Сохраняет ключевые изображения для обратной совместимости и базового анализа.

        Args:
            scan_info: Информация о скане
            stencil_ref: Референсные данные Gerber
            original_scan: Исходное изображение скана
            pipeline_result: Результаты пайплайна

        Returns:
            Dict с результатами сохранения
        """
        order_number = scan_info.order_number
        try:
            workspace = self.file_manager.create_order_workspace(order_number)
            saved_files = {}

            # 1. Сохранение изображения Gerber
            if stencil_ref.gerber_image is not None:
                path = workspace / f"{order_number}_1_gerber.png"
                result = self._save_image_fast(path, stencil_ref.gerber_image)
                if result['success']:
                    self.logger.info(f"🖼️  Сохранен Gerber: {path.name}")
                    saved_files['gerber'] = result

            # 2. Сохранение оригинального скана
            path = workspace / f"{order_number}_2_original_scan.png"
            result = self._save_image_fast(path, original_scan)
            if result['success']:
                self.logger.info(f"🖼️  Сохранен исходный скан: {path.name}")
                saved_files['original_scan'] = result

            # 3. Сохранение финального результата (опционально)
            final_image = self._extract_final_image(pipeline_result)
            if final_image is not None and self.config_service.save_final_image:
                path = workspace / f"{order_number}_final.png"
                result = self._save_image_fast(path, final_image)
                if result['success']:
                    self.logger.info(
                        f"🖼️  Сохранен финальный результат: {path.name}")
                    saved_files['final_image'] = result

            return {
                'success': True,
                'workspace': str(workspace),
                'saved_files': saved_files
            }

        except Exception as e:
            self.logger.error(
                f"Ошибка при сохранении compatibility-артефактов: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _extract_final_image(self, pipeline_result: Any) -> Optional[np.ndarray]:
        """Извлекает финальное изображение из результата пайплайна."""
        if not getattr(pipeline_result, 'success', False):
            return None

        final_data = getattr(pipeline_result, 'final_result', None)
        if isinstance(final_data, np.ndarray):
            return final_data

        if isinstance(final_data, dict):
            # Ищем по приоритетным ключам
            for key in ['aligned_image', 'result_image', 'image']:
                if key in final_data and isinstance(final_data[key], np.ndarray):
                    return final_data[key]
        return None

    def _save_image_fast(self, path: Path, image: np.ndarray) -> Dict[str, Any]:
        """Универсальное и быстрое сохранение изображения."""
        try:
            # Нормализация изображения для сохранения
            if image.dtype == bool:
                image = image.astype(np.uint8) * 255
            elif image.dtype in (np.float32, np.float64):
                # Безопасная нормализация к диапазону 0-255
                min_val, max_val = np.min(image), np.max(image)
                if max_val > min_val:
                    image = 255 * (image - min_val) / (max_val - min_val)
                image = image.astype(np.uint8)

            # Преобразование RGB -> BGR для cv2.imwrite
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_to_save = image

            # Кодирование в память и запись на диск для поддержки кириллицы
            ext = path.suffix
            success, buffer = cv2.imencode(ext, image_to_save)
            if not success:
                raise IOError(
                    f"Не удалось закодировать изображение для {path.name}")

            path.write_bytes(buffer)
            return {'success': True, 'path': str(path)}

        except Exception as e:
            self.logger.error(
                f"Ошибка сохранения изображения {path.name}: {e}")
            return {'success': False, 'error': str(e)}
