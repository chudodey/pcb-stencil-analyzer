# order/order_coordinator.py
"""
Координатор обработки заказов
"""
import time
import traceback
from typing import Optional, List
from pathlib import Path

# Импорты доменных моделей
from domain import OrderResult, ProcessingSession, ProcessedScan
from infrastructure import ConfigService, UIService, FileManager
from infrastructure.debug_formatter import DebugFormatter
from order.order_manager import OrderManager
from processing import MatplotlibVisualizer


class OrderCoordinator:
    """Координатор обработки заказов"""

    def __init__(self, config_service: ConfigService, ui_service: UIService,
                 file_manager: FileManager):
        """
        Инициализация координатора

        Args:
            config_service: Сервис конфигурации
            ui_service: Сервис пользовательского интерфейса
            file_manager: Файловый менеджер
        """
        self.config_service = config_service
        self.ui_service = ui_service
        self.file_manager = file_manager
        self.order_manager = OrderManager(
            config_service, ui_service, file_manager)

        # Инициализация DebugFormatter для отладочного вывода
        self.debug_fmt = DebugFormatter(config_service.debug_mode, __name__)

    def process_order(self, order_number: str,
                      session: ProcessingSession) -> OrderResult:
        """
        Полная обработка заказа

        Args:
            order_number: Номер заказа
            session: Текущая сессия обработки

        Returns:
            Результат обработки заказа
        """
        # ШАГ 2: Обработка Gerber-файла
        self.ui_service.show_main_stage('GERBER')

        gerber_files = self.file_manager.find_gerber_files(order_number)
        self.ui_service.show_gerber_search_result(gerber_files, order_number)

        if not gerber_files:
            self.debug_fmt.error(
                f"Gerber файл не найден для заказа {order_number}")
            return OrderResult(
                order_number=order_number,
                processed_scans=[],
                processing_errors=["Gerber файл не найден"]
            )

        order_state = self.order_manager.create_order(order_number)

        try:
            self.ui_service.show_processing_step("Обработка Gerber файла...")
            stencil_ref = self.order_manager.load_gerber(
                order_state, gerber_files[0])

            # Отладочная информация о Gerber
            self._debug_gerber_loaded(stencil_ref)

        except Exception as e:
            # Детальный дебаг ошибки Gerber
            self._debug_gerber_error(e)

            self.ui_service.show_gerber_error(order_number, str(e))
            action = self.ui_service.ask_parsing_failed_action(str(e))
            result = self._create_failed_order_result(
                order_number, [str(e)], action)
            if action == '2':  # Выход
                result.next_action = "exit"
            elif action == '1':  # Новый заказ
                result.next_action = "new_order"
            return result

        # Основной цикл обработки сканов
        processed_scans = []

        while True:
            # ШАГ 3: Получение скана
            self.ui_service.show_main_stage('ПОЛУЧЕНИЕ СКАНА')

            scan_path = self._wait_for_scan_with_ui()
            if not scan_path:
                self.debug_fmt.info("Ожидание скана отменено пользователем")
                return self.order_manager.finalize_order(order_state, processed_scans)

            # ШАГ 4: Обработка скана
            self.ui_service.show_main_stage('ОБРАБОТКА СКАНА')

            try:
                processed_scan = self.order_manager.process_scan(
                    order_state, scan_path, stencil_ref
                )
                processed_scans.append(processed_scan)

                # Отладочная информация о скане
                self._debug_scan_processed(processed_scan)

                # ШАГ 5: Совмещение
                self.ui_service.show_main_stage('СОВМЕЩЕНИЕ')

                # Используем правильный метод display_alignment_summary
                is_aligned = self.ui_service.display_alignment_summary(
                    processed_scan)

                if is_aligned:
                    # Успех - создаем и показываем совмещенное изображение
                    self._handle_successful_alignment(
                        processed_scan, stencil_ref, order_number)

                    # При успехе завершаем заказ
                    result = self.order_manager.finalize_order(
                        order_state, processed_scans)
                    self.ui_service.show_order_summary(
                        order_number, len(processed_scans), True)
                    self.debug_fmt.success(
                        f"Заказ {order_number} успешно обработан")
                    return result
                else:
                    # Неудача - спрашиваем пользователя что делать
                    action = self._handle_processing_failure(processed_scan)
                    if action == "exit":
                        result = self.order_manager.finalize_order(
                            order_state, processed_scans)
                        result.next_action = "exit"
                        self.ui_service.show_order_summary(
                            order_number, len(processed_scans), False)
                        self.debug_fmt.warn(
                            f"Заказ {order_number} завершен с ошибками")
                        return result
                    elif action == "new_order":
                        result = self.order_manager.finalize_order(
                            order_state, processed_scans)
                        result.next_action = "new_order"
                        self.ui_service.show_order_summary(
                            order_number, len(processed_scans), False)
                        self.debug_fmt.warn(
                            f"Заказ {order_number} завершен с ошибками")
                        return result
                    # action == "retry" - продолжаем цикл while

            except Exception as e:
                self.ui_service.show_scan_error(scan_path, str(e))
                action = self.ui_service.ask_alignment_failed_action(0.0)
                if action == "2":  # Новый заказ
                    result = self._create_failed_order_result(
                        order_number, [str(e)], action)
                    result.next_action = "new_order"
                    self.debug_fmt.error(f"Ошибка обработки скана: {e}")
                    return result
                elif action == "3":  # Выход
                    result = self._create_failed_order_result(
                        order_number, [str(e)], action)
                    result.next_action = "exit"
                    self.debug_fmt.error(f"Ошибка обработки скана: {e}")
                    return result
                # action == "1" (retry) - продолжаем цикл

    def _handle_successful_alignment(self, processed_scan: ProcessedScan,
                                     stencil_ref, order_number: str):
        """
        Обработка успешного совмещения - создание и отображение результата.

        Args:
            processed_scan: Обработанный скан
            stencil_ref: Референс Gerber
            order_number: Номер заказа
        """
        try:
            # ОБНОВЛЕННЫЙ ВЫЗОВ: Создаем MatplotlibVisualizer
            visualizer = MatplotlibVisualizer(
                self.config_service, self.file_manager)

            # Подготавливаем данные для визуализатора
            visualization_data = self._prepare_visualization_data(
                processed_scan, stencil_ref)

            # Создаем и сохраняем финальный отчет
            output_path = visualizer._save_final_comparison_sync(
                visualization_data, order_number
            )

            if output_path and Path(output_path).exists():
                # ИСПРАВЛЕНИЕ 2: Передаем путь к файлу, а не изображение
                self.ui_service.show_combined_image(
                    output_path, f"Финальный отчет - Заказ {order_number}")

                self.debug_fmt.info(f"Финальный отчет сохранен: {output_path}")
            else:
                self.ui_service.show_warning(
                    "Не удалось создать финальный отчет")
                self.debug_fmt.warn("Создание финального отчета не удалось")
                # Fallback: показываем хотя бы выровненное изображение
                if (hasattr(processed_scan, 'scan_analysis') and
                    hasattr(processed_scan.scan_analysis, 'aligned_image') and
                        processed_scan.scan_analysis.aligned_image is not None):
                    self.ui_service.show_combined_image(
                        processed_scan.scan_analysis.aligned_image,
                        f"Выровненный скан - Заказ {order_number}"
                    )

        except Exception as e:
            self.ui_service.show_warning(
                f"Ошибка при создании финального отчета: {e}")
            self.debug_fmt.error(f"Ошибка создания отчета: {e}")
            # Fallback: показываем просто выровненное изображение
            if (hasattr(processed_scan, 'scan_analysis') and
                hasattr(processed_scan.scan_analysis, 'aligned_image') and
                    processed_scan.scan_analysis.aligned_image is not None):
                self.ui_service.show_combined_image(
                    processed_scan.scan_analysis.aligned_image,
                    f"Выровненный скан - Заказ {order_number}"
                )

    def _prepare_visualization_data(self, processed_scan: ProcessedScan, stencil_ref) -> dict:
        """
        Подготавливает данные для визуализатора Matplotlib.

        Args:
            processed_scan: Обработанный скан
            stencil_ref: Референс Gerber

        Returns:
            Словарь с данными для визуализации
        """
        visualization_data = {}

        # Оригинальный скан
        if (hasattr(processed_scan, 'scan_analysis') and
            hasattr(processed_scan.scan_analysis, 'original_image') and
                processed_scan.scan_analysis.original_image is not None):
            visualization_data['original_scan'] = processed_scan.scan_analysis.original_image
        else:
            visualization_data['original_scan'] = None

        # Референсное изображение (Gerber)
        if (hasattr(stencil_ref, 'gerber_image') and
                stencil_ref.gerber_image is not None):
            visualization_data['reference_image'] = stencil_ref.gerber_image
        else:
            visualization_data['reference_image'] = None

        # Выровненный скан
        if (hasattr(processed_scan, 'scan_analysis') and
            hasattr(processed_scan.scan_analysis, 'aligned_image') and
                processed_scan.scan_analysis.aligned_image is not None):
            visualization_data['aligned_scan'] = processed_scan.scan_analysis.aligned_image
        else:
            visualization_data['aligned_scan'] = None

        return visualization_data

    def _wait_for_scan_with_ui(self) -> Optional[Path]:
        """
        Ожидание скана с UI-интерфейсом

        Returns:
            Путь к файлу скана или None если отменено
        """
        self.ui_service.show_scanning_instructions()

        try:
            # Используем правильный метод wait_for_scan вместо wait_for_scan_file
            scan_path = self.file_manager.wait_for_scan(
                callback=self.ui_service.show_scan_waiting_animation
            )

            if scan_path:
                # Используем правильный метод show_scan_details
                image_info = self.file_manager.get_image_info(scan_path)
                self.ui_service.show_scan_details(
                    scan_path=scan_path,
                    dpi=image_info['dpi'],
                    size_pixels=image_info['size_pixels'],
                    size_mm=image_info['size_mm'],
                    file_size=image_info['file_size_bytes']
                )
                self.debug_fmt.info(f"Получен скан: {scan_path.name}")
                return scan_path
            else:
                # Таймаут - спрашиваем что делать
                last_file = self.file_manager.get_last_existing_scan()
                action = self.ui_service.ask_scan_timeout_action(last_file)

                if action == "1":  # Продолжить ожидание
                    return self._wait_for_scan_with_ui()
                elif action == "2" and last_file:  # Использовать последний файл
                    image_info = self.file_manager.get_image_info(last_file)
                    self.ui_service.show_scan_details(
                        scan_path=last_file,
                        dpi=image_info['dpi'],
                        size_pixels=image_info['size_pixels'],
                        size_mm=image_info['size_mm'],
                        file_size=image_info['file_size_bytes']
                    )
                    self.debug_fmt.info(
                        f"Использован последний скан: {last_file.name}")
                    return last_file
                elif action == "3":  # Новый заказ
                    self.debug_fmt.info("Пользователь выбрал новый заказ")
                    return None
                elif action == "4":  # Выход
                    self.debug_fmt.info("Пользователь выбрал выход")
                    return None

        except Exception as e:
            self.ui_service.show_error(f"Ошибка при ожидании скана: {e}")
            self.debug_fmt.error(f"Ошибка ожидания скана: {e}")
            return None

    def _handle_processing_failure(self, processed_scan: ProcessedScan) -> str:
        """
        Обработка неудачного совмещения

        Args:
            processed_scan: Результат обработки скана

        Returns:
            Действие: "retry", "new_order", "exit"
        """
        correlation = processed_scan.alignment.alignment_score
        action = self.ui_service.ask_alignment_failed_action(correlation)

        if action == "1":  # Повторить сканирование
            self.debug_fmt.info("Пользователь выбрал повторное сканирование")
            return "retry"
        elif action == "2":  # Новый заказ
            self.debug_fmt.info("Пользователь выбрал новый заказ")
            return "new_order"
        elif action == "3":  # Выход
            self.debug_fmt.info("Пользователь выбрал выход")
            return "exit"
        else:
            return "retry"  # fallback

    def _create_failed_order_result(self,
                                    order_number: str,
                                    errors: List[str],
                                    action: str) -> OrderResult:
        """
        Создает результат заказа с ошибкой

        Args:
            order_number: Номер заказа
            errors: Список ошибок
            action: Выбранное действие

        Returns:
            Результат с ошибкой
        """
        result = OrderResult(
            order_number=order_number,
            processed_scans=[],
            processing_errors=errors
        )

        if action == '2':  # Выход
            result.next_action = "exit"
        elif action == '1':  # Новый заказ
            result.next_action = "new_order"

        return result

    def _debug_gerber_loaded(self, stencil_ref):
        """Отладочная информация о загруженном Gerber."""
        self.debug_fmt.debug("Gerber файл успешно загружен")

        gerber_info = {
            "Order number": stencil_ref.order_number,
            "Gerber filename": stencil_ref.gerber_filename,
            "Stencil size": f"{stencil_ref.stencil_size_mm} mm",
            "Board shape": str(stencil_ref.board_shape),
            "Aperture count": stencil_ref.aperture_metrics.count
        }

        if hasattr(stencil_ref, 'gerber_image') and stencil_ref.gerber_image is not None:
            gerber_info["Image shape"] = stencil_ref.gerber_image.shape
        else:
            gerber_info["Image"] = "Not available"

        self.debug_fmt.box("Gerber Details", gerber_info)

    def _debug_scan_processed(self, processed_scan: ProcessedScan):
        """Отладочная информация об обработанном скане."""
        scan_info = {
            "Alignment score": f"{processed_scan.alignment.alignment_score:.3f}",
            "Is aligned": processed_scan.alignment.is_aligned,
            "Rotation": f"{processed_scan.alignment.alignment_metrics.rotation_angle:.2f}°",
            "Offset X": f"{processed_scan.alignment.alignment_metrics.shift_x_px:.1f} px",
            "Offset Y": f"{processed_scan.alignment.alignment_metrics.shift_y_px:.1f} px"
        }

        if (hasattr(processed_scan, 'scan_analysis') and
                processed_scan.scan_analysis.aligned_image is not None):
            scan_info["Aligned image shape"] = processed_scan.scan_analysis.aligned_image.shape
        else:
            scan_info["Aligned image"] = "Not available"

        if (hasattr(processed_scan, 'scan_analysis') and
                hasattr(processed_scan.scan_analysis, 'contours') and
                processed_scan.scan_analysis.contours):
            scan_info["Contours count"] = len(
                processed_scan.scan_analysis.contours)
        else:
            scan_info["Contours"] = "Not available"

        self.debug_fmt.box(
            "OrderCoordinator: Scan Processing Results", scan_info)

    def _debug_gerber_error(self, error: Exception):
        """Детальный дебаг ошибки Gerber."""
        self.debug_fmt.error("Ошибка загрузки Gerber файла")
        self.debug_fmt.error(f"Exception: {error}")

        # Выводим traceback только в самом детальном режиме
        if self.config_service.debug_mode:
            self.debug_fmt.debug("Full traceback:")
            self.debug_fmt.debug(traceback.format_exc())
