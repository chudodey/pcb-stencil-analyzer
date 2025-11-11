# order/order_manager.py
"""
Менеджер заказов - управление заказами и их метриками

ИЗМЕНЕНИЯ:
- ProcessingEngine теперь получает progress_callback
- Убраны прямые вызовы UI из ProcessingEngine
- Добавлен DebugFormatter для отладочного вывода
"""
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from domain.data_models import (
    OrderResult, StencilReference, ProcessedScan,
)
from infrastructure import ConfigService, UIService, FileManager
from infrastructure.debug_formatter import DebugFormatter
from processing.processing_engine import ProcessingEngine


class OrderState:
    """Состояние обработки заказа"""

    def __init__(self, order_number: str):
        self.order_number = order_number
        self.start_time = datetime.now()
        self.stencil_reference: Optional[StencilReference] = None
        self.processing_errors: List[str] = []


class OrderManager:
    """Управление заказами и их статистикой"""

    def __init__(self, config_service: ConfigService, ui_service: UIService, file_manager: FileManager):
        """
        Инициализация менеджера заказов

        Args:
            config_service: Сервис конфигурации
            ui_service: Сервис пользовательского интерфейса
            file_manager: Файловый менеджер
        """
        self.config_service = config_service
        self.ui_service = ui_service
        self.file_manager = file_manager
        self.debug_fmt = DebugFormatter(config_service.debug_mode, __name__)
        self.active_orders: Dict[str, OrderState] = {}

        self._processing_engine = None

    @property
    def processing_engine(self):
        """
        Ленивая инициализация ProcessingEngine

        ИЗМЕНЕНИЕ: Передаём progress_callback вместо ui_service
        """
        if self._processing_engine is None:

            # Создаём callback функцию, которая делегирует в UI
            def progress_callback(message: str):
                self.ui_service.show_processing_step(message)

            self._processing_engine = ProcessingEngine(
                self.config_service,
                self.file_manager,
                progress_callback=progress_callback  # ← ИЗМЕНЕНИЕ
            )
            self.debug_fmt.success("ProcessingEngine инициализирован")
        return self._processing_engine

    def create_order(self, order_number: str) -> OrderState:
        """
        Создание нового заказа

        Args:
            order_number: Номер заказа

        Returns:
            Объект состояния заказа
        """
        if order_number in self.active_orders:
            self.ui_service.show_warning(
                f"Заказ {order_number} уже был в работе. Начинаем заново.")
            self.debug_fmt.warn(f"Пересоздание заказа {order_number}")

        order_state = OrderState(order_number)
        self.active_orders[order_number] = order_state

        self.debug_fmt.info(f"Создан новый заказ: {order_number}")
        self.debug_fmt.debug(f"Активных заказов: {len(self.active_orders)}")

        return order_state

    def load_gerber(self, order_state: OrderState, gerber_path: Path) -> StencilReference:
        """
        Загрузка и анализ Gerber-файла

        Args:
            order_state: Состояние заказа
            gerber_path: Путь к Gerber-файлу

        Returns:
            Эталонные данные трафарета

        Raises:
            Exception: Если не удалось обработать Gerber
        """
        self.debug_fmt.info(f"Загрузка Gerber: {gerber_path.name}")

        try:
            # TODO тут программа сразу идёт в process engine и запускается
            # TODO инициализация стратегий, а это нужно сделать либо раньше, либо позже
            # TODO может быть упрозднить или упростить order_manager
            stencil_ref = self.processing_engine.analyze_gerber(
                order_state.order_number, gerber_path
            )
            order_state.stencil_reference = stencil_ref

            # Отладочная информация о загруженном Gerber
            gerber_metrics = {
                "Файл": gerber_path.name,
                "Размер платы": f"{stencil_ref.stencil_size_mm[0]:.1f}×{stencil_ref.stencil_size_mm[1]:.1f} мм",
                "Количество контуров": len(stencil_ref.contours) if stencil_ref.contours else 0,
                "Количество апертур": stencil_ref.aperture_metrics.count
            }
            self.debug_fmt.box("Gerber Analysis Results", gerber_metrics)

            # Показываем результаты парсинга оператору
            metrics_to_show = {
                'board_width_mm': stencil_ref.stencil_size_mm[0] - 2 * self.config_service.gerber_margin_mm,
                'board_height_mm': stencil_ref.stencil_size_mm[1] - 2 * self.config_service.gerber_margin_mm,
                'contour_count': len(stencil_ref.contours) if stencil_ref.contours else 0
            }
            self.ui_service.show_parsing_stats(metrics_to_show)

            self.debug_fmt.success("Gerber файл успешно обработан")
            return stencil_ref

        except Exception as e:
            error_msg = f"Ошибка анализа Gerber: {e}"
            order_state.processing_errors.append(error_msg)
            self.debug_fmt.error(error_msg)
            raise

    def process_scan(self, order_state: OrderState, scan_path: Path,
                     stencil_ref: StencilReference) -> ProcessedScan:
        """
        Обработка одного скана

        Args:
            order_state: Состояние заказа
            scan_path: Путь к файлу скана
            stencil_ref: Эталонные данные трафарета

        Returns:
            Результат обработки скана

        Raises:
            Exception: Если не удалось обработать скан
        """
        self.debug_fmt.info(f"Обработка скана: {scan_path.name}")

        try:
            processed_scan = self.processing_engine.process_scan(
                scan_path, stencil_ref
            )

            # Отладочная информация о результате обработки
            if processed_scan and hasattr(processed_scan, 'alignment'):
                alignment_info = {
                    "Score": f"{processed_scan.alignment.alignment_score:.3f}",
                    "Status": "Aligned" if processed_scan.alignment.is_aligned else "Failed",
                    "Rotation": f"{processed_scan.alignment.alignment_metrics.rotation_angle:.2f}°",
                    "Offset": f"X:{processed_scan.alignment.alignment_metrics.shift_x_px:.1f}, Y:{processed_scan.alignment.alignment_metrics.shift_y_px:.1f} px"
                }
                # TODO дублирование информации с Order Coordinator, возможно выкинуть
                self.debug_fmt.metrics_table(
                    "OrderManager: Alignment Results", alignment_info)

            self.debug_fmt.success("Скан успешно обработан")
            return processed_scan

        except Exception as e:
            error_msg = f"Ошибка обработки скана: {e}"
            order_state.processing_errors.append(error_msg)
            self.debug_fmt.error(error_msg)
            raise

    def finalize_order(self, order_state: OrderState,
                       processed_scans: List[ProcessedScan]) -> OrderResult:
        """
        Финализация заказа и создание итогового результата

        Args:
            order_state: Состояние заказа
            processed_scans: Список обработанных сканов

        Returns:
            Итоговый результат обработки заказа
        """
        self.debug_fmt.info(f"Финализация заказа: {order_state.order_number}")

        board_size = ""
        polygon_count = 0

        if order_state.stencil_reference:
            width, height = order_state.stencil_reference.stencil_size_mm
            board_size = f"{width:.1f}x{height:.1f} мм"
            polygon_count = order_state.stencil_reference.aperture_metrics.count

        order_result = OrderResult(
            order_number=order_state.order_number,
            processed_scans=processed_scans,
            board_size=board_size,
            polygon_count=polygon_count,
            processing_errors=order_state.processing_errors.copy()
        )

        # Статистика заказа для отладки
        order_stats = {
            "Обработано сканов": len(processed_scans),
            "Ошибки": len(order_state.processing_errors),
            "Размер платы": board_size,
            "Апертуры": polygon_count
        }
        self.debug_fmt.metrics_table("Order Statistics", order_stats)

        # Удаляем заказ из активных
        if order_state.order_number in self.active_orders:
            del self.active_orders[order_state.order_number]
            self.debug_fmt.debug(
                f"Заказ удален из активных. Осталось: {len(self.active_orders)}")

        self.debug_fmt.success(
            f"Заказ {order_state.order_number} финализирован")
        return order_result
