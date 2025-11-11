# app/application.py
"""
Главный класс приложения
"""
from typing import Optional
from datetime import datetime

# Импорты инфраструктуры
from infrastructure import ConfigService, UIService, FileManager, DebugFormatter
# Импорты доменных моделей
from domain import Operator, ProcessingSession
from app.session_manager import SessionManager
from order.order_coordinator import OrderCoordinator


class StencilAnalyzerApplication:
    """Главный класс приложения, управляющий жизненным циклом"""

    def __init__(self,
                 config_service: ConfigService,
                 ui_service: UIService,
                 file_manager: FileManager):
        """
        Инициализация приложения

        Args:
            config_service: Сервис конфигурации
            ui_service: Сервис пользовательского интерфейса
            file_manager: Менеджер файловых операций (включает DirectoryManager функциональность)
        """
        self.config_service = config_service
        self.ui_service = ui_service
        self.file_manager = file_manager
        self.debug_formatter = DebugFormatter(
            config_service.debug_mode, __name__)

        self.session_manager: Optional[SessionManager] = None
        self.order_coordinator: Optional[OrderCoordinator] = None
        self._initialized = False
        self._startup_time = datetime.now()

    def initialize(self) -> bool:
        """
        Инициализация приложения

        Returns:
            True если инициализация прошла успешно
        """
        try:
            self.ui_service.show_main_stage('ЗАПУСК СЕССИИ - ИНИЦИАЛИЗАЦИЯ')

            # Создание менеджеров
            self.session_manager = SessionManager()

            # Создание координатора заказов
            self.order_coordinator = OrderCoordinator(
                self.config_service,
                self.ui_service,
                self.file_manager
            )

            self._initialized = True
            self.ui_service.show_success("Приложение инициализировано успешно")
            return True

        except Exception as e:  # pylint: disable=broad-except
            self.ui_service.show_error(f"Ошибка инициализации: {e}")
            return False

    def run(self) -> None:
        """
        Основной цикл работы приложения

        Выполняет:
        - Показ заголовка
        - Создание сессии оператора
        - Основной цикл обработки заказов
        - Обработку пользовательских команд

        Raises:
            RuntimeError: Если приложение не инициализировано
        """
        if not self._initialized:
            raise RuntimeError("Приложение не инциализировано")

        # Проверяем, что менеджеры инициализированы
        if self.session_manager is None or self.order_coordinator is None:
            raise RuntimeError(
                "Неправильно инициализированы компоненты приложения")

        # ШАГ 1: Создание сессии оператора
        self.ui_service.show_main_stage('СОЗДАНИЕ СЕССИИ ОПЕРАТОРА')
        operator = self._create_operator()
        session = self.session_manager.create_session(operator)

        # self.ui_service.show_success(f"Оператор: {operator.full_name}")
        self.ui_service.show_success(f"Сессия создана: {session.session_id}")

        try:
            # Основной цикл обработки заказов
            while True:
                # Получение номера заказа
                order_number = self.ui_service.get_order_number()
                if not order_number:
                    if self.ui_service.confirm_exit():
                        break
                    continue

                # Делегируем обработку заказа координатору
                try:
                    order_result = self.order_coordinator.process_order(
                        order_number, session
                    )

                    # Добавляем результат в сессии
                    session.orders_processed[order_number] = order_result

                    # Отладочная статистика сессии
                    if self.config_service.debug_mode:
                        self._debug_session_stats(session)

                    # Проверяем, какое действие нужно выполнить дальше
                    if hasattr(order_result, 'next_action'):
                        if order_result.next_action == "exit":
                            if self.ui_service.confirm_exit():
                                break
                            continue
                        elif order_result.next_action == "new_order":
                            continue  # Переходим к следующей итерации для нового заказа

                    # Если координатор не определил next_action, показываем главное меню
                    # (это происходит только в случае успешной обработки заказа)
                    choice = self.ui_service.show_main_menu()
                    if choice == '2':  # Выход
                        if self.ui_service.confirm_exit():
                            break
                    # choice == '1' - новый заказ, продолжаем цикл

                except Exception as e:  # pylint: disable=broad-except
                    self.ui_service.show_error(
                        f"Ошибка обработки заказа {order_number}: {e}")
                    # Продолжаем работу - не падаем из-за ошибок в заказе
                    continue

        except KeyboardInterrupt:
            self.ui_service.show_warning(
                "Программа прервана пользователем", add_newline=True)
        except Exception as e:  # pylint: disable=broad-except
            self.ui_service.show_error(
                f"Критическая ошибка в главном цикле: {e}")
            raise
        finally:
            # Завершение сессии
            self.ui_service.show_main_stage('ЗАВЕРШЕНИЕ СЕССИИ')
            if self.session_manager:
                closed_session = self.session_manager.close_session()
                if closed_session:
                    self._show_session_summary(closed_session)

    def shutdown(self) -> None:
        """
        Корректное завершение работы приложения

        Выполняет:
        - Закрытие текущей сессии
        - Сохранение данных
        - Освобождение ресурсов
        - Показ финального сообщения
        """
        try:
            # Закрываем сессию если была активна
            if self.session_manager and self.session_manager.is_session_active():
                self.ui_service.show_main_stage('ЗАВЕРШЕНИЕ РАБОТЫ')
                closed_session = self.session_manager.close_session()
                if closed_session and self.config_service.debug_mode:
                    self._show_session_summary(closed_session)

            # Показываем сообщение о завершении
            # self.ui_service.show_success("Работа программы завершена")

        except Exception as e:  # pylint: disable=broad-except
            # Не падаем при завершении, только логируем
            self.debug_formatter.error(f"Ошибка при завершении: {e}")

    def _create_operator(self) -> Operator:
        """
        Создание объекта оператора на основе пользовательского ввода

        Returns:
            Объект оператора
        """
        full_name = self.ui_service.get_operator_name()

        # Генерируем ID оператора из ФИО
        operator_id = full_name.replace(" ", "_").lower()

        self.debug_formatter.debug(
            f"Создан оператор: {full_name} (ID: {operator_id})")
        return Operator(
            id=operator_id,
            full_name=full_name,
            department=None  # Можно расширить в будущем
        )

    def _debug_session_stats(self, session: ProcessingSession) -> None:
        """
        Показ отладочной статистики сессии

        Args:
            session: Активная сессия
        """
        stats = {
            "total_orders": session.total_orders,
            "total_scans": session.total_scans,
            "successful_scans": session.successful_scans,
            "success_rate": f"{session.success_rate:.1f}%"
        }
        self.debug_formatter.metrics_table("Статистика сессии", stats)

    def _show_session_summary(self, session: ProcessingSession) -> None:
        """
        Показ итоговой статистики сессии для оператора

        Args:
            session: Завершенная сессия
        """
        duration_min = session.duration_seconds / 60 if session.duration_seconds else 0

        summary_items = [
            ("Оператор", session.operator.full_name),
            ("Длительность", f"{duration_min:.1f} мин"),
            ("Обработано заказов", str(session.total_orders)),
            ("Всего сканов", str(session.total_scans)),
            ("Успешных", str(session.successful_scans)),
            ("Процент успеха", f"{session.success_rate:.1f}%")
        ]

        self.ui_service.show_compact_block("ИТОГИ СЕССИИ", summary_items)
