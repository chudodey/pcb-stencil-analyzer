# app/session_manager.py
"""
Менеджер сессий обработки
"""
from typing import Optional
from datetime import datetime
from domain.data_models import ProcessingSession, Operator

from infrastructure import DebugFormatter, ConfigService


class SessionManager:
    """Управление сессиями обработки"""

    def __init__(self, config_service: Optional[ConfigService] = None):
        """Инициализация менеджера сессий"""
        self.current_session: Optional[ProcessingSession] = None
        self.debug_formatter = DebugFormatter(
            config_service if config_service else ConfigService(),
            __name__
        )

    def create_session(self, operator: Operator) -> ProcessingSession:
        """
        Создание новой сессии

        Args:
            operator: Данные оператора

        Returns:
            Созданная сессия
        """
        # Закрываем предыдущую сессию если была
        if self.current_session:
            self.debug_formatter.debug(
                "Закрытие предыдущей сессии перед созданием новой")
            self.close_session()

        # Генерируем ID сессии
        session_id = f"{operator.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Создаем новую сессию
        self.current_session = ProcessingSession(
            session_id=session_id,
            operator=operator,
            start_time=datetime.now()
        )

        # self.debug_formatter.info(f"Создана новая сессия: {session_id}")
        # self.debug_formatter.debug(f"Оператор сессии: {operator.full_name}")
        # TODO почему-то этом месте debug режим не выводиться :/ проверить почему

        return self.current_session

    def close_session(self) -> Optional[ProcessingSession]:
        """
        Закрытие текущей сессии

        Returns:
            Закрытая сессия или None если не было активной сессии
        """
        if self.current_session:
            self.current_session.end_time = datetime.now()

            # Логируем статистику закрытой сессии
            session_stats = self._get_session_metrics()
            self.debug_formatter.metrics_table(
                "Статистика закрытой сессии", session_stats)

            closed_session = self.current_session
            self.current_session = None

            self.debug_formatter.info(
                f"Сессия закрыта: {closed_session.session_id}")
            return closed_session

        self.debug_formatter.debug("Попытка закрыть несуществующую сессию")
        return None

    def get_current_session(self) -> ProcessingSession:
        """
        Получение текущей активной сессии

        Returns:
            Текущая сессия

        Raises:
            RuntimeError: Если нет активной сессии
        """
        if not self.current_session:
            self.debug_formatter.error(
                "Попытка доступа к несуществующей сессии")
            raise RuntimeError("Нет активных сессий")
        return self.current_session

    def is_session_active(self) -> bool:
        """
        Проверка наличия активной сессии

        Returns:
            True если есть активная сессия
        """
        return self.current_session is not None

    def get_session_stats(self) -> dict:
        """
        Получение статистики текущей сессии

        Returns:
            Словарь со статистикой сессии
        """
        if not self.current_session:
            self.debug_formatter.debug(
                "Запрос статистики несуществующей сессии")
            return {}

        stats = self._get_session_metrics()
        self.debug_formatter.debug("Запрошена статистика текущей сессии")
        return stats

    def _get_session_metrics(self) -> dict:
        """
        Формирование метрик сессии для отладочного вывода

        Returns:
            Словарь с метриками сессии
        """
        if not self.current_session:
            return {}

        duration_seconds = (
            (datetime.now() - self.current_session.start_time).total_seconds()
            if not self.current_session.end_time
            else (self.current_session.end_time - self.current_session.start_time).total_seconds()
        )

        return {
            'session_id': self.current_session.session_id,
            'operator': self.current_session.operator.full_name,
            'duration_minutes': f"{duration_seconds / 60:.1f}",
            'total_orders': self.current_session.total_orders,
            'total_scans': self.current_session.total_scans,
            'successful_scans': self.current_session.successful_scans,
            'success_rate': f"{self.current_session.success_rate:.1f}%"
        }
