# test_debug_messages_v4.py
"""
Скрипт для тестирования и демонстрации возможностей модуля `DebugFormatter`.

НАЗНАЧЕНИЕ:
1. Тестирование: Проверка корректности работы всех методов отладочного вывода
2. Справка: Демонстрация использования DebugFormatter для разработчиков
3. Валидация: Проверка соответствия обновленной архитектуре

ОБНОВЛЕННАЯ АРХИТЕКТУРА:
- DebugFormatter: единый класс для всего отладочного вывода
- Прямая настройка через debug_mode параметр
- Логирование через LoggingService
- Гарантированный composite_score в корне метрик
- Четкое разделение ответственности по уровням

ЗАПУСК:
- python test_debug_messages_v4.py
- Создает лог-файл `test_debug.log` с детальным выводом
"""

from infrastructure.debug_formatter import DebugFormatter, LogLevel
from infrastructure.logging_service import LoggingService
import numpy as np
import time
import sys
from pathlib import Path
from typing import Dict, List, Any

# --- Настройка путей для импорта ---
sys.path.insert(0, str(Path(__file__).parent))

# --- Импорт реальных моделей данных ---
try:
    from domain.data_models import (
        AlignmentResult, AlignmentMetrics, AlignmentStatus, ProcessedScan,
        ScanImage, ScanAnalysisResult, ComparisonResult, StencilReference,
        BoardShape, ApertureMetrics, SpatialMetrics
    )
    from datetime import datetime
except ImportError as e:
    try:
        from data_models import (
            AlignmentResult, AlignmentMetrics, AlignmentStatus, ProcessedScan,
            ScanImage, ScanAnalysisResult, ComparisonResult, StencilReference,
            BoardShape, ApertureMetrics, SpatialMetrics
        )
        from datetime import datetime
    except ImportError:
        print(f"Ошибка импорта моделей данных: {e}")
        print("Убедитесь, что 'data_models.py' доступен для импорта.")
        sys.exit(1)


# ========================================================================
# ФАБРИКА ТЕСТОВЫХ ОБЪЕКТОВ
# ========================================================================

class MockObjectFactory:
    """Создает минимально валидные объекты реальных dataclasses."""

    @staticmethod
    def get_minimal_aperture_metrics() -> ApertureMetrics:
        """Минимальный ApertureMetrics."""
        return ApertureMetrics(
            count=100, mean_area=0.5, std_area=0.1, min_area=0.1, max_area=1.0,
            median_area=0.45, min_circularity=0.8, mean_circularity=0.9, mean_ellipticity=1.1
        )

    @staticmethod
    def get_minimal_spatial_metrics() -> SpatialMetrics:
        """Минимальный SpatialMetrics."""
        return SpatialMetrics(
            min_clearance_global=0.05, clearance_mean=0.1, clearance_std=0.02,
            clearance_percentiles={50: 0.1, 90: 0.15}
        )

    @staticmethod
    def get_minimal_stencil_reference() -> StencilReference:
        """Минимальный StencilReference."""
        return StencilReference(
            order_number="12345", gerber_filename="mock.gbr", gerber_path=Path("gerbers/mock.gbr"),
            stencil_size_mm=(100.0, 50.0), board_shape=BoardShape.HORIZONTAL,
            aperture_metrics=MockObjectFactory.get_minimal_aperture_metrics(),
            spatial_metrics=MockObjectFactory.get_minimal_spatial_metrics()
        )

    @staticmethod
    def get_minimal_scan_image() -> ScanImage:
        """Минимальный ScanImage."""
        return ScanImage(
            order_number="12345", filename="scan.tiff", scan_path=Path("scans/scan.tiff"),
            image_size_px=(4000, 3000), dpi=1200, scan_timestamp=datetime.now()
        )

    @staticmethod
    def get_minimal_scan_analysis_result() -> ScanAnalysisResult:
        """Минимальный ScanAnalysisResult."""
        return ScanAnalysisResult()

    @staticmethod
    def get_minimal_comparison_result(quality_score: float = 0.95) -> ComparisonResult:
        """Минимальный ComparisonResult."""
        return ComparisonResult(match_percentage=98.5, quality_score=quality_score)

    @staticmethod
    def get_mock_processed_scan(is_aligned: bool = True, correlation: float = 0.95) -> ProcessedScan:
        """Создает тестовый объект ProcessedScan."""
        metrics = AlignmentMetrics(
            correlation=correlation,
            rotation_angle=0.12,
            shift_x_px=10.5,
            shift_y_px=-5.0,
        )

        alignment = AlignmentResult(
            is_aligned=is_aligned,
            alignment_score=correlation,
            alignment_metrics=metrics,
            alignment_status=AlignmentStatus.SUCCESS if is_aligned else AlignmentStatus.FAILED
        )

        scan = ProcessedScan(
            scan_info=MockObjectFactory.get_minimal_scan_image(),
            scan_analysis=MockObjectFactory.get_minimal_scan_analysis_result(),
            alignment=alignment,
            comparison=MockObjectFactory.get_minimal_comparison_result(
                correlation),
            stencil_reference=MockObjectFactory.get_minimal_stencil_reference()
        )
        return scan

    @staticmethod
    def get_mock_contours() -> List[np.ndarray]:
        """Создает тестовые контуры для contour_info."""
        return [
            np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
            np.array([[7, 8], [9, 10]], dtype=np.int32),
            np.array([[11, 12], [13, 14], [15, 16], [17, 18]], dtype=np.int32),
        ]


# ========================================================================
# ОСНОВНОЙ ТЕСТЕР
# ========================================================================

class DebugMessagesTester:
    """
    Тестер всех возможностей модуля DebugFormatter.
    Демонстрирует использование всех методов для разработчиков.
    """

    def __init__(self):
        # НОВАЯ АРХИТЕКТУРА: DebugFormatter принимает только debug_mode
        LoggingService.configure(
            log_level="debug",
            debug=True,
            log_file=Path("test_debug.log"),
            datetime_format="%H:%M:%S"
        )

        self.debug = DebugFormatter(debug_mode=True, module_name=__name__)

        # Тестовые данные с гарантированным composite_score
        self.test_metrics = {
            "composite_score": 0.856,  # ← ГАРАНТИРОВАННО в корне
            "processing_time_s": 1.2345,
            "accuracy_pct": 0.9567,
            "memory_usage_mb": 128.45,
            "iterations": 1000,
            "success_rate": 0.995
        }

        self.test_strategies = {
            "adaptive_threshold": {
                "composite_score": 0.89,  # ← ГАРАНТИРОВАННО
                "score": 0.89, "time": 0.45, "passed": True
            },
            "otsu_binarization": {
                "composite_score": 0.92,  # ← ГАРАНТИРОВАННО
                "score": 0.92, "time": 0.32, "passed": True
            },
            "gaussian_blur": {
                "composite_score": 0.78,  # ← ГАРАНТИРОВАННО
                "score": 0.78, "time": 0.67, "passed": False
            },
            "morphological_ops": {
                "composite_score": 0.95,  # ← ГАРАНТИРОВАННО
                "score": 0.95, "time": 0.23, "passed": True
            }
        }

        # Тестовые объекты для методов отладки
        self.mock_scan_ok = MockObjectFactory.get_mock_processed_scan(
            is_aligned=True, correlation=0.95)
        self.mock_scan_fail = MockObjectFactory.get_mock_processed_scan(
            is_aligned=False, correlation=0.55)

    def show_architecture_overview(self):
        """Демонстрация обновленной архитектуры системы."""
        self.debug.header("🏗️ ОБНОВЛЕННАЯ АРХИТЕКТУРА DEBUGFORMATTER")

        self.debug.info("Упрощенная инициализация:")
        self.debug.info(
            "• DebugFormatter(debug_mode=True, module_name=__name__)", indent=1)
        self.debug.info("• Больше не требуется ConfigService", indent=1)
        self.debug.info(
            "• Прямой контроль через debug_mode параметр", indent=1)

        self.debug.info("")
        self.debug.info("Основные принципы форматирования:")
        self.debug.info("• Префикс ВСЕГДА в колонке 0", indent=1)
        self.debug.info(
            "• Отступы ТОЛЬКО для текста (не для префиксов)", indent=1)
        self.debug.info("• Структурные элементы БЕЗ префиксов", indent=1)
        self.debug.info("• Все сообщения серого цвета", indent=1)

        self.debug.info("")
        self.debug.info("🎯 НОВАЯ АРХИТЕКТУРА РАСЧЕТА МЕТРИК:")
        self.debug.info("• Стратегии: рассчитывают composite_score", indent=1)
        self.debug.info("• Evaluator: корректирует → final_score", indent=1)
        self.debug.info("• StageRunner: отображает рейтинг", indent=1)
        self.debug.info(
            "• Гарантированный composite_score в корне метрик", indent=1)

    # ========================================================================
    # ЧАСТЬ 1: ОСНОВНЫЕ МЕТОДЫ DEBUGFORMATTER
    # ========================================================================

    def test_debug_basic_logging(self):
        """1.1. Базовые сообщения (debug, info, warn, error, success)"""
        self.debug.header("1.1. БАЗОВЫЙ ВЫВОД DEBUGFORMATTER")

        self.debug.info("Все сообщения DebugFormatter имеют серый цвет")
        self.debug.info("и предназначены для отладочной информации")

        self.debug.log("Прямой вызов .log()", LogLevel.INFO)
        self.debug.debug(
            "Отладочное сообщение (видно только в debug_mode=True)")
        self.debug.info("Информационное сообщение")
        self.debug.warn("Предупреждение о проблеме")
        self.debug.error("Сообщение об ошибке")
        self.debug.success("Успешное завершение операции")

        # Демонстрация новой логики отступов
        self.debug.info("Демонстрация отступов:")
        with self.debug.indent():
            self.debug.info("Сообщение с отступом 1")
            self.debug.info("Сообщение с отступом 2", indent=2)

    def test_debug_structural_elements(self):
        """1.2. Структурные элементы (header, subheader, section, separator)"""
        self.debug.subheader("1.2. СТРУКТУРНЫЕ ЭЛЕМЕНТЫ DEBUGFORMATTER")

        self.debug.info("Структурные элементы отображаются без префиксов")
        self.debug.info("для лучшей визуальной организации вывода")

        self.debug.header("ГЛАВНЫЙ ЗАГОЛОВОК (header)", width=60)
        self.debug.subheader("Подзаголовок (subheader)", width=50)
        self.debug.section("СЕКЦИЯ С ФАЗОЙ", phase="1/3", width=40)
        self.debug.section("СЕКЦИЯ БЕЗ ФАЗЫ", width=40)
        self.debug.separator(char="*")

    def test_debug_data_display(self):
        """1.3. Отображение данных (metrics_table, metrics_summary, box)"""
        self.debug.subheader("1.3. ОТОБРАЖЕНИЕ ДАННЫХ DEBUGFORMATTER")

        self.debug.info("Таблица метрик (metrics_table):")
        self.debug.info("• composite_score всегда в корне метрик", indent=1)
        self.debug.metrics_table(
            "Основные метрики", self.test_metrics, indent=1)

        self.debug.info("Сводка метрик (metrics_summary):")
        summary = {
            "composite_score": {"min": 0.7, "max": 0.99, "mean": 0.85, "std": 0.08},
            "time": {"min": 0.1, "max": 0.5, "mean": 0.3, "std": 0.1},
            "correlation": {"min": 0.8, "max": 0.99, "mean": 0.9, "std": 0.05}
        }
        self.debug.metrics_summary("Сводка", summary, indent=1)

        self.debug.info("Блок в рамке (box):")
        box_content = {
            "composite_score": 0.856,  # ← ГАРАНТИРОВАННО
            "Файл": "scan.tiff",
            "DPI": 1200,
            "Статус": "OK"
        }
        self.debug.box("Инфо о скане", box_content, indent=1, width=40)

    def test_debug_strategy_table(self):
        """1.4. Таблица оценки стратегий"""
        self.debug.subheader("1.4. ТАБЛИЦА СТРАТЕГИЙ DEBUGFORMATTER")

        self.debug.info("Таблицы стратегий отображаются без префиксов")
        self.debug.info("для лучшего визуального восприятия")
        self.debug.info("• Использует composite_score от стратегий", indent=1)

        self.debug.strategy_table_start(
            "Бинаризация", len(self.test_strategies))
        for name, data in self.test_strategies.items():
            self.debug.strategy_table_row(
                name, data["composite_score"], data["time"], data["passed"]
            )
        self.debug.strategy_table_end()

    def test_debug_specialized(self):
        """1.5. Специализированный вывод (contour_info, progress)"""
        self.debug.subheader("1.5. СПЕЦИАЛИЗИРОВАННЫЙ ВЫВОД DEBUGFORMATTER")

        self.debug.info("Информация о контурах (contour_info):")
        self.debug.contour_info(
            MockObjectFactory.get_mock_contours(), indent=1, max_display=2)

        self.debug.info("Прогресс-бар (progress):")
        total = 4

        # ВАЖНО: Прогресс-бар должен быть ОТДЕЛЬНО от других сообщений
        # Выводим начальное сообщение ДО прогресс-бара
        print()  # Новая строка для разделения

        for i in range(total + 1):
            self.debug.progress(i, total, f"Загрузка данных {i*25}%", indent=1)
            time.sleep(0.1)

        # ВАЖНО: НЕ используем print() после цикла - прогресс-бар сам завершит строку
        # Добавляем пустую строку для разделения с последующими сообщениями
        print()

    def test_debug_context_managers(self):
        """1.6. Контекстные менеджеры (indent, timed_section)"""
        self.debug.subheader("1.6. КОНТЕКСТНЫЕ МЕНЕДЖЕРЫ DEBUGFORMATTER")

        self.debug.info("Контекст отступов (indent):")
        with self.debug.indent(2):
            self.debug.info("Отступ x2")
            with self.debug.indent():
                self.debug.info("Отступ x3")

        self.debug.info("Контекст с таймером (timed_section):")
        with self.debug.timed_section("Тестовая операция", level=LogLevel.INFO):
            self.debug.info("Выполняется работа...", indent=1)
            time.sleep(0.2)

    # ========================================================================
    # ЧАСТЬ 2: ПРАВИЛА ФОРМАТИРОВАНИЯ И РАСПРЕДЕЛЕНИЕ ОТВЕТСТВЕННОСТИ
    # ========================================================================

    def test_formatting_rules_demonstration(self):
        """2.1. Демонстрация правил форматирования"""
        self.debug.header("2.1. ДЕМОНСТРАЦИЯ ПРАВИЛ ФОРМАТИРОВАНИЯ")

        self.debug.info(
            "ПРАВИЛО: Префикс ВСЕГДА в колонке 0, отступы ТОЛЬКО к тексту")

        self.debug.info("✅ ПРАВИЛЬНО: '[INFO]   текст с отступом'")
        self.debug.info("               ^^^^^^^   ^^^^^^^^^^^^^^^^")
        self.debug.info("               колонка 0  отступ применен")

        self.debug.info("")
        self.debug.info("❌ НЕПРАВИЛЬНО: '  [INFO]  текст'")
        self.debug.info("                ^^^^^^^^^^  ^^^^")
        self.debug.info("                префикс сдвинут - плохо!")

        self.debug.info("")
        self.debug.info("ПРАВИЛО: Структурные элементы БЕЗ префиксов")
        self.debug.info("Заголовки, разделители, таблицы не имеют префиксов")

    def test_common_mistakes_demonstration(self):
        """2.2. Демонстрация частых ошибок и их исправлений"""
        self.debug.header("2.2. ЧАСТЫЕ ОШИБКИ И ИСПРАВЛЕНИЯ")

        self.debug.info("❌ Ошибка: Отступ применен к префиксу")
        self.debug.info("   indent_str = ' ' * indent")
        self.debug.info("   message = f'{indent_str}{prefix} {text}'  # ПЛОХО")

        self.debug.info("✅ Исправление: Отступ только к тексту")
        self.debug.info("   indent_str = ' ' * indent")
        self.debug.info(
            "   message = f'{prefix} {indent_str}{text}'  # ХОРОШО")

        self.debug.info("")
        self.debug.info("❌ Ошибка: Забыли is_preformatted для заголовка")
        self.debug.info(
            "   self.logger.info(header, extra=self._log_extra)  # Добавится префикс")

        self.debug.info("✅ Исправление: Правильная обработка заголовка")
        self.debug.info(
            "   log_extra = {**self._log_extra, 'is_preformatted': True}")
        self.debug.info(
            "   self.logger.info(header, extra=log_extra)  # Без префикса")

    def test_responsibility_distribution(self):
        """2.3. Демонстрация распределения ответственности по уровням"""
        self.debug.header("2.3. РАСПРЕДЕЛЕНИЕ ОТВЕТСТВЕННОСТИ ПО УРОВНЯМ")

        self.debug.info("🎯 Уровень 4: ProcessingStrategy (Исполнитель)")
        self.debug.info("• Рассчитывает composite_score и метрики", indent=1)
        self.debug.info("• Логирует детали своего выполнения", indent=1)

        # Демонстрация стратегии
        self.debug.section(
            "Запуск стратегии: AdaptiveOtsuHybrid", phase="START")
        self.debug.debug("Тип: Гибридная бинаризация", indent=1)
        self.debug.debug("Адаптивные параметры:", indent=1)
        self.debug.debug("block_size: 31", indent=2)
        self.debug.debug("morph_kernel_size: 3", indent=2)
        self.debug.debug("Расчет метрик качества...", indent=2)
        self.debug.debug("composite_score гарантирован: 0.727", indent=3)

        metrics = {
            "composite_score": 0.727,  # ← ГАРАНТИРОВАННО в корне
            "component_count": 142,
            "size_uniformity": 0.854,
            "noise_ratio": 0.023
        }
        self.debug.metrics_table("Метрики выполнения", metrics, indent=1)
        self.debug.success(
            "Успешно | Качество: 0.727 | Время: 0.245с", indent=1)

        self.debug.info("")
        self.debug.info("🎯 Уровень 3: StrategyEvaluator (Корректор)")
        self.debug.info(
            "• Корректирует composite_score → final_score", indent=1)
        self.debug.info("• Учитывает контекст выполнения", indent=1)

        # Демонстрация коррекции
        self.debug.debug("[Прагматичная оценка AdaptiveOtsuHybrid]:", indent=1)
        self.debug.debug(
            "composite=0.727, time_penalty=0.950, final=0.690", indent=2)
        self.debug.debug(
            "Контекст: complexity=0.650, local_contrast=0.820", indent=3)

        self.debug.info("")
        self.debug.info("🎯 Уровень 2: StageRunner (Менеджер этапа)")
        self.debug.info("• Отображает сравнительную таблицу", indent=1)
        self.debug.info("• Объявляет лучшую стратегию", indent=1)

        # Демонстрация сводки этапа
        self.debug.subheader("Сводка этапа: BINARIZATION", indent=1)
        self.debug.strategy_table_start(
            "BINARIZATION", len(self.test_strategies), indent=1)
        for name, data in self.test_strategies.items():
            self.debug.strategy_table_row(
                name, data["composite_score"], data["time"], data["passed"], indent=1
            )
        self.debug.strategy_table_end(indent=1)
        self.debug.debug(
            "Лучшая стратегия: morphological_ops (оценка: 0.950)", indent=1)
        self.debug.success("Этап BINARIZATION завершен", indent=1)

    # ========================================================================
    # ЧАСТЬ 3: ИНТЕГРАЦИОННОЕ ТЕСТИРОВАНИЕ
    # ========================================================================

    def test_integration_scenarios(self):
        """3.1. Реальные сценарии использования системы"""
        self.debug.header("3.1. РЕАЛЬНЫЕ СЦЕНАРИИ ИСПОЛЬЗОВАНИЯ")

        # Сценарий 1: Успешная обработка с новой архитектурой
        self.debug.header("СЦЕНАРИЙ: УСПЕШНАЯ ОБРАБОТКА (НОВАЯ АРХИТЕКТУРА)")

        self.debug.info("🎯 ProcessingEngine: Бизнес-фазы для пользователя")
        self.debug.info("Фаза 5: Бинаризация")

        self.debug.info("")
        self.debug.info("🎯 StageRunner: Организация этапа")
        self.debug.section("Этап: BINARIZATION", phase="запуск")
        self.debug.debug("Доступно стратегий: 3", indent=1)

        # Демонстрация выполнения стратегий
        strategies_data = [
            {"name": "AdaptiveOtsuHybrid", "composite": 0.727,
                "time": 0.245, "passed": True},
            {"name": "AdaptiveThresholding", "composite": 0.520,
                "time": 0.180, "passed": False},
            {"name": "GlobalOtsu", "composite": 0.310,
                "time": 0.050, "passed": False}
        ]

        for strategy in strategies_data:
            self.debug.section(
                f"Запуск стратегии: {strategy['name']}", phase="START")
            self.debug.debug(
                f"composite_score гарантирован: {strategy['composite']:.3f}", indent=1)
            self.debug.success(
                f"Успешно | Качество: {strategy['composite']:.3f} | Время: {strategy['time']:.3f}с",
                indent=1
            )

        self.debug.info("")
        self.debug.info("🎯 StrategyEvaluator: Коррекция оценок")
        self.debug.debug("[Прагматичная оценка AdaptiveOtsuHybrid]:", indent=1)
        self.debug.debug(
            "composite=0.727, time_penalty=0.950, final=0.690", indent=2)

        self.debug.info("")
        self.debug.info("🎯 StageRunner: Сводка этапа")
        self.debug.subheader("Сводка этапа: BINARIZATION", indent=1)
        self.debug.strategy_table_start(
            "BINARIZATION", len(strategies_data), indent=1)
        for strategy in strategies_data:
            final_score = strategy['composite'] * 0.95  # имитация коррекции
            self.debug.strategy_table_row(
                strategy['name'], final_score, strategy['time'], strategy['passed'], indent=1
            )
        self.debug.strategy_table_end(indent=1)
        self.debug.debug(
            "Лучшая стратегия: AdaptiveOtsuHybrid (оценка: 0.690)", indent=1)
        self.debug.metrics_table(
            "Метрики лучшей стратегии",
            {"composite_score": 0.727, "components": 142, "uniformity": 0.854},
            indent=2
        )
        self.debug.success("Этап BINARIZATION завершен", indent=1)

        # Сценарий 2: Обработка с ошибками
        self.debug.header("СЦЕНАРИЙ: ОБРАБОТКА С ОШИБКАМИ")

        self.debug.error("Обнаружены проблемы при обработке")
        self.debug.warn("Низкая корреляция (0.45)")
        problem_metrics = {
            "composite_score": 0.45,  # ← ГАРАНТИРОВАННО
            "correlation": 0.45,
            "rotation": 2.1
        }
        self.debug.metrics_table("Проблемные метрики", problem_metrics)

    def test_performance_and_edge_cases(self):
        """3.2. Тестирование производительности и граничных случаев"""
        self.debug.header("3.2. ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ")

        # Тест большого количества сообщений
        start_time = time.time()
        for i in range(10):
            self.debug.info(f"Тестовое сообщение {i+1}/10")
        duration = time.time() - start_time

        self.debug.info(f"Время обработки 10 сообщений: {duration:.3f} сек")

        # Тест длинных сообщений
        long_text = "Очень длинное сообщение " * 10
        self.debug.info(f"Длинное сообщение: {long_text[:50]}...")

        # Тест специальных символов
        self.debug.info("Сообщение со спецсимволами: │ ├─ └─ ═ ║ ╔ ╗ ╚ ╝")

    def test_debug_mode_behavior(self):
        """3.3. Тестирование поведения в разных режимах"""
        self.debug.header("3.3. ПОВЕДЕНИЕ В РАЗНЫХ РЕЖИМАХ")

        self.debug.info("Создание форматтера с debug_mode=False:")
        debug_disabled = DebugFormatter(debug_mode=False, module_name=__name__)

        self.debug.info("Сообщение с debug_mode=True (должно отобразиться):")
        self.debug.debug("Это отладочное сообщение")

        self.debug.info(
            "Сообщение с debug_mode=False (не должно отобразиться):")
        debug_disabled.debug("Это сообщение не должно быть видно")

    def test_iteration_table(self):
        """3.4. Тестирование новой таблицы итераций"""
        self.debug.header("3.4. ТАБЛИЦА ИТЕРАЦИЙ DEBUGFORMATTER")

        # Тестовые данные для итераций
        iterations_data = [
            {"iteration": 1, "composite_score": 0.45, "time_s": 0.1,
                "status": "running", "is_best": False},
            {"iteration": 2, "composite_score": 0.67, "time_s": 0.2,
                "status": "running", "is_best": False},
            {"iteration": 3, "composite_score": 0.89, "time_s": 0.3,
                "status": "completed", "is_best": True},
            {"iteration": 4, "composite_score": 0.82, "time_s": 0.25,
                "status": "completed", "is_best": False},
        ]

        self.debug.info("Компактная таблица итераций:")
        self.debug.iteration_table(
            "Результаты оптимизации",
            iterations_data,
            key_columns=["iteration", "composite_score", "time_s", "status"],
            indent=1
        )

        self.debug.info("Автоматический выбор колонок:")
        self.debug.iteration_table(
            "Автоматические колонки", iterations_data, indent=1)

    def run_all_tests(self):
        """Запуск всех тестов последовательно."""
        print("🚀 ЗАПУСК ТЕСТИРОВАНИЯ DEBUGFORMATTER\n")
        print("📚 Этот файл служит справочником по использованию DebugFormatter\n")

        # Обновленный список тестов
        tests = [
            self.show_architecture_overview,
            self.test_debug_basic_logging,
            self.test_debug_structural_elements,
            self.test_debug_data_display,
            self.test_debug_strategy_table,
            self.test_debug_specialized,
            self.test_debug_context_managers,
            self.test_formatting_rules_demonstration,
            self.test_common_mistakes_demonstration,
            self.test_responsibility_distribution,
            self.test_integration_scenarios,
            self.test_performance_and_edge_cases,
            self.test_debug_mode_behavior,
            self.test_iteration_table,  # ← НОВЫЙ ТЕСТ
        ]

        for i, test in enumerate(tests, 1):
            try:
                test()
                self.debug.success(
                    f"✅ Тест {i}/{len(tests)} пройден: {test.__name__}")
                time.sleep(0.3)  # Пауза для лучшей читаемости
            except Exception as e:
                self.debug.error(f"❌ Тест {i} завершился с ошибкой: {e}")
                raise

        self.debug.header("🎉 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")


def main():
    """Основная функция запуска тестирования."""
    try:
        tester = DebugMessagesTester()
        tester.run_all_tests()

        print("\n" + "="*70)
        print("🎯 ВСЕ ТЕСТЫ УСПЕШНО ЗАВЕРШЕНЫ")
        print("📁 Подробный лог сохранен в: test_debug.log")
        print("👀 Проверьте файл лога для просмотра детального вывода")
        print("📚 Этот файл служит справочником по использованию DebugFormatter")
        print("🎯 Демонстрирует новую архитектуру с гарантированным composite_score")
        print("="*70)

    except Exception as e:
        print(f"\n💥 КРИТИЧЕСКАЯ ОШИБКА ПРИ ЗАПУСКЕ ТЕСТОВ: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
