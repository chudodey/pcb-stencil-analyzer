"""
Единая система структур данных для проекта анализа трафаретов
Архитектура: 3-уровневая система с четким разделением ответственности
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List, Any
from enum import Enum
from datetime import datetime
from pathlib import Path
import numpy as np

# =====================================================================================
# УРОВЕНЬ 1: БАЗОВЫЕ СУЩНОСТИ (ядро системы)
# =====================================================================================


class BoardShape(Enum):
    """Форма печатной платы"""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    SQUARE = "square"


class AlignmentStatus(Enum):
    """Статус совмещения"""
    SUCCESS = "success"
    WARNING = "warning"
    FAILED = "failed"


class PipelineStage(Enum):
    """Этапы пайплайна обработки изображений"""
    PREPROCESSING = "preprocessing"
    BINARIZATION = "binarization"
    ROI_EXTRACTION = "roi_extraction"
    ALIGNMENT = "alignment"


@dataclass
class Operator:
    """Данные об операторе"""
    id: str
    full_name: str
    department: Optional[str] = None


@dataclass
class ApertureMetrics:
    """
    Метрики апертур (отверстий в трафарете)
    """
    count: int                      # Общее количество апертур
    mean_area: float                # Средняя площадь апертур
    std_area: float                 # Стандартное отклонение площади
    min_area: float                 # Минимальная площадь в мм2
    max_area: float                 # Максимальная площадь в мм2
    median_area: float              # Медианная площадь в мм2
    min_circularity: float          # Минимальная округлость (4πS/P²)
    mean_circularity: float         # Средняя округлость
    mean_ellipticity: float         # Средняя эллиптичность (отношение осей)
    aspect_ratios: List[float] = field(
        default_factory=list)       # Соотношения сторон
    mean_aspect_ratio: float = 1.0  # Среднее соотношение сторон


@dataclass
class SpatialMetrics:
    """
    Пространственные метрики между объектами
    """
    min_clearance_global: float                   # Минимальный зазор на всем изображении
    clearance_mean: float                         # Средний зазор между объектами
    clearance_std: float                          # Стандартное отклонение зазоров
    # Процентили зазоров {процентиль: значение}
    clearance_percentiles: Dict[int, float]


@dataclass
class StencilReference:
    """
    Эталонные данные из Gerber-файла
    """
    order_number: str                                 # Номер заказа
    gerber_filename: str                              # Имя Gerber файла
    gerber_path: Path                                 # Путь к Gerber файлу
    # Размер трафарета в мм (ширина, высота)
    stencil_size_mm: Tuple[float, float]
    board_shape: BoardShape                           # Форма платы

    # Метрики эталона
    aperture_metrics: ApertureMetrics    # Метрики апертур в мм²
    spatial_metrics: SpatialMetrics      # Пространственные метрики в мм

    # Геометрические данные
    contours: List[np.ndarray] = field(default_factory=list)  # Контуры апертур
    gerber_image: Optional[np.ndarray] = None         # Изображение эталона
    # Границы в мм (x1,y1,x2,y2)
    gerber_bounds_mm: Optional[Tuple[float, float, float, float]] = None

    def __post_init__(self):
        if isinstance(self.gerber_path, str):
            self.gerber_path = Path(self.gerber_path)


@dataclass
class ScanImage:
    """
    Данные о сканированном изображении
    """
    order_number: str               # Номер заказа
    filename: str                   # Имя файла скана
    scan_path: Path                 # Путь к файлу скана
    # Размер изображения в пикселях (ширина, высота)
    image_size_px: Tuple[int, int]
    dpi: int                        # Разрешение сканирования (точек на дюйм)
    scan_timestamp: datetime = field(
        default_factory=datetime.now)  # Время сканирования

    def __post_init__(self):
        if isinstance(self.scan_path, str):
            self.scan_path = Path(self.scan_path)

# =====================================================================================
# УРОВЕНЬ 2: РЕЗУЛЬТАТЫ ОБРАБОТКИ (вычисляемые данные)
# =====================================================================================


@dataclass
class PreprocessingMetrics:
    """
    Метрики качества предобработки изображения
    """
    contrast_improvement: float = 0.0       # Коэффициент улучшения контраста
    # Коэффициент улучшения соотношения сигнал/шум
    snr_improvement: float = 0.0
    edge_preservation: float = 0.0          # Степень сохранения границ (0-1)
    # Стандартное отклонение исходного изображения
    original_std: float = 0.0
    # Стандартное отклонение обработанного изображения
    processed_std: float = 0.0


@dataclass
class ContourDetectionMetrics:
    """
    Метрики качества выделения контуров
    """
    precision: float = 0.0                  # Точность: TP/(TP+FP)
    recall: float = 0.0                      # Полнота: TP/(TP+FN)
    # F1-мера: 2*precision*recall/(precision+recall)
    f1_score: float = 0.0
    false_positives: int = 0             # Количество ложных срабатываний
    false_negatives: int = 0              # Количество пропущенных объектов
    detection_rate: float = 0             # Доля обнаруженных объектов от ожидаемого
    total_detected: int = 0               # Общее количество обнаруженных контуров
    expected_count: int = 0               # Ожидаемое количество контуров


@dataclass
class ROIMetrics:
    """
    Метрики качества выделения области интереса (ROI)
    """
    boundary_match: float = 0.0             # Совпадение границ (0-1)
    area_coverage: float = 0.0              # Покрытие площади (0-1)
    # Совпадение соотношения сторон (0-1)
    aspect_ratio_match: float = 0.0
    center_alignment: float = 0.0           # Выравнивание центра (0-1)


@dataclass
class AlignmentMetrics:
    """
    Метрики совмещения изображений
    """
    correlation: float = 0.0          # Коэффициент корреляции (0-1)
    rotation_angle: float = 0.0             # Угол поворота в градусах
    shift_x_px: float = 0.0                 # Смещение по X в пикселях
    shift_y_px: float = 0.0                 # Смещение по Y в пикселях
    shift_x_mm: float = 0.0                 # Смещение по X в мм
    shift_y_mm: float = 0.0                 # Смещение по Y в мм
    scale_factor: float = 0.0               # Фактор масштаба
    homography_matrix: Optional[np.ndarray] = None  # Матрица гомографии

    # 🔧 Добавленные optional поля для extra метрик из стратегий (alignment_utils, moment_scale и т.д.)
    iou: Optional[float] = None
    dice_coefficient: Optional[float] = None
    intersection_pixels: Optional[int] = None
    union_pixels: Optional[int] = None
    mean_contour_distance: Optional[float] = None
    ref_contours_count: Optional[int] = None
    aligned_contours_count: Optional[int] = None
    ref_nonzero_pixels: Optional[int] = None
    aligned_nonzero_pixels: Optional[int] = None
    phase_shift_x: Optional[float] = None  # Из phaseCorrelate
    phase_shift_y: Optional[float] = None  # Из phaseCorrelate
    # Если появятся новые — добавьте аналогично

    phase_response: Optional[float] = None  # 🔧 Для phaseCorrelate


@dataclass
class ScanAnalysisResult:
    """
    Результаты анализа скана трафарета
    Содержит как исходные данные, так и результаты всех этапов обработки
    """
    # Исходное сканированное изображение в сыром виде
    original_image: Optional[np.ndarray] = None
    # Предобработанное изображение после коррекций
    processed_image: Optional[np.ndarray] = None
    # Выровненное изображение после совмещения с эталоном
    aligned_image: Optional[np.ndarray] = None

    # Найденные контуры апертур
    contours: List[np.ndarray] = field(
        default_factory=list)

    # Метрики апертур (отверстий)
    aperture_metrics: Optional[ApertureMetrics] = None
    # Пространственные метрики
    spatial_metrics: Optional[SpatialMetrics] = None
    # Метрики предобработки изображения
    preprocessing_metrics: Optional[PreprocessingMetrics] = None
    # Метрики обнаружения контуров
    contour_metrics: Optional[ContourDetectionMetrics] = None
    # Метрики выделения области интереса
    roi_metrics: Optional[ROIMetrics] = None

    # Время проведения анализа
    analysis_timestamp: datetime = field(
        default_factory=datetime.now)


@dataclass
class AlignmentResult:
    """
    Результаты совмещения
    """
    is_aligned: bool       # Успешно ли совмещение
    # Общая оценка совмещения (0-1)
    alignment_score: float
    alignment_metrics: Optional[AlignmentMetrics] = None  # Детальные метрики
    alignment_status: AlignmentStatus = AlignmentStatus.FAILED  # Статус


@dataclass
class ComparisonResult:
    """
    Результаты сравнения скана и эталона
    """
    match_percentage: float = 0.0  # 🔧 Default
    mismatch_areas: List[Tuple[int, int, int, int]] = field(
        default_factory=list)  # 🔧 Default
    missing_apertures: int = 0
    excess_apertures: int = 0
    quality_score: float = 0.0
    aperture_count_diff: int = 0     # 🔧 Добавлено с default
    area_deviation_px: float = 0.0  # 🔧 Добавлено с default
    clearance_deviation_px: float = 0.0  # 🔧 Добавлено с default
    comparison_timestamp: datetime = field(default_factory=datetime.now)

# =====================================================================================
# УРОВЕНЬ 3: АГРЕГАЦИОННЫЕ СУЩНОСТИ (композиция)
# =====================================================================================


@dataclass
class StrategyResult:
    """Результат выполнения стратегии."""
    strategy_name: str
    success: bool
    result_data: Any
    metrics: Dict[str, Any]  # Изменено с float на Any
    processing_time: float
    error_message: Optional[str] = None
    artifacts: Dict[str, np.ndarray] = field(default_factory=dict)
    # 🔧 Добавлено для traceability в StageRunner
    stage: Optional[PipelineStage] = None


@dataclass
class EvaluationResult:
    """Упрощенный результат оценки."""
    strategy_name: str
    quality_score: float
    strategy_result: StrategyResult
    evaluation_time: float

    @staticmethod
    def empty() -> 'EvaluationResult':  # 🔧 Добавлен метод empty() для StageRunner
        """
        Возвращает пустой/failed EvaluationResult для случаев без стратегий.
        """
        empty_strategy_result = StrategyResult(
            strategy_name="none",
            success=False,
            result_data=None,
            metrics={},
            processing_time=0.0,
            error_message="No strategies available"
        )
        return EvaluationResult(
            strategy_name="none",
            quality_score=0.0,
            strategy_result=empty_strategy_result,
            evaluation_time=0.0
        )


@dataclass
class PipelineResult:
    """Упрощенный результат пайплайна."""
    success: bool
    final_result: Any
    stage_results: Dict[PipelineStage, EvaluationResult]
    total_processing_time: float
    strategy_combination: Dict[PipelineStage, str]
    global_validation_passed: bool = False
    error_message: Optional[str] = None
    stage_evaluations: Optional[Dict[PipelineStage,
                                     List[EvaluationResult]]] = None
    intermediate_images: Optional[Dict[str, Any]] = None
    # 🔧 Добавлено для логов из сервисов (e.g., StageRunner)
    service_traces: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Инициализация после создания dataclass."""
        if self.stage_evaluations is None:
            self.stage_evaluations = {}
        if self.intermediate_images is None:
            self.intermediate_images = {}


@dataclass
class ProcessedScan:
    """
    Полные результаты обработки одного скана
    """
    scan_info: ScanImage                    # Информация о скане
    scan_analysis: ScanAnalysisResult       # Результаты анализа скана
    alignment: AlignmentResult              # Результаты совмещения
    comparison: ComparisonResult            # Результаты сравнения
    stencil_reference: StencilReference     # Эталонные данные

    # Дополнительные данные
    processing_errors: List[str] = field(
        default_factory=list)  # Ошибки обработки

    @property
    def correlation(self) -> float:
        """Корреляция для удобства доступа"""
        if self.alignment.alignment_metrics:
            return self.alignment.alignment_metrics.correlation
        return self.alignment.alignment_score


@dataclass
class OrderResult:
    """
    Результаты обработки одного заказа
    """
    order_number: str                                       # Номер заказа
    # Список обработанных сканов
    processed_scans: List[ProcessedScan]

    # Метрики заказа
    # Размер платы (строковое представление)
    board_size: str = ""
    # Количество полигонов в дизайне
    polygon_count: int = 0
    processing_errors: List[str] = field(
        default_factory=list)  # Ошибки обработки заказа
    # "continue", "new_order", "exit"
    next_action: Optional[str] = None

    @property
    def success_count(self) -> int:
        """Количество успешных сканов"""
        return sum(1 for scan in self.processed_scans if scan.alignment.is_aligned)

    @property
    def failure_count(self) -> int:
        """Количество неудачных сканов"""
        return len(self.processed_scans) - self.success_count

    @property
    def overall_quality_score(self) -> float:
        """Общий показатель качества заказа"""
        if not self.processed_scans:
            return 0.0
        scores = [scan.comparison.quality_score for scan in self.processed_scans]
        return sum(scores) / len(scores)


@dataclass
class ProcessingSession:
    """
    Сессия обработки одного или нескольких заказов
    """
    session_id: str                                         # Уникальный ID сессии
    operator: Operator                                      # Данные оператора
    start_time: datetime                                    # Время начала сессии
    # Время окончания сессии
    end_time: Optional[datetime] = None
    orders_processed: Dict[str, OrderResult] = field(
        default_factory=dict)  # Обработанные заказы

    @property
    def duration_seconds(self) -> Optional[float]:
        """Длительность сессии в секундах"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def total_orders(self) -> int:
        """Общее количество обработанных заказов"""
        return len(self.orders_processed)

    @property
    def total_scans(self) -> int:
        """Общее количество сканов"""
        return sum(len(order.processed_scans) for order in self.orders_processed.values())

    @property
    def successful_scans(self) -> int:
        """Количество успешных сканов"""
        return sum(order.success_count for order in self.orders_processed.values())

    @property
    def success_rate(self) -> float:
        """Общий процент успешных обработок"""
        total = self.total_scans
        return (self.successful_scans / total * 100) if total > 0 else 0.0

# =====================================================================================
# УТИЛИТАРНЫЕ ФУНКЦИИ (только простые вычисления)
# =====================================================================================


def calculate_alignment_status(correlation: float,
                               thresholds: Dict[str, float]) -> Tuple[AlignmentStatus, str]:
    """
    Определяет статус совмещения на основе корреляции и пороговых значений из конфига

    Args:
        correlation: Значение корреляции
        thresholds: Словарь с пороговыми значениями {'high': 0.8, 'medium': 0.6}

    Returns:
        Tuple[AlignmentStatus, str]: (статус, текстовое описание)
    """
    high_threshold = thresholds.get('high', 0.8)
    medium_threshold = thresholds.get('medium', 0.6)

    if correlation >= high_threshold:
        return AlignmentStatus.SUCCESS, "УСПЕШНО"
    elif correlation >= medium_threshold:
        return AlignmentStatus.WARNING, "С ПРЕДУПРЕЖДЕНИЕМ"
    else:
        return AlignmentStatus.FAILED, "НЕУДАЧНО"
