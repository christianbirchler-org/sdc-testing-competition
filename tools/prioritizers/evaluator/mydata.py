from dataclasses import dataclass
import datetime

@dataclass
class ExperimentalResults:
    tool_name: str
    start_utc: datetime.datetime
    end_utc: datetime.datetime

@dataclass
class EvaluationReport:
    """ This class holds evaluation metrics of a tool."""

    test_suite_cnt: int
    benchmark: str
    time_to_initialize: float
    time_to_prioritize_tests: float
    tool_name: str
    time_to_first_fault: float | None
    time_to_last_fault: float | None
    apfd: float
    apfdc: float


@dataclass
class TreatmentData:
    time_to_prioritize_tests: float
    time_to_first_fault: float | None
    time_to_last_fault: float | None
    apfd: float
    apfdc: float


@dataclass
class TestDetails:
    test_id: str
    hasFailed: bool
    sim_time: float
    road_points: list[tuple[float, float]]


@dataclass
class PGTestData:
    test_case_id: int
    sensodat_collection_id: int
    object_id: str
    has_passed: bool
    has_failed: bool
    risk_factor: float
    oob: float
    max_speed_kmh: int
    is_valid: bool
    sensodat_file_path: str
    duration_seconds: float

@dataclass
class Subject:
    subject_id: int
    pg_test_data: list[PGTestData]

@dataclass
class Sample:
    sample_id: int
    subjects: list[Subject]
