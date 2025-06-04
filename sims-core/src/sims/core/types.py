from dataclasses import dataclass
from enum import StrEnum


class SupportedCity(StrEnum):
    """Names of the supported cities."""

    LAGOS_NIGERIA = "lagos_nigeria"
    MEXICO_CITY = "mexico_city"
    PARIS = "paris"
    RIO_DE_JANEIRO = "rio_de_janeiro"
    TOKYO_BAY = "tokyo_bay"


class ObjectiveType(StrEnum):
    """Types of the objectives."""

    MIN_COST = "min_cost"
    MIN_CLOUD_COVER = "min_cloud_cover"
    MAX_RESOLUTION = "max_resolution"
    MIN_INCIDENCE_ANGLE = "min_incidence_angle"


class AlgorithmType(StrEnum):
    """Types of the algorithms."""

    MIXED_INTEGER_LINEAR_PROGRAMMING = "milp"
    CONSTRAINT_PROGRAMMING = "cp"
    PARETO_LOCAL_SEARCH = "pls"
    HYBRID = "hybrid"


class TaskStatus(StrEnum):
    """Status of the task."""

    SUBMITTED = "submitted"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStage(StrEnum):
    """
    Stage of the internal processing of the task.
    """

    VALIDATION = "validation"
    IMAGES_FETCH = "images_fetch"
    COST_ESTIMATION = "cost_estimation"
    IMAGES_PREPROCESSING = "images_preprocessing"
    PROBLEM_FORMULATION = "problem_formulation"
    SOLVING = "solving"
    FINISHED = "finished"
