"""KALESS Engine — Normalized Result Schema.

Every statistical analysis must return data conforming to these models.
This ensures the frontend can render any analysis type uniformly.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SignificanceLevel(str, Enum):
    SIGNIFICANT = "significant"
    NOT_SIGNIFICANT = "not_significant"
    MARGINAL = "marginal"


class MeasureLevel(str, Enum):
    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    SCALE = "scale"


class OutputBlockType(str, Enum):
    TABLE = "table"
    CHART = "chart"
    TEXT = "text"
    ASSUMPTION = "assumption"
    INTERPRETATION = "interpretation"


# --- Assumption Results ---


class AssumptionResult(BaseModel):
    """Result of a single assumption check."""

    test_name: str
    description: str
    statistic: float | None = None
    p_value: float | None = None
    passed: bool
    note: str = ""


class AssumptionsBlock(BaseModel):
    """Collection of assumption check results for an analysis."""

    checks: list[AssumptionResult] = []
    overall_passed: bool = True
    warnings: list[str] = []


# --- Descriptive Statistics ---


class GroupDescriptive(BaseModel):
    """Descriptive statistics for a group/sample."""

    name: str
    n: int
    mean: float | None = None
    median: float | None = None
    sd: float | None = None
    se: float | None = None
    min: float | None = None
    max: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None


# --- Effect Size ---


class EffectSize(BaseModel):
    """Standardized effect size measure."""

    name: str  # e.g., "Cohen's d", "eta-squared", "Cramér's V"
    value: float
    interpretation: str  # "small", "medium", "large"
    ci_lower: float | None = None
    ci_upper: float | None = None


# --- Confidence Interval ---


class ConfidenceInterval(BaseModel):
    """Confidence interval for a parameter."""

    lower: float
    upper: float
    level: float = 0.95


# --- Primary Result ---


class PrimaryResult(BaseModel):
    """The main statistical test value (e.g., t, F, Chi-Square)."""
    statistic_name: str
    statistic_value: float
    df: float
    df2: Optional[float] = None # Added for ANOVA (df_within)
    p_value: float
    p_value_formatted: str  # "p = .017" or "p < .001"
    significance: SignificanceLevel
    effect_size: Optional[float] = None
    effect_size_type: Optional[str] = None
    alpha: float = 0.05


# --- Coefficient Table (for regression) ---


class CoefficientRow(BaseModel):
    """A single row in a regression coefficient table."""

    name: str
    b: float
    se: float
    beta: float | None = None  # Standardized
    statistic: float  # t or Wald
    p_value: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    vif: float | None = None  # For multicollinearity


# --- Factor Loading (for factor analysis) ---


class FactorLoading(BaseModel):
    """Factor loading for a single variable."""

    variable: str
    loadings: dict[str, float]  # factor_name -> loading value
    communality: float | None = None


# --- Post-Hoc Result ---


class PostHocPair(BaseModel):
    """Result of a pairwise post-hoc comparison."""

    group1: str
    group2: str
    mean_diff: float
    se: float | None = None
    statistic: float | None = None
    p_value: float
    p_adjusted: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    significant: bool


# --- Interpretation ---


class Interpretation(BaseModel):
    """Human-readable interpretation of results in multiple languages."""

    summary_en: str
    summary_tr: str
    academic_sentence_en: str
    academic_sentence_tr: str
    recommendations_en: list[str] = []
    recommendations_tr: list[str] = []


# --- Output Block ---


class OutputBlock(BaseModel):
    """A single renderable block in the output viewer."""

    block_type: OutputBlockType
    title: str
    content: dict[str, Any]  # Flexible content per block type
    display_order: int = 0


# --- Chart Data ---


class ChartData(BaseModel):
    """Chart-ready data for Recharts rendering."""

    chart_type: str  # "histogram", "bar", "scatter", "boxplot", etc.
    data: list[dict[str, Any]]
    config: dict[str, Any] = {}  # Axes labels, colors, options


# --- Top-Level Normalized Result ---


class NormalizedResult(BaseModel):
    """The universal result schema returned by every analysis.

    Frontend renders all analyses using this single structure.
    """

    analysis_type: str
    title: str

    # Variable info
    variables: dict[str, str | list[str]]

    # Assumptions
    assumptions: AssumptionsBlock | None = None

    # Primary result(s)
    primary: PrimaryResult | None = None
    additional_results: list[PrimaryResult] = []

    # Descriptives
    descriptives: list[GroupDescriptive] = []

    # Effect size
    effect_size: EffectSize | None = None

    # Confidence interval
    confidence_interval: ConfidenceInterval | None = None

    # Regression-specific
    coefficients: list[CoefficientRow] = []
    model_summary: dict[str, Any] = {}  # R², adj R², F, etc.

    # Factor analysis-specific
    factor_loadings: list[FactorLoading] = []
    eigenvalues: list[dict[str, Any]] = []
    variance_explained: list[dict[str, Any]] = []

    # Correlation matrix
    correlation_matrix: list[dict[str, Any]] = []

    # Post-hoc
    post_hoc: list[PostHocPair] = []

    # Reliability-specific
    reliability: dict[str, Any] = {}  # alpha, item stats, etc.

    # Frequency table
    frequency_table: list[dict[str, Any]] = []

    # Crosstab
    crosstab: dict[str, Any] = {}

    # Interpretation
    interpretation: Interpretation | None = None

    # Charts
    charts: list[ChartData] = []

    # Output blocks (for structured rendering)
    output_blocks: list[OutputBlock] = []

    # Warnings
    warnings: list[str] = []

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=lambda: {
            "n_total": 0,
            "missing_excluded": 0,
            "library": "",
            "duration_ms": 0,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )
