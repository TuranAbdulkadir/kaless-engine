"""KALESS Engine — Analysis Registry & Dispatcher.

Maps analysis_type strings to module functions and provides frontend display metadata.
NOTE: Not all theoretical features are yet implemented (e.g., Logistic Regression,
Repeated-Measures ANOVA, Reliability, Factor Analysis). These are marked with
'implemented: False' below to make deviations explicit.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from app.schemas.results import NormalizedResult
from app.utils.errors import ValidationError

# Import all completed analysis modules
from app.analysis.descriptives import run_descriptives, run_frequencies
from app.analysis.ttest import calculate_independent_t, calculate_paired_t, run_one_sample_t_test
from app.analysis.anova import run_one_way_anova
from app.analysis.chi_square import run_chi_square_independence
from app.analysis.correlation import calculate_correlation
from app.analysis.regression import run_linear_regression
from app.analysis.reliability import run_reliability
from app.analysis.chart import run_chart_builder


# Analysis function type
AnalysisFunc = Callable[..., NormalizedResult]

# Registry mapping analysis_type -> metadata + function
ANALYSIS_REGISTRY: dict[str, dict[str, Any]] = {
    # --- DESCRIPTIVES ---
    "descriptives": {
        "display_name": "Descriptive Statistics",
        "category": "Descriptives",
        "func": run_descriptives,
        "required": ["variables"],
        "required_pattern": "1+ numeric variables",
        "optional": {"alpha": 0.05},
        "description": "Descriptive statistics (mean, SD, skewness, kurtosis) for numeric variables.",
        "min_plan": "free",
        "implemented": True,
    },
    "frequencies": {
        "display_name": "Frequencies",
        "category": "Descriptives",
        "func": run_frequencies,
        "required": ["variable"],
        "required_pattern": "1 categorical/discrete variable",
        "optional": {},
        "description": "Frequency distribution for a categorical or discrete variable.",
        "min_plan": "free",
        "implemented": True,
    },

    # --- T-TESTS ---
    "one_sample_t_test": {
        "display_name": "One-Sample t-Test",
        "category": "T-Tests",
        "func": run_one_sample_t_test,
        "required": ["variable"],
        "required_pattern": "1 numeric variable",
        "optional": {"test_value": 0.0, "alpha": 0.05},
        "description": "Test whether a sample mean differs from a known value.",
        "min_plan": "free",
        "implemented": True,
    },
    "independent_t_test": {
        "display_name": "Independent Samples t-Test",
        "category": "T-Tests",
        "func": calculate_independent_t,
        "required": ["dependent", "grouping"],
        "required_pattern": "1 numeric dependent, 1 grouping (2 levels) variable",
        "optional": {"alpha": 0.05},
        "description": "Compare means between two independent groups.",
        "min_plan": "free",
        "implemented": True,
    },
    "paired_t_test": {
        "display_name": "Paired Samples t-Test",
        "category": "T-Tests",
        "func": calculate_paired_t,
        "required": ["variable1", "variable2"],
        "required_pattern": "2 numeric variables (related)",
        "optional": {"alpha": 0.05},
        "description": "Compare means for two related/paired measurements.",
        "min_plan": "free",
        "implemented": True,
    },

    # --- ANOVA ---
    "one_way_anova": {
        "display_name": "One-Way ANOVA",
        "category": "ANOVA",
        "func": run_one_way_anova,
        "required": ["dependent", "grouping"],
        "required_pattern": "1 numeric dependent, 1 grouping (2+ levels) variable",
        "optional": {"alpha": 0.05, "post_hoc": True},
        "description": "Test mean differences across multiple groups.",
        "min_plan": "free",
        "implemented": True,
    },
    "repeated_measures_anova": {
        "display_name": "Repeated-Measures ANOVA",
        "category": "ANOVA",
        "func": None,
        "required": ["variables"],
        "required_pattern": "3+ numeric variables (related)",
        "optional": {"alpha": 0.05},
        "description": "Test mean differences across multiple related measurements. (Not yet implemented)",
        "min_plan": "premium",
        "implemented": False,
    },

    # --- NON-PARAMETRIC / PROPORTIONS ---
    "chi_square_independence": {
        "display_name": "Chi-Square Test of Independence",
        "category": "Non-Parametric",
        "func": run_chi_square_independence,
        "required": ["variable1", "variable2"],
        "required_pattern": "2 categorical variables",
        "optional": {"alpha": 0.05},
        "description": "Test association between two categorical variables.",
        "min_plan": "free",
        "implemented": True,
    },

    # --- CORRELATION ---
    "pearson_correlation": {
        "display_name": "Pearson Correlation",
        "category": "Correlation",
        "func": calculate_correlation,
        "required": ["variable1", "variable2"],
        "required_pattern": "2 numeric variables",
        "optional": {"alpha": 0.05},
        "description": "Pearson product-moment correlation between two numeric variables.",
        "min_plan": "free",
        "implemented": True,
    },
    "spearman_correlation": {
        "display_name": "Spearman Correlation",
        "category": "Correlation",
        "func": calculate_correlation,
        "required": ["variable1", "variable2"],
        "required_pattern": "2 numeric or ordinal variables",
        "optional": {"alpha": 0.05},
        "description": "Spearman rank-order correlation between two variables.",
        "min_plan": "free",
        "implemented": True,
    },
    "reliability": {
        "display_name": "Reliability Analysis",
        "category": "Scale",
        "func": run_reliability,
        "required": ["variables"],
        "required_pattern": "2+ numeric variables",
        "optional": {"model": "alpha"},
        "description": "Internal consistency analysis (Cronbach's Alpha).",
        "min_plan": "premium",
        "implemented": True,
    },
    "correlation_matrix": {
        "display_name": "Correlation Matrix",
        "category": "Correlation",
        "func": None,
        "required": ["variables"],
        "required_pattern": "2+ numeric variables",
        "optional": {"method": "pearson", "alpha": 0.05},
        "description": "Correlation matrix for multiple numeric variables.",
        "min_plan": "free",
        "implemented": True,
    },

    # --- REGRESSION ---
    "linear_regression": {
        "display_name": "Linear Regression (OLS)",
        "category": "Regression",
        "func": run_linear_regression,
        "required": ["dependent", "predictors"],
        "required_pattern": "1 numeric dependent, 1+ numeric predictors",
        "optional": {"alpha": 0.05},
        "description": "Simple or multiple linear regression (OLS). Note: currently relies on scipy/numpy rather than statsmodels.",
        "min_plan": "free",
        "implemented": True,
    },
    "logistic_regression": {
        "display_name": "Logistic Regression",
        "category": "Regression",
        "func": None,
        "required": ["dependent", "predictors"],
        "required_pattern": "1 binary dependent, 1+ numeric/categorical predictors",
        "optional": {"alpha": 0.05},
        "description": "Binary logistic regression. (Not yet implemented)",
        "min_plan": "premium",
        "implemented": False,
    },

    # --- GRAPHS ---
    "chart_builder": {
        "display_name": "Chart Builder",
        "category": "Graphs",
        "func": run_chart_builder,
        "required": ["chart_type", "x_axis"],
        "required_pattern": "Any variables suitable for chosen chart",
        "optional": {"y_axis": None},
        "description": "Generates aggregated data for rendering classic SPSS charts.",
        "min_plan": "free",
        "implemented": True,
    },
    
    # --- ADVANCED ---
    "reliability_analysis": {
        "display_name": "Reliability Analysis (Cronbach's Alpha)",
        "category": "Advanced",
        "func": None,
        "required": ["variables"],
        "required_pattern": "2+ numeric or Likert variables",
        "optional": {},
        "description": "Internal consistency reliability. (Not yet implemented)",
        "min_plan": "premium",
        "implemented": False,
    },
    "factor_analysis": {
        "display_name": "Exploratory Factor Analysis",
        "category": "Advanced",
        "func": None,
        "required": ["variables"],
        "required_pattern": "3+ numeric variables",
        "optional": {"rotation": "varimax"},
        "description": "Identify latent variables. (Not yet implemented)",
        "min_plan": "premium",
        "implemented": False,
    },
}


def get_available_analyses() -> list[dict[str, Any]]:
    """Return metadata about all available analysis types for frontend consumption."""
    return [
        {
            "analysis_type": key,
            "display_name": entry["display_name"],
            "category": entry["category"],
            "description": entry["description"],
            "required_params": entry["required"],
            "required_pattern": entry["required_pattern"],
            "optional_params": list(entry["optional"].keys()),
            "min_plan": entry["min_plan"],
            "implemented": entry["implemented"],
        }
        for key, entry in ANALYSIS_REGISTRY.items()
    ]


def dispatch_analysis(
    analysis_type: str,
    df: pd.DataFrame,
    params: dict[str, Any],
) -> NormalizedResult:
    """Dispatch an analysis request to the correct module.

    Args:
        analysis_type: The registered analysis type string.
        df: The dataset as a DataFrame.
        params: Analysis parameters (variable names, options, etc.)

    Returns:
        NormalizedResult from the analysis module.

    Raises:
        ValidationError on unknown analysis type or missing required params.
    """
    if analysis_type not in ANALYSIS_REGISTRY:
        available = ", ".join(ANALYSIS_REGISTRY.keys())
        raise ValidationError(
            f"Unknown analysis type: '{analysis_type}'. "
            f"Available: {available}"
        )

    entry = ANALYSIS_REGISTRY[analysis_type]
    
    if not entry.get("implemented", False) or entry["func"] is None:
        raise ValidationError(
            f"Analysis type '{analysis_type}' is on the product roadmap but not yet implemented."
        )

    func = entry["func"]
    required = entry["required"]
    optional = entry["optional"]

    # Check required params
    missing = [p for p in required if p not in params]
    if missing:
        raise ValidationError(
            f"Missing required parameters for '{analysis_type}': {missing}"
        )

    # Build kwargs: df + required + optional (with defaults)
    kwargs: dict[str, Any] = {"df": df}
    for p in required:
        kwargs[p] = params[p]
    for p, default in optional.items():
        kwargs[p] = params.get(p, default)

    return func(**kwargs)
