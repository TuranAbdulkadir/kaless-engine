"""KALESS Engine — Effect Size Calculations.

Standard effect size measures for each analysis family.
"""

from __future__ import annotations

import math

import numpy as np

from app.schemas.results import EffectSize


# --- Cohen's d thresholds ---
def _interpret_d(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


# --- Eta-squared thresholds ---
def _interpret_eta_sq(eta: float) -> str:
    if eta < 0.01:
        return "negligible"
    if eta < 0.06:
        return "small"
    if eta < 0.14:
        return "medium"
    return "large"


# --- r / correlation thresholds ---
def _interpret_r(r: float) -> str:
    r = abs(r)
    if r < 0.10:
        return "negligible"
    if r < 0.30:
        return "small"
    if r < 0.50:
        return "medium"
    return "large"


# --- Cramér's V thresholds ---
def _interpret_cramers_v(v: float) -> str:
    if v < 0.10:
        return "negligible"
    if v < 0.30:
        return "small"
    if v < 0.50:
        return "medium"
    return "large"


# --- R² thresholds ---
def _interpret_r_squared(r2: float) -> str:
    if r2 < 0.02:
        return "negligible"
    if r2 < 0.13:
        return "small"
    if r2 < 0.26:
        return "medium"
    return "large"


def cohens_d_one_sample(mean: float, mu0: float, sd: float) -> EffectSize:
    """Cohen's d for one-sample t-test."""
    if sd == 0:
        d = 0.0
    else:
        d = (mean - mu0) / sd
    return EffectSize(name="Cohen's d", value=round(d, 4), interpretation=_interpret_d(d))


def cohens_d_independent(
    mean1: float, mean2: float, sd1: float, sd2: float, n1: int, n2: int
) -> EffectSize:
    """Cohen's d for independent samples (pooled SD)."""
    pooled_sd = math.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        d = 0.0
    else:
        d = (mean1 - mean2) / pooled_sd
    return EffectSize(name="Cohen's d", value=round(d, 4), interpretation=_interpret_d(d))


def cohens_d_paired(mean_diff: float, sd_diff: float) -> EffectSize:
    """Cohen's d for paired samples."""
    if sd_diff == 0:
        d = 0.0
    else:
        d = mean_diff / sd_diff
    return EffectSize(name="Cohen's d", value=round(d, 4), interpretation=_interpret_d(d))


def eta_squared(ss_between: float, ss_total: float) -> EffectSize:
    """Eta-squared for ANOVA."""
    if ss_total == 0:
        eta = 0.0
    else:
        eta = ss_between / ss_total
    return EffectSize(
        name="η² (eta-squared)",
        value=round(eta, 4),
        interpretation=_interpret_eta_sq(eta),
    )


def cramers_v(chi2: float, n: int, min_dim: int) -> EffectSize:
    """Cramér's V for chi-square."""
    if n == 0 or min_dim <= 1:
        v = 0.0
    else:
        v = math.sqrt(chi2 / (n * (min_dim - 1)))
    return EffectSize(
        name="Cramér's V",
        value=round(v, 4),
        interpretation=_interpret_cramers_v(v),
    )


def r_effect_size(r: float) -> EffectSize:
    """Effect size from Pearson/Spearman r."""
    return EffectSize(
        name="r",
        value=round(r, 4),
        interpretation=_interpret_r(r),
    )


def r_squared_effect(r2: float) -> EffectSize:
    """R² as effect size for regression."""
    return EffectSize(
        name="R²",
        value=round(r2, 4),
        interpretation=_interpret_r_squared(r2),
    )
