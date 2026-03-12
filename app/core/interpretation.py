"""KALESS Engine — Interpretation Helpers.

Generates plain-language and APA-style interpretations.
"""

from __future__ import annotations

from app.schemas.results import Interpretation, SignificanceLevel


def format_p(p: float) -> str:
    """Format p-value per APA conventions."""
    if p < 0.001:
        return "p < .001"
    return f"p = .{p:.3f}".split(".")[0] + f".{p:.3f}".split(".")[1]


def significance_word(sig: SignificanceLevel) -> str:
    """Get human-readable significance for interpretation."""
    if sig == SignificanceLevel.SIGNIFICANT:
        return "statistically significant"
    if sig == SignificanceLevel.MARGINAL:
        return "marginally significant"
    return "not statistically significant"


def determine_significance(p: float, alpha: float = 0.05) -> SignificanceLevel:
    """Determine significance level from p-value."""
    if p <= alpha:
        return SignificanceLevel.SIGNIFICANT
    if p <= alpha + 0.05:
        return SignificanceLevel.MARGINAL
    return SignificanceLevel.NOT_SIGNIFICANT


def interpret_ttest(
    test_type: str, stat: float, df: float, p: float, sig: SignificanceLevel,
    effect_name: str, effect_val: float, effect_interp: str,
    var_name: str, group_info: str = "",
) -> Interpretation:
    """Generate interpretation for a t-test."""
    sig_word = significance_word(sig)
    p_fmt = format_p(p)

    summary = (
        f"The {test_type} was {sig_word}, "
        f"t({df:.1f}) = {stat:.3f}, {p_fmt}. "
        f"The effect size ({effect_name} = {effect_val:.3f}) is {effect_interp}."
    )

    apa = (
        f"A {test_type} indicated that {group_info}the score on {var_name} was "
        f"{sig_word}, t({df:.1f}) = {stat:.3f}, {p_fmt}, "
        f"{effect_name} = {effect_val:.3f}."
    )

    recs = []
    if sig == SignificanceLevel.NOT_SIGNIFICANT:
        recs.append("Consider whether the sample size provides sufficient power.")
    if effect_interp in ("negligible", "small"):
        recs.append("The effect is small; practical significance may be limited.")

    return Interpretation(summary=summary, academic_sentence=apa, recommendations=recs)


def interpret_anova(
    f_stat: float, df_between: float, df_within: float, p: float,
    sig: SignificanceLevel, eta_sq: float, eta_interp: str, var_name: str,
) -> Interpretation:
    p_fmt = format_p(p)
    sig_word = significance_word(sig)

    summary = (
        f"The one-way ANOVA was {sig_word}, "
        f"F({df_between:.0f}, {df_within:.0f}) = {f_stat:.3f}, {p_fmt}. "
        f"Effect size (η² = {eta_sq:.4f}) is {eta_interp}."
    )

    apa = (
        f"A one-way ANOVA revealed a {sig_word} effect of the grouping variable "
        f"on {var_name}, F({df_between:.0f}, {df_within:.0f}) = {f_stat:.3f}, "
        f"{p_fmt}, η² = {eta_sq:.4f}."
    )

    recs = []
    if sig == SignificanceLevel.SIGNIFICANT:
        recs.append("Run post-hoc pairwise comparisons to identify which groups differ.")

    return Interpretation(summary=summary, academic_sentence=apa, recommendations=recs)


def interpret_chi_square(
    chi2: float, df: float, p: float, sig: SignificanceLevel,
    cramers_v: float, v_interp: str,
) -> Interpretation:
    p_fmt = format_p(p)
    sig_word = significance_word(sig)

    summary = (
        f"The chi-square test of independence was {sig_word}, "
        f"χ²({df:.0f}) = {chi2:.3f}, {p_fmt}. "
        f"Effect size (Cramér's V = {cramers_v:.4f}) is {v_interp}."
    )

    apa = (
        f"A chi-square test of independence indicated a {sig_word} association, "
        f"χ²({df:.0f}) = {chi2:.3f}, {p_fmt}, V = {cramers_v:.4f}."
    )

    return Interpretation(summary=summary, academic_sentence=apa, recommendations=[])


def interpret_correlation(
    method: str, r: float, p: float, sig: SignificanceLevel,
    r_interp: str, var1: str, var2: str,
) -> Interpretation:
    p_fmt = format_p(p)
    sig_word = significance_word(sig)
    direction = "positive" if r > 0 else "negative" if r < 0 else "no"

    summary = (
        f"The {method} correlation between {var1} and {var2} was "
        f"{sig_word} (r = {r:.3f}, {p_fmt}). "
        f"There is a {direction}, {r_interp} association."
    )

    apa = (
        f"A {method} correlation found a {sig_word} {direction} association "
        f"between {var1} and {var2}, r = {r:.3f}, {p_fmt}."
    )

    return Interpretation(summary=summary, academic_sentence=apa, recommendations=[])


def interpret_regression(
    r2: float, adj_r2: float, f_stat: float, df_model: int, df_resid: int,
    p: float, sig: SignificanceLevel,
) -> Interpretation:
    p_fmt = format_p(p)
    sig_word = significance_word(sig)

    summary = (
        f"The regression model was {sig_word}, "
        f"F({df_model}, {df_resid}) = {f_stat:.3f}, {p_fmt}. "
        f"The model explains {r2 * 100:.1f}% of the variance (R² = {r2:.4f}, "
        f"Adjusted R² = {adj_r2:.4f})."
    )

    apa = (
        f"A {'multiple ' if df_model > 1 else ''}linear regression analysis indicated "
        f"that the model was {sig_word}, F({df_model}, {df_resid}) = {f_stat:.3f}, "
        f"{p_fmt}, R² = {r2:.4f}."
    )

    return Interpretation(summary=summary, academic_sentence=apa, recommendations=[])
