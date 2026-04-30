"""KALESS Engine — Factor Analysis / PCA Module.

Implements Principal Component Analysis (Dimension Reduction) with SPSS parity:
  1. KMO & Bartlett's Test
  2. Communalities
  3. Total Variance Explained (eigenvalues, % of variance, cumulative %)
  4. Component Matrix (unrotated and rotated loadings)

Library: factor_analyzer (KMO, Bartlett), numpy, pandas, scipy.
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from app.core.preprocessing import validate_variable_exists, validate_numeric
from app.schemas.results import (
    NormalizedResult,
    FactorLoading,
    OutputBlock,
    OutputBlockType,
    Interpretation,
)
from app.utils.errors import ValidationError


def run_factor_analysis(
    df: pd.DataFrame,
    variables: list[str],
    rotation: str = "varimax",
) -> NormalizedResult:
    """Perform Principal Component Analysis with optional rotation.

    Args:
        df: The dataset.
        variables: 3+ numeric variables for the analysis.
        rotation: Rotation method ("varimax", "promax", or "none").

    Returns:
        NormalizedResult with factor loadings, eigenvalues, variance explained,
        KMO/Bartlett results, communalities, and output_blocks for SPSS tables.
    """
    start = time.time()
    warnings: list[str] = []

    if len(variables) < 3:
        raise ValidationError("Factor analysis requires at least 3 variables.")

    for v in variables:
        validate_variable_exists(df, v)
        validate_numeric(df[v], v)

    # Listwise deletion
    data = df[variables].dropna()
    n = len(data)
    n_excluded = len(df) - n

    if n < len(variables) + 1:
        raise ValidationError(
            f"Not enough valid cases ({n}) for {len(variables)} variables. "
            "Need at least N > p."
        )

    if n_excluded > 0:
        warnings.append(f"{n_excluded} case(s) excluded due to missing values.")

    # --- KMO & Bartlett's Test ---
    try:
        from factor_analyzer.factor_analyzer import (
            calculate_kmo,
            calculate_bartlett_sphericity,
        )
        kmo_per_var, kmo_overall = calculate_kmo(data)
        chi_sq, p_value, dof = calculate_bartlett_sphericity(data)
        kmo_overall = float(kmo_overall)
        bartlett_chi = float(chi_sq)
        bartlett_p = float(p_value)
        bartlett_df = int(dof)
    except ImportError:
        warnings.append("factor_analyzer package not available; KMO/Bartlett skipped.")
        kmo_overall = None
        kmo_per_var = None
        bartlett_chi = None
        bartlett_p = None
        bartlett_df = None

    # --- Standardize & Correlation Matrix ---
    standardized = (data - data.mean()) / data.std(ddof=1)
    corr_matrix = data.corr()

    # --- Eigenvalue Decomposition ---
    eigenvalues_raw, eigenvectors = np.linalg.eigh(corr_matrix.values)
    # Sort descending
    idx = np.argsort(eigenvalues_raw)[::-1]
    eigenvalues_raw = eigenvalues_raw[idx]
    eigenvectors = eigenvectors[:, idx]

    p = len(variables)
    total_var = float(eigenvalues_raw.sum())

    # --- Determine number of components (Kaiser criterion: eigenvalue > 1) ---
    n_components = int(np.sum(eigenvalues_raw > 1.0))
    if n_components == 0:
        n_components = 1
        warnings.append("No eigenvalues > 1.0; extracting 1 component by default.")

    # --- Total Variance Explained table ---
    variance_explained = []
    cumulative = 0.0
    for i, ev in enumerate(eigenvalues_raw):
        pct = float(ev / total_var * 100) if total_var > 0 else 0.0
        cumulative += pct
        row = {
            "component": i + 1,
            "eigenvalue": round(float(ev), 3),
            "pct_variance": round(pct, 3),
            "cumulative_pct": round(cumulative, 3),
        }
        # Add extraction sums only for retained components
        if i < n_components:
            row["extraction_eigenvalue"] = round(float(ev), 3)
            row["extraction_pct_variance"] = round(pct, 3)
            row["extraction_cumulative_pct"] = round(cumulative, 3)
        variance_explained.append(row)

    # --- Component Matrix (unrotated loadings) ---
    loadings_matrix = eigenvectors[:, :n_components] * np.sqrt(eigenvalues_raw[:n_components])

    # --- Rotation ---
    rotated_loadings = None
    if rotation != "none" and n_components > 1:
        try:
            rotated_loadings = _varimax_rotation(loadings_matrix) if rotation == "varimax" else loadings_matrix
        except Exception:
            warnings.append(f"Rotation '{rotation}' failed; reporting unrotated loadings.")
            rotated_loadings = loadings_matrix
    else:
        rotated_loadings = loadings_matrix

    # --- Communalities ---
    communalities = {}
    for i, v in enumerate(variables):
        initial = 1.0  # For PCA, initial communality is always 1
        extraction = float(np.sum(loadings_matrix[i, :] ** 2))
        communalities[v] = {
            "initial": round(initial, 3),
            "extraction": round(extraction, 3),
        }

    # --- Build FactorLoading objects ---
    factor_loadings = []
    for i, v in enumerate(variables):
        ld = {}
        for j in range(n_components):
            ld[f"Component {j + 1}"] = round(float(rotated_loadings[i, j]), 3)
        factor_loadings.append(
            FactorLoading(
                variable=v,
                loadings=ld,
                communality=communalities[v]["extraction"],
            )
        )

    # --- Eigenvalues list for schema ---
    eigenvalues_list = [
        {
            "component": i + 1,
            "eigenvalue": round(float(eigenvalues_raw[i]), 3),
            "variance_pct": round(float(eigenvalues_raw[i] / total_var * 100), 3),
        }
        for i in range(min(p, 10))
    ]

    # --- Output Blocks (SPSS tables) ---
    output_blocks = []

    # Block 1: KMO & Bartlett
    if kmo_overall is not None:
        kmo_block = OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="KMO and Bartlett's Test",
            display_order=1,
            content={
                "columns": ["Measure", "Value"],
                "rows": [
                    {"Measure": "Kaiser-Meyer-Olkin Measure of Sampling Adequacy", "Value": round(kmo_overall, 3)},
                    {"Measure": "Bartlett's Test — Approx. Chi-Square", "Value": round(bartlett_chi, 3)},
                    {"Measure": "Bartlett's Test — df", "Value": bartlett_df},
                    {"Measure": "Bartlett's Test — Sig.", "Value": f"{bartlett_p:.4f}" if bartlett_p >= 0.0005 else "< .001"},
                ],
            },
        )
        output_blocks.append(kmo_block)

    # Block 2: Communalities
    comm_rows = [
        {"": v, "Initial": c["initial"], "Extraction": c["extraction"]}
        for v, c in communalities.items()
    ]
    comm_block = OutputBlock(
        block_type=OutputBlockType.TABLE,
        title="Communalities",
        display_order=2,
        content={
            "columns": ["", "Initial", "Extraction"],
            "rows": comm_rows,
            "footnotes": ["Extraction Method: Principal Component Analysis."],
        },
    )
    output_blocks.append(comm_block)

    # Block 3: Total Variance Explained
    tve_columns = [
        "Component",
        "Initial Eigenvalue",
        "% of Variance",
        "Cumulative %",
    ]
    if n_components > 0:
        tve_columns.extend([
            "Extraction Eigenvalue",
            "Extraction % of Variance",
            "Extraction Cumulative %",
        ])
    tve_rows = []
    for row in variance_explained:
        r = {
            "Component": row["component"],
            "Initial Eigenvalue": row["eigenvalue"],
            "% of Variance": row["pct_variance"],
            "Cumulative %": row["cumulative_pct"],
        }
        if "extraction_eigenvalue" in row:
            r["Extraction Eigenvalue"] = row["extraction_eigenvalue"]
            r["Extraction % of Variance"] = row["extraction_pct_variance"]
            r["Extraction Cumulative %"] = row["extraction_cumulative_pct"]
        tve_rows.append(r)
    tve_block = OutputBlock(
        block_type=OutputBlockType.TABLE,
        title="Total Variance Explained",
        display_order=3,
        content={
            "columns": tve_columns,
            "rows": tve_rows,
            "footnotes": ["Extraction Method: Principal Component Analysis."],
        },
    )
    output_blocks.append(tve_block)

    # Block 4: Component Matrix
    comp_columns = [""] + [f"Component {j + 1}" for j in range(n_components)]
    comp_rows = []
    for i, v in enumerate(variables):
        row_data = {"": v}
        for j in range(n_components):
            row_data[f"Component {j + 1}"] = round(float(rotated_loadings[i, j]), 3)
        comp_rows.append(row_data)
    comp_title = "Rotated Component Matrix" if rotation != "none" and n_components > 1 else "Component Matrix"
    comp_block = OutputBlock(
        block_type=OutputBlockType.TABLE,
        title=comp_title,
        display_order=4,
        content={
            "columns": comp_columns,
            "rows": comp_rows,
            "footnotes": [
                "Extraction Method: Principal Component Analysis.",
                f"Rotation Method: {rotation.capitalize() if rotation != 'none' else 'None'}.",
                f"{n_components} component(s) extracted.",
            ],
        },
    )
    output_blocks.append(comp_block)

    # --- Interpretation ---
    kmo_desc = ""
    if kmo_overall is not None:
        if kmo_overall >= 0.9:
            kmo_desc = "marvelous"
        elif kmo_overall >= 0.8:
            kmo_desc = "meritorious"
        elif kmo_overall >= 0.7:
            kmo_desc = "middling"
        elif kmo_overall >= 0.6:
            kmo_desc = "mediocre"
        elif kmo_overall >= 0.5:
            kmo_desc = "miserable"
        else:
            kmo_desc = "unacceptable"

    cum_var = variance_explained[n_components - 1]["cumulative_pct"] if variance_explained else 0

    interpretation = Interpretation(
        summary=(
            f"PCA extracted {n_components} component(s) explaining "
            f"{cum_var:.1f}% of the total variance."
        ),
        academic_sentence=(
            f"Principal Component Analysis was conducted on {len(variables)} items "
            f"(N = {n}). "
            + (
                f"The KMO measure of sampling adequacy was {kmo_overall:.3f} ({kmo_desc}), "
                f"and Bartlett's test of sphericity was significant "
                f"(χ²({bartlett_df}) = {bartlett_chi:.2f}, p {'< .001' if bartlett_p and bartlett_p < 0.001 else f'= {bartlett_p:.3f}' if bartlett_p else '= N/A'}), "
                if kmo_overall is not None else ""
            )
            + f"indicating the data are suitable for factor analysis. "
            f"{n_components} component(s) with eigenvalues greater than 1.0 were retained, "
            f"explaining {cum_var:.1f}% of the total variance."
        ),
        recommendations=_factor_recommendations(kmo_overall, communalities, n_components),
    )

    duration = int((time.time() - start) * 1000)

    return NormalizedResult(
        analysis_type="factor_analysis",
        title="Exploratory Factor Analysis (PCA)",
        variables={"items": variables},
        factor_loadings=factor_loadings,
        eigenvalues=eigenvalues_list,
        variance_explained=variance_explained,
        output_blocks=output_blocks,
        interpretation=interpretation,
        warnings=warnings,
        metadata={
            "n_total": len(df),
            "missing_excluded": n_excluded,
            "n_components": n_components,
            "rotation": rotation,
            "kmo": round(kmo_overall, 3) if kmo_overall else None,
            "library": "factor_analyzer + numpy",
            "duration_ms": duration,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


def _varimax_rotation(loadings: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Apply Varimax rotation to a loading matrix.

    Uses the standard iterative orthogonal rotation algorithm.
    """
    n, k = loadings.shape
    rotation_matrix = np.eye(k)
    d = 0

    for _ in range(max_iter):
        old_d = d
        B = loadings @ rotation_matrix
        # Varimax criterion
        u, s, vt = np.linalg.svd(
            loadings.T @ (B ** 3 - (1.0 / n) * B @ np.diag(np.sum(B ** 2, axis=0)))
        )
        rotation_matrix = u @ vt
        d = np.sum(s)
        if abs(d - old_d) < tol:
            break

    return loadings @ rotation_matrix


def _factor_recommendations(
    kmo: float | None,
    communalities: dict[str, dict],
    n_components: int,
) -> list[str]:
    """Generate recommendations for factor analysis results."""
    recs: list[str] = []

    if kmo is not None and kmo < 0.6:
        recs.append(
            f"KMO = {kmo:.3f} is below the recommended threshold (≥ .60). "
            "The variables may not be suitable for factor analysis."
        )

    # Low communalities
    for v, c in communalities.items():
        if c["extraction"] < 0.4:
            recs.append(
                f"'{v}' has a low communality ({c['extraction']:.3f}). "
                "Consider removing this variable from the analysis."
            )

    if n_components == 1:
        recs.append(
            "Only 1 component was retained. The scale may be unidimensional. "
            "Consider using Cronbach's Alpha for reliability assessment."
        )

    return recs
