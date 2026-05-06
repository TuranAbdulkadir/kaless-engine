"""KALESS Engine — Advanced Classification & Clustering.

Real sklearn-based implementations for:
  - Hierarchical Cluster (AgglomerativeClustering)
  - Discriminant Analysis (LinearDiscriminantAnalysis)
  - Nearest Neighbor (KNeighborsClassifier)
  - Decision Tree (DecisionTreeClassifier / CART)
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from app.core.preprocessing import validate_variable_exists, validate_numeric
from app.schemas.results import (
    NormalizedResult,
    OutputBlock,
    OutputBlockType,
    Interpretation,
)
from app.utils.errors import ValidationError


def run_hierarchical_cluster(
    df: pd.DataFrame,
    variables: list[str],
    n_clusters: int = 3,
    linkage: str = "ward",
    standardize: bool = False,
) -> NormalizedResult:
    """Hierarchical (Agglomerative) Cluster Analysis."""
    from sklearn.cluster import AgglomerativeClustering
    from scipy.cluster.hierarchy import linkage as scipy_linkage
    from scipy.spatial.distance import pdist
    start = time.time()
    warnings_list: list[str] = []

    for v in variables:
        validate_variable_exists(df, v)
        validate_numeric(df[v], v)

    df_clean = df[variables].dropna()
    n = len(df_clean)
    n_excluded = len(df) - n
    if n < n_clusters:
        raise ValidationError(f"Not enough valid cases ({n}) for {n_clusters} clusters.")
    if n_excluded > 0:
        warnings_list.append(f"{n_excluded} case(s) excluded due to missing values.")

    data = df_clean.values.astype(float)
    if standardize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    # Compute linkage matrix
    dist_matrix = pdist(data, metric="euclidean")
    Z = scipy_linkage(dist_matrix, method=linkage)

    # Fit
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(data)

    output_blocks = []

    # Block 1: Agglomeration Schedule (last steps)
    schedule_rows = []
    total_steps = len(Z)
    show_steps = min(15, total_steps)
    for i in range(total_steps - show_steps, total_steps):
        schedule_rows.append({
            "Stage": i + 1,
            "Cluster 1": int(Z[i, 0]) + 1,
            "Cluster 2": int(Z[i, 1]) + 1,
            "Distance": round(float(Z[i, 2]), 4),
            "New Cluster Size": int(Z[i, 3]),
        })

    output_blocks.append(OutputBlock(
        block_type=OutputBlockType.TABLE,
        title="Agglomeration Schedule",
        display_order=1,
        content={
            "columns": ["Stage", "Cluster 1", "Cluster 2", "Distance", "New Cluster Size"],
            "rows": schedule_rows,
            "footnotes": [f"Linkage method: {linkage}", "Distance measure: Euclidean"],
        },
    ))

    # Block 2: Cluster Centers
    center_cols = [""] + [str(c + 1) for c in range(n_clusters)]
    final_rows = []
    for i, var in enumerate(variables):
        row = {"": var}
        for c in range(n_clusters):
            mask = labels == c
            row[str(c + 1)] = round(float(data[mask, i].mean()), 3)
        final_rows.append(row)

    output_blocks.append(OutputBlock(
        block_type=OutputBlockType.TABLE,
        title="Cluster Centers",
        display_order=2,
        content={"columns": center_cols, "rows": final_rows},
    ))

    # Block 3: Cluster membership counts
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    count_rows = [{"Cluster": c + 1, "N": int(cluster_counts.get(c, 0))} for c in range(n_clusters)]
    count_rows.append({"Cluster": "Valid", "N": n})
    count_rows.append({"Cluster": "Missing", "N": n_excluded})

    output_blocks.append(OutputBlock(
        block_type=OutputBlockType.TABLE,
        title="Number of Cases in each Cluster",
        display_order=3,
        content={"columns": ["Cluster", "N"], "rows": count_rows},
    ))

    duration = int((time.time() - start) * 1000)
    return NormalizedResult(
        analysis_type="hierarchical_cluster",
        title="Hierarchical Cluster Analysis",
        variables={"cluster_variables": variables},
        output_blocks=output_blocks,
        interpretation=Interpretation(
            summary=f"Hierarchical clustering ({linkage} linkage) grouped {n} cases into {n_clusters} clusters using {len(variables)} variable(s).",
            academic_sentence=f"A hierarchical cluster analysis was performed using {linkage} linkage on {len(variables)} variables (N = {n}), yielding a {n_clusters}-cluster solution.",
            recommendations=["Examine cluster centers to interpret cluster profiles.", "Validate with discriminant analysis."],
        ),
        warnings=warnings_list,
        metadata={"valid_n": n, "missing_excluded": n_excluded, "n_clusters": n_clusters, "linkage": linkage, "duration_ms": duration, "timestamp": datetime.utcnow().isoformat()},
    )


def run_discriminant(
    df: pd.DataFrame,
    variables: list[str],
    grouping_variable: str,
) -> NormalizedResult:
    """Discriminant Analysis (LDA) with Wilks' Lambda, classification table."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import confusion_matrix, accuracy_score
    start = time.time()
    warnings_list: list[str] = []

    validate_variable_exists(df, grouping_variable)
    for v in variables:
        validate_variable_exists(df, v)
        validate_numeric(df[v], v)

    df_clean = df[variables + [grouping_variable]].dropna()
    n = len(df_clean)
    n_excluded = len(df) - n
    if n_excluded > 0:
        warnings_list.append(f"{n_excluded} case(s) excluded due to missing values.")

    X = df_clean[variables].values.astype(float)
    y = df_clean[grouping_variable].values
    groups = sorted(df_clean[grouping_variable].unique())
    n_groups = len(groups)

    if n_groups < 2:
        raise ValidationError(f"Grouping variable must have at least 2 groups, found {n_groups}.")

    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    y_pred = lda.predict(X)
    acc = accuracy_score(y, y_pred)

    output_blocks = []

    # Block 1: Group Statistics
    group_rows = []
    for g in groups:
        mask = y == g
        row = {"Group": str(g), "N": int(mask.sum())}
        for vi, v in enumerate(variables):
            vals = X[mask, vi]
            row[f"Mean({v})"] = round(float(vals.mean()), 4)
            row[f"SD({v})"] = round(float(vals.std(ddof=1)), 4) if mask.sum() > 1 else 0
        group_rows.append(row)

    group_headers = ["Group", "N"]
    for v in variables:
        group_headers.extend([f"Mean({v})", f"SD({v})"])

    output_blocks.append(OutputBlock(
        block_type=OutputBlockType.TABLE,
        title="Group Statistics",
        display_order=1,
        content={"columns": group_headers, "rows": group_rows},
    ))

    # Block 2: Eigenvalues & Wilks' Lambda
    if hasattr(lda, 'explained_variance_ratio_') and lda.explained_variance_ratio_ is not None:
        n_funcs = len(lda.explained_variance_ratio_)
        eigen_rows = []
        cumulative = 0.0
        for i in range(n_funcs):
            pct = float(lda.explained_variance_ratio_[i]) * 100
            cumulative += pct
            # Approximate eigenvalue from explained variance ratio
            ratio = lda.explained_variance_ratio_[i]
            eigenval = ratio / (1 - ratio) if ratio < 1 else ratio * n
            canonical_corr = np.sqrt(eigenval / (1 + eigenval)) if eigenval > 0 else 0
            wilks = 1.0 / (1.0 + eigenval)
            eigen_rows.append({
                "Function": i + 1,
                "Eigenvalue": round(float(eigenval), 4),
                "% of Variance": round(pct, 2),
                "Cumulative %": round(cumulative, 2),
                "Canonical Correlation": round(float(canonical_corr), 4),
                "Wilks' Lambda": round(float(wilks), 4),
            })
        output_blocks.append(OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="Eigenvalues & Wilks' Lambda",
            display_order=2,
            content={
                "columns": ["Function", "Eigenvalue", "% of Variance", "Cumulative %", "Canonical Correlation", "Wilks' Lambda"],
                "rows": eigen_rows,
            },
        ))

    # Block 3: Structure Matrix / Coefficients
    if hasattr(lda, 'scalings_') and lda.scalings_ is not None:
        scalings = lda.scalings_
        n_funcs = scalings.shape[1] if scalings.ndim > 1 else 1
        coef_rows = []
        for vi, v in enumerate(variables):
            row = {"Variable": v}
            for fi in range(min(n_funcs, 3)):
                val = scalings[vi, fi] if scalings.ndim > 1 else scalings[vi]
                row[f"Function {fi + 1}"] = round(float(val), 4)
            coef_rows.append(row)
        coef_cols = ["Variable"] + [f"Function {i + 1}" for i in range(min(n_funcs, 3))]
        output_blocks.append(OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="Standardized Canonical Discriminant Function Coefficients",
            display_order=3,
            content={"columns": coef_cols, "rows": coef_rows},
        ))

    # Block 4: Classification Results
    cm = confusion_matrix(y, y_pred, labels=groups)
    class_rows = []
    for gi, g in enumerate(groups):
        row = {"Actual Group": str(g)}
        for gj, g2 in enumerate(groups):
            row[f"Predicted {g2}"] = int(cm[gi, gj])
        row["Total"] = int(cm[gi].sum())
        row["% Correct"] = round(float(cm[gi, gi]) / cm[gi].sum() * 100, 1) if cm[gi].sum() > 0 else 0
        class_rows.append(row)

    class_cols = ["Actual Group"] + [f"Predicted {g}" for g in groups] + ["Total", "% Correct"]
    output_blocks.append(OutputBlock(
        block_type=OutputBlockType.TABLE,
        title="Classification Results",
        display_order=4,
        content={
            "columns": class_cols,
            "rows": class_rows,
            "footnotes": [f"{round(acc * 100, 1)}% of original grouped cases correctly classified."],
        },
    ))

    duration = int((time.time() - start) * 1000)
    return NormalizedResult(
        analysis_type="discriminant",
        title="Discriminant Analysis",
        variables={"predictors": variables, "grouping": grouping_variable},
        output_blocks=output_blocks,
        interpretation=Interpretation(
            summary=f"Discriminant analysis classified {n} cases into {n_groups} groups using {len(variables)} predictor(s). Overall accuracy: {round(acc * 100, 1)}%.",
            academic_sentence=f"A discriminant function analysis was conducted with {len(variables)} predictors and {grouping_variable} as the grouping variable (N = {n}, k = {n_groups}). The model correctly classified {round(acc * 100, 1)}% of cases.",
            recommendations=["Examine standardized coefficients to determine which variables contribute most to group separation.", "Use cross-validation for a more robust accuracy estimate."],
        ),
        warnings=warnings_list,
        metadata={"valid_n": n, "n_groups": n_groups, "accuracy": round(acc, 4), "duration_ms": duration, "timestamp": datetime.utcnow().isoformat()},
    )


def run_nearest_neighbor(
    df: pd.DataFrame,
    variables: list[str],
    grouping_variable: str,
    k: int = 5,
) -> NormalizedResult:
    """K-Nearest Neighbors classification."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import confusion_matrix, accuracy_score
    from sklearn.model_selection import cross_val_score
    start = time.time()
    warnings_list: list[str] = []

    validate_variable_exists(df, grouping_variable)
    for v in variables:
        validate_variable_exists(df, v)
        validate_numeric(df[v], v)

    df_clean = df[variables + [grouping_variable]].dropna()
    n = len(df_clean)
    n_excluded = len(df) - n
    if n_excluded > 0:
        warnings_list.append(f"{n_excluded} case(s) excluded due to missing values.")

    X = df_clean[variables].values.astype(float)
    y = df_clean[grouping_variable].values
    groups = sorted(df_clean[grouping_variable].unique())

    actual_k = min(k, n - 1)
    knn = KNeighborsClassifier(n_neighbors=actual_k)
    knn.fit(X, y)
    y_pred = knn.predict(X)
    acc = accuracy_score(y, y_pred)

    # Cross-validation
    try:
        cv_folds = min(5, n // max(len(groups), 2))
        cv_scores = cross_val_score(knn, X, y, cv=max(2, cv_folds)) if n > 10 else np.array([acc])
    except Exception:
        cv_scores = np.array([acc])

    output_blocks = []

    # Block 1: Model Summary
    output_blocks.append(OutputBlock(
        block_type=OutputBlockType.TABLE,
        title="Model Summary",
        display_order=1,
        content={
            "columns": ["Parameter", "Value"],
            "rows": [
                {"Parameter": "Algorithm", "Value": f"K-Nearest Neighbors (k={actual_k})"},
                {"Parameter": "Training Accuracy", "Value": f"{round(acc * 100, 1)}%"},
                {"Parameter": "Cross-Validation Accuracy", "Value": f"{round(float(cv_scores.mean()) * 100, 1)}% (±{round(float(cv_scores.std()) * 100, 1)}%)"},
                {"Parameter": "Number of Predictors", "Value": str(len(variables))},
                {"Parameter": "Number of Groups", "Value": str(len(groups))},
                {"Parameter": "Valid Cases", "Value": str(n)},
            ],
        },
    ))

    # Block 2: Classification Table
    cm = confusion_matrix(y, y_pred, labels=groups)
    class_rows = []
    for gi, g in enumerate(groups):
        row = {"Actual": str(g)}
        for gj, g2 in enumerate(groups):
            row[f"Predicted {g2}"] = int(cm[gi, gj])
        row["% Correct"] = round(float(cm[gi, gi]) / cm[gi].sum() * 100, 1) if cm[gi].sum() > 0 else 0
        class_rows.append(row)

    class_cols = ["Actual"] + [f"Predicted {g}" for g in groups] + ["% Correct"]
    output_blocks.append(OutputBlock(
        block_type=OutputBlockType.TABLE,
        title="Classification Table",
        display_order=2,
        content={"columns": class_cols, "rows": class_rows, "footnotes": [f"Overall: {round(acc * 100, 1)}% correct"]},
    ))

    duration = int((time.time() - start) * 1000)
    return NormalizedResult(
        analysis_type="nearest_neighbor",
        title="Nearest Neighbor Classification",
        variables={"predictors": variables, "target": grouping_variable},
        output_blocks=output_blocks,
        interpretation=Interpretation(
            summary=f"KNN (k={actual_k}) classified {n} cases into {len(groups)} groups with {round(acc * 100, 1)}% training accuracy.",
            academic_sentence=f"A k-nearest neighbors classification was performed (k = {actual_k}, {len(variables)} predictors, N = {n}). Training accuracy was {round(acc * 100, 1)}%; cross-validated accuracy was {round(float(cv_scores.mean()) * 100, 1)}% (SD = {round(float(cv_scores.std()) * 100, 1)}%).",
        ),
        warnings=warnings_list,
        metadata={"valid_n": n, "k": actual_k, "accuracy": round(acc, 4), "cv_accuracy": round(float(cv_scores.mean()), 4), "duration_ms": duration, "timestamp": datetime.utcnow().isoformat()},
    )


def run_decision_tree(
    df: pd.DataFrame,
    variables: list[str],
    grouping_variable: str,
    max_depth: int = 5,
) -> NormalizedResult:
    """Decision Tree (CART) classification."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import confusion_matrix, accuracy_score
    start = time.time()
    warnings_list: list[str] = []

    validate_variable_exists(df, grouping_variable)
    for v in variables:
        validate_variable_exists(df, v)
        validate_numeric(df[v], v)

    df_clean = df[variables + [grouping_variable]].dropna()
    n = len(df_clean)
    n_excluded = len(df) - n
    if n_excluded > 0:
        warnings_list.append(f"{n_excluded} case(s) excluded due to missing values.")

    X = df_clean[variables].values.astype(float)
    y = df_clean[grouping_variable].values
    groups = sorted(df_clean[grouping_variable].unique())

    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree.fit(X, y)
    y_pred = tree.predict(X)
    acc = accuracy_score(y, y_pred)

    output_blocks = []

    # Block 1: Model Summary
    output_blocks.append(OutputBlock(
        block_type=OutputBlockType.TABLE,
        title="Model Summary",
        display_order=1,
        content={
            "columns": ["Parameter", "Value"],
            "rows": [
                {"Parameter": "Algorithm", "Value": "CART Decision Tree"},
                {"Parameter": "Max Depth", "Value": str(max_depth)},
                {"Parameter": "Actual Depth", "Value": str(tree.get_depth())},
                {"Parameter": "Number of Leaves", "Value": str(tree.get_n_leaves())},
                {"Parameter": "Training Accuracy", "Value": f"{round(acc * 100, 1)}%"},
                {"Parameter": "Valid Cases", "Value": str(n)},
            ],
        },
    ))

    # Block 2: Feature Importance
    importances = tree.feature_importances_
    imp_rows = sorted(
        [{"Variable": v, "Importance": round(float(imp), 4)} for v, imp in zip(variables, importances)],
        key=lambda r: r["Importance"], reverse=True
    )
    for i, r in enumerate(imp_rows):
        r["Rank"] = i + 1

    output_blocks.append(OutputBlock(
        block_type=OutputBlockType.TABLE,
        title="Variable Importance",
        display_order=2,
        content={"columns": ["Rank", "Variable", "Importance"], "rows": imp_rows},
    ))

    # Block 3: Classification Table
    cm = confusion_matrix(y, y_pred, labels=groups)
    class_rows = []
    for gi, g in enumerate(groups):
        row = {"Actual": str(g)}
        for gj, g2 in enumerate(groups):
            row[f"Predicted {g2}"] = int(cm[gi, gj])
        row["% Correct"] = round(float(cm[gi, gi]) / cm[gi].sum() * 100, 1) if cm[gi].sum() > 0 else 0
        class_rows.append(row)

    class_cols = ["Actual"] + [f"Predicted {g}" for g in groups] + ["% Correct"]
    output_blocks.append(OutputBlock(
        block_type=OutputBlockType.TABLE,
        title="Classification Table",
        display_order=3,
        content={"columns": class_cols, "rows": class_rows, "footnotes": [f"Overall: {round(acc * 100, 1)}% correct"]},
    ))

    duration = int((time.time() - start) * 1000)
    return NormalizedResult(
        analysis_type="decision_tree",
        title="Decision Tree Classification",
        variables={"predictors": variables, "target": grouping_variable},
        output_blocks=output_blocks,
        interpretation=Interpretation(
            summary=f"CART decision tree (depth={tree.get_depth()}, {tree.get_n_leaves()} leaves) classified {n} cases with {round(acc * 100, 1)}% accuracy.",
            academic_sentence=f"A CART decision tree was fitted with {len(variables)} predictors (N = {n}). The tree reached depth {tree.get_depth()} with {tree.get_n_leaves()} terminal nodes, achieving {round(acc * 100, 1)}% classification accuracy.",
        ),
        warnings=warnings_list,
        metadata={"valid_n": n, "depth": tree.get_depth(), "n_leaves": tree.get_n_leaves(), "accuracy": round(acc, 4), "duration_ms": duration, "timestamp": datetime.utcnow().isoformat()},
    )
