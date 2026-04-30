"""KALESS Engine — Analyze Route (statistical analysis execution)."""

from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import verify_engine_key
from app.analysis.registry import dispatch_analysis, get_available_analyses
from app.core.parser import _read_file
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from app.utils.errors import KalessEngineError
from app.utils.storage import download_file

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(
    request: AnalyzeRequest,
    _key: str = Depends(verify_engine_key),
):
    """Execute a statistical analysis on a dataset.

    Downloads the dataset, loads it into a DataFrame, dispatches
    to the requested analysis module, and returns NormalizedResult.
    """
    # Load into DataFrame
    try:
        warnings: list[str] = []
        if request.raw_data is not None:
            # Load directly from raw JSON payload with bulletproof format handling
            try:
                if isinstance(request.raw_data, list):
                    # List of dicts (records)
                    df = pd.DataFrame(request.raw_data)
                elif isinstance(request.raw_data, dict):
                    # Dict of lists or orient index
                    df = pd.DataFrame.from_dict(request.raw_data, orient='columns')
                else:
                    raise ValueError(f"Unsupported raw_data type: {type(request.raw_data)}")
            except Exception as e:
                print(f"PANDAS CRASH: {e}")
                raise HTTPException(status_code=500, detail=f"Pandas format error: {str(e)}")
        else:
            if not request.dataset_url:
                raise ValueError("Either dataset_url or raw_data must be provided.")
                
            parts = request.dataset_url.split("/", 1)
            if len(parts) != 2:
                raise ValueError("Invalid dataset_url format.")
            bucket, path = parts
            file_content = download_file(bucket, path)
            
            df = _read_file(
                file_content,
                request.file_type,
                request.encoding,
                request.delimiter,
                warnings,
            )
    except Exception as e:
        return AnalyzeResponse(
            success=False,
            error=f"Failed to load dataset: {str(e)}",
            error_code="LOAD_ERROR",
        )

    # ═══════════════════════════════════════════════════════════════════
    # UNIVERSAL SANITIZATION PIPELINE (runs before EVERY analysis)
    # Makes ANY file work regardless of data quality.
    # ═══════════════════════════════════════════════════════════════════
    try:
        # 1. AUTO-HEADER STRIPPING: If first row values match column names, remove it
        if len(df) > 1:
            first_row_vals = [str(v).strip().lower() for v in df.iloc[0].values]
            col_names = [str(c).strip().lower() for c in df.columns]
            match_count = sum(1 for v in first_row_vals if v in col_names)
            if match_count >= len(col_names) * 0.5:
                df = df.iloc[1:].reset_index(drop=True)

        # 2. SMART TYPE COERCION: For each column, try pd.to_numeric(errors='coerce')
        #    This converts "19" -> 19, "65.5" -> 65.5, "Ankara" -> NaN
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                coerced = pd.to_numeric(df[col], errors="coerce")
                non_null_original = df[col].dropna().shape[0]
                non_null_coerced = coerced.dropna().shape[0]
                # Only apply coercion if ≥50% of values survived (it's a numeric column with noise)
                if non_null_original > 0 and (non_null_coerced / non_null_original) >= 0.5:
                    df[col] = coerced

        # 3. DROPNA for analysis variables (if specified in params)
        if request.params and request.params.get("dropna"):
            analysis_vars = []
            if "variables" in request.params:
                analysis_vars.extend(request.params["variables"] if isinstance(request.params["variables"], list) else [request.params["variables"]])
            if "test_variable" in request.params:
                analysis_vars.append(request.params["test_variable"])
            if "grouping_variable" in request.params:
                analysis_vars.append(request.params["grouping_variable"])
            if "dependent" in request.params:
                analysis_vars.append(request.params["dependent"])
            if "factor" in request.params:
                analysis_vars.append(request.params["factor"])
            # Only drop for columns that actually exist
            valid_vars = [v for v in analysis_vars if v in df.columns]
            if valid_vars:
                before_n = len(df)
                df = df.dropna(subset=valid_vars).reset_index(drop=True)
                dropped = before_n - len(df)
                if dropped > 0:
                    print(f"   🧹 Sanitizer: Dropped {dropped} rows with missing values in {valid_vars}")
    except Exception as san_err:
        print(f"   ⚠️ Sanitization warning (non-fatal): {san_err}")

    # Dispatch analysis
    try:
        # APPLY GLOBAL FILTERING (Select Cases)
        if request.params and "filter_condition" in request.params and request.params["filter_condition"]:
            try:
                df = df.query(request.params["filter_condition"]).copy()
            except Exception as e:
                raise ValueError(f"Invalid Select Cases filter condition: {str(e)}")
        
        if request.analysis_type == "frequencies":
            from app.analysis.frequencies import run_frequencies
            if not request.params or "variables" not in request.params:
                raise ValueError("Missing 'variables' in params for frequencies.")
            # Take the first variable from the list
            variable = request.params["variables"][0] if isinstance(request.params["variables"], list) else request.params["variables"]
            result = run_frequencies(df, variable)
        elif request.analysis_type == "independent_t":
            from app.analysis.ttest import calculate_independent_t
            if not request.params or "test_variable" not in request.params or "grouping_variable" not in request.params:
                # Handle frontend format which sends { variables: [test_var, group_var] }
                if "variables" in request.params and len(request.params["variables"]) >= 2:
                    test_var = request.params["variables"][0]
                    group_var = request.params["variables"][1]
                else:
                    raise ValueError("Missing 'test_variable' and 'grouping_variable' in params.")
            else:
                test_var = request.params["test_variable"]
                group_var = request.params["grouping_variable"]
            
            # Use explicitly provided group values from the frontend Define Groups dialog
            if "group1_val" in request.params and "group2_val" in request.params:
                group1_val = request.params["group1_val"]
                group2_val = request.params["group2_val"]
            else:
                # Fallback to distinct values if not explicitly provided
                unique_vals = df[group_var].dropna().unique().tolist()
                if len(unique_vals) < 2:
                    raise ValueError(f"Grouping variable '{group_var}' must have at least 2 distinct values.")
                group1_val, group2_val = unique_vals[0], unique_vals[1]
            
            result = calculate_independent_t(df, test_var, group_var, group1_val, group2_val)
        elif request.analysis_type == "paired_t":
            from app.analysis.ttest import calculate_paired_t
            if not request.params or "variables" not in request.params or len(request.params["variables"]) < 2:
                raise ValueError("Missing 'variables' list with at least 2 variables in params for paired t-test.")
            var1, var2 = request.params["variables"][0], request.params["variables"][1]
            result = calculate_paired_t(df, var1, var2)
        elif request.analysis_type == "correlation":
            from app.analysis.correlation import calculate_correlation
            if not request.params or "variables" not in request.params or len(request.params["variables"]) < 2:
                raise ValueError("Missing 'variables' list with at least 2 variables in params for correlation.")
            result = calculate_correlation(df, request.params["variables"])
        elif request.analysis_type == "reliability":
            from app.analysis.reliability import run_reliability
            if not request.params or "variables" not in request.params or len(request.params["variables"]) < 2:
                raise ValueError("Missing 'variables' list with at least 2 variables in params for reliability.")
            result = run_reliability(
                df,
                request.params["variables"],
                item_deleted=bool(request.params.get("item_deleted", True)),
            )
        elif request.analysis_type == "linear_regression":
            from app.analysis.regression import run_linear_regression
            if not request.params or "dependent" not in request.params or "variables" not in request.params:
                raise ValueError("Missing 'dependent' or 'variables' (independents) in params for linear regression.")
            result = run_linear_regression(df, request.params["dependent"], request.params["variables"])
        elif request.analysis_type == "one_way_anova":
            from app.analysis.anova import run_one_way_anova
            if not request.params or "dependent" not in request.params or "factor" not in request.params:
                # Fallback to variable array like T-Test
                if "variables" in request.params and len(request.params["variables"]) >= 2:
                    dep = request.params["variables"][0]
                    fac = request.params["variables"][1]
                else:
                    raise ValueError("Missing 'dependent' and 'factor' in params for One-Way ANOVA.")
            else:
                dep = request.params["dependent"]
                fac = request.params["factor"]
            result = run_one_way_anova(df, dep, fac)
        elif request.analysis_type == "nonparametric":
            from app.analysis.nonparametric import run_nonparametric
            if not request.params:
                raise ValueError("Missing params for nonparametric test.")
            result = run_nonparametric(df, request.params)
        elif request.analysis_type == "factor_analysis":
            from app.analysis.factor import run_factor_analysis
            if not request.params or "variables" not in request.params or len(request.params["variables"]) < 3:
                raise ValueError("Factor analysis requires at least 3 variables.")
            result = run_factor_analysis(
                df,
                variables=request.params["variables"],
                rotation=request.params.get("rotation", "varimax"),
            )
        elif request.analysis_type == "glm_univariate":
            from app.analysis.glm import run_glm_univariate
            if not request.params or "dependent" not in request.params or "fixed_factors" not in request.params:
                raise ValueError("GLM requires 'dependent' and 'fixed_factors' in params.")
            result = run_glm_univariate(
                df,
                dependent=request.params["dependent"],
                fixed_factors=request.params["fixed_factors"],
                covariates=request.params.get("covariates", []),
            )
        elif request.analysis_type == "kmeans_cluster":
            from app.analysis.classify import run_kmeans_cluster
            if not request.params or "variables" not in request.params or len(request.params["variables"]) < 1:
                raise ValueError("K-Means requires at least 1 variable.")
            result = run_kmeans_cluster(
                df,
                variables=request.params["variables"],
                n_clusters=int(request.params.get("n_clusters", 3)),
                max_iter=int(request.params.get("max_iter", 10)),
                standardize=bool(request.params.get("standardize", False)),
                save_membership=bool(request.params.get("save_membership", True)),
            )
        elif request.analysis_type == "neural_network":
            from app.analysis.neural_net import run_neural_network
            if not request.params or "dependent" not in request.params or "covariates" not in request.params:
                raise ValueError("Neural Network requires 'dependent' and 'covariates'.")
            result = run_neural_network(
                df,
                dependent=request.params["dependent"],
                covariates=request.params["covariates"],
                factors=request.params.get("factors", []),
                is_categorical=bool(request.params.get("is_categorical", True))
            )
        elif request.analysis_type == "forecasting":
            from app.analysis.forecasting import run_forecasting
            if not request.params or "dependent" not in request.params:
                raise ValueError("Forecasting requires 'dependent'.")
            result = run_forecasting(
                df,
                dependent=request.params["dependent"],
                date_var=request.params.get("date_var"),
                steps=int(request.params.get("steps", 10))
            )
        elif request.analysis_type == "survival":
            from app.analysis.survival import run_survival_analysis
            if not request.params or "time_var" not in request.params or "status_var" not in request.params:
                raise ValueError("Survival Analysis requires 'time_var' and 'status_var'.")
            result = run_survival_analysis(
                df,
                time_var=request.params["time_var"],
                status_var=request.params["status_var"],
                factor_var=request.params.get("factor_var")
            )
        elif request.analysis_type == "missing_value":
            from app.analysis.missing_value import run_missing_value_analysis
            result = run_missing_value_analysis(
                df,
                variables=request.params.get("variables", [])
            )
        elif request.analysis_type == "multiple_response":
            from app.analysis.multiple_response import run_multiple_response
            if not request.params or "variables" not in request.params:
                raise ValueError("Multiple Response requires 'variables'.")
            result = run_multiple_response(
                df,
                variables=request.params["variables"],
                count_value=str(request.params.get("count_value", "1"))
            )
        elif request.analysis_type == "direct_marketing":
            from app.analysis.direct_marketing import run_direct_marketing
            if not request.params or "customer_id" not in request.params or "date_var" not in request.params or "monetary_var" not in request.params:
                raise ValueError("Direct Marketing requires 'customer_id', 'date_var', and 'monetary_var'.")
            result = run_direct_marketing(
                df,
                customer_id=request.params["customer_id"],
                date_var=request.params["date_var"],
                monetary_var=request.params["monetary_var"]
            )
        elif request.analysis_type == "mixed_models":
            from app.analysis.mixed_models import run_mixed_model
            if not request.params or "dependent" not in request.params or "fixed_factors" not in request.params or "random_factor" not in request.params:
                raise ValueError("Mixed Models requires 'dependent', 'fixed_factors', and 'random_factor'.")
            result = run_mixed_model(
                df,
                dependent=request.params["dependent"],
                fixed_factors=request.params["fixed_factors"],
                random_factor=request.params["random_factor"]
            )
        else:
            result = dispatch_analysis(
                analysis_type=request.analysis_type,
                df=df,
                params=request.params,
            )

        return AnalyzeResponse(
            success=True,
            result=result.model_dump(),
        )
    except KalessEngineError as e:
        return AnalyzeResponse(
            success=False,
            error=e.message,
            error_code=e.code,
        )
    except Exception as e:
        return AnalyzeResponse(
            success=False,
            error=f"Unexpected analysis error: {str(e)}",
            error_code="INTERNAL_ERROR",
        )


@router.get("/analyses")
async def list_analyses(
    _key: str = Depends(verify_engine_key),
):
    """List available analysis types with their parameters."""
    return {"analyses": get_available_analyses()}
