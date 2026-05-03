"""Neural Networks (Multilayer Perceptron)."""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from app.schemas.results import NormalizedResult, OutputBlock, ChartData, OutputBlockType

def run_neural_network(df: pd.DataFrame, dependent: str, covariates: list[str], factors: list[str] = None, is_categorical: bool = True) -> NormalizedResult:
    """Run Multilayer Perceptron."""
    df_clean = df.dropna(subset=[dependent] + covariates + (factors or [])).copy()
    
    if len(df_clean) == 0:
        raise ValueError("No valid cases found after dropping missing values.")
        
    X = df_clean[covariates].copy()
    
    if factors:
        X = pd.concat([X, pd.get_dummies(df_clean[factors], drop_first=True)], axis=1)
        
    y = df_clean[dependent]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if is_categorical:
        model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        accuracy = model.score(X_test_scaled, y_test)
        
        output_blocks = [
            OutputBlock(
                block_type=OutputBlockType.TABLE,
                title="Model Summary",
                content={
                    "columns": ["Metric", "Value"],
                    "rows": [
                        {"Metric": "Network Structure", "Value": "1 Hidden Layer (10 units)"},
                        {"Metric": "Training Samples", "Value": str(len(X_train))},
                        {"Metric": "Testing Samples", "Value": str(len(X_test))},
                        {"Metric": "Test Accuracy", "Value": f"{accuracy:.3f}"}
                    ],
                    "footnotes": ["Multilayer Perceptron Classifier"]
                }
            )
        ]
        
    else:
        model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        r2 = model.score(X_test_scaled, y_test)
        
        output_blocks = [
            OutputBlock(
                block_type=OutputBlockType.TABLE,
                title="Model Summary",
                content={
                    "columns": ["Metric", "Value"],
                    "rows": [
                        {"Metric": "Network Structure", "Value": "1 Hidden Layer (10 units)"},
                        {"Metric": "Training Samples", "Value": str(len(X_train))},
                        {"Metric": "Testing Samples", "Value": str(len(X_test))},
                        {"Metric": "Test R-squared", "Value": f"{r2:.3f}"}
                    ],
                    "footnotes": ["Multilayer Perceptron Regressor"]
                }
            )
        ]

    from app.utils.interpretation import generate_interpretation

    res = NormalizedResult(
        analysis_type="neural_network",
        title="Neural Network (Multilayer Perceptron)",
        variables={"dependent": dependent, "covariates": covariates},
        output_blocks=output_blocks,
        metadata={
            "n_train": len(X_train),
            "n_test": len(X_test),
            "is_categorical": is_categorical,
            "accuracy": accuracy if is_categorical else None,
            "r2": r2 if not is_categorical else None,
            "duration_ms": 0,
            "timestamp": pd.Timestamp.utcnow().isoformat()
        }
    )
    res.interpretation = generate_interpretation(res)
    return res
