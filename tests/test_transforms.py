import pytest
import pandas as pd
import numpy as np
from app.transforms import operations

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [20, 25, 30, 35],
        "score": [10.5, 12.0, 15.5, 9.0],
        "category": ["A", "B", "A", "C"],
        "gender": [1, 2, 1, 2]
    })

def test_compute_variable(sample_df):
    df = operations.compute_variable(sample_df, "age_double", "age * 2")
    assert "age_double" in df.columns
    assert df["age_double"].tolist() == [40, 50, 60, 70]
    
    df = operations.compute_variable(df, "total", "age + score")
    assert df["total"].tolist() == [30.5, 37.0, 45.5, 44.0]

def test_z_score(sample_df):
    df = operations.z_score(sample_df, ["score"])
    assert "z_score" in df.columns
    assert np.isclose(df["z_score"].mean(), 0.0, atol=1e-7)
    assert np.isclose(df["z_score"].std(), 1.0, atol=1e-7)

def test_recode(sample_df):
    mapping = {"1": "Male", "2": "Female"}
    df = operations.recode(sample_df, "gender", "gender_str", mapping)
    assert df["gender_str"].tolist() == ["Male", "Female", "Male", "Female"]

def test_reverse_code(sample_df):
    # Reverse 1-5 scale (min=1, max=5) -> formula: 6 - x
    df = operations.reverse_code(sample_df, ["gender"], 1, 5) # Note gender has 1 & 2
    assert "rev_gender" in df.columns
    assert df["rev_gender"].tolist() == [5, 4, 5, 4]

def test_filter_cases(sample_df):
    df = operations.filter_cases(sample_df, "age >= 30")
    assert len(df) == 2
    assert df["age"].tolist() == [30, 35]

def test_sort_cases(sample_df):
    df = operations.sort_cases(sample_df, "score", ascending=False)
    assert df["score"].tolist() == [15.5, 12.0, 10.5, 9.0]
