"""
KALESS — Automated Analysis Test Suite (Fixed)
Verifies mathematical correctness of core statistical modules.
"""

import pytest
import pandas as pd
import numpy as np
from app.analysis.ttest import calculate_independent_t, run_one_sample_t_test
from app.analysis.anova import run_one_way_anova
from app.analysis.regression import run_linear_regression
from app.analysis.reliability import run_reliability

@pytest.fixture
def sample_data():
    """Generates a small controlled dataset for testing."""
    return pd.DataFrame({
        "score": [10.0, 12.0, 11.0, 15.0, 14.0, 18.0, 20.0, 22.0, 21.0, 25.0],
        "group": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2], # Group 1 mean=12.4, Group 2 mean=21.2
        "q1": [5, 4, 5, 4, 5, 5, 4, 5, 4, 5],
        "q2": [5, 4, 5, 4, 5, 5, 4, 5, 4, 5], # Perfect correlation with q1
        "q3": [1, 2, 1, 2, 1, 5, 4, 5, 4, 5]  # Correlated with score
    })

def test_one_sample_t_test(sample_data):
    # Overall mean = 16.8
    result = run_one_sample_t_test(sample_data, "score", test_value=15.0)
    
    assert result.analysis_type == "one_sample_t_test"
    
    # Check mean
    stats_block = next(b for b in result.output_blocks if b.title == "One-Sample Statistics")
    assert stats_block.content["rows"][0]["Mean"] == 16.8
    
    # Check Mean Difference
    test_block = next(b for b in result.output_blocks if b.title == "One-Sample Test")
    assert round(test_block.content["rows"][0]["Mean Difference"], 1) == 1.8

def test_independent_t_test(sample_data):
    result = calculate_independent_t(sample_data, "score", "group", 1, 2)
    
    assert result.analysis_type == "independent_t"
    assert result.metadata["valid_n"] == 10
    
    # Check if we have group statistics
    stats_block = next(b for b in result.output_blocks if "Statistics" in b.title)
    assert len(stats_block.content["rows"]) == 2

def test_one_way_anova(sample_data):
    result = run_one_way_anova(sample_data, "score", "group")
    
    assert result.analysis_type == "one_way_anova"
    anova_block = next(b for b in result.output_blocks if b.title == "ANOVA")
    
    # Sig should be < 0.05
    sig = anova_block.content["rows"][0]["Sig."]
    assert sig < 0.01

def test_linear_regression(sample_data):
    result = run_linear_regression(sample_data, "score", ["q3"])
    
    assert result.analysis_type == "linear_regression"
    summary_block = next(b for b in result.output_blocks if b.title == "Model Summary")
    
    # R Square should be high
    r_square = float(summary_block.content["rows"][0]["R Square"])
    assert r_square > 0.8

def test_reliability_analysis(sample_data):
    result = run_reliability(sample_data, ["q1", "q2"])
    
    assert result.analysis_type == "reliability"
    stats_block = next(b for b in result.output_blocks if b.title == "Reliability Statistics")
    
    alpha = float(stats_block.content["rows"][0]["Cronbach's Alpha"])
    assert alpha == 1.0

def test_validation_errors(sample_data):
    # Test non-existent variable
    # The error raised depends on how validate_variable_exists is implemented
    from app.core.preprocessing import validate_variable_exists
    with pytest.raises(Exception):
        validate_variable_exists(sample_data, "missing_var")
