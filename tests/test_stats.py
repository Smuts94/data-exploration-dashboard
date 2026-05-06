"""
tests/test_stats.py — Unit tests for core/stats.py pure functions.
Run with: python -m pytest tests/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from core.stats import (
    descriptive_stats,
    descriptive_table,
    normality_tests,
    outlier_summary,
    get_iqr_outlier_rows,
    get_zscore_outlier_rows,
    correlation_matrix,
    pvalue_matrix,
    significance_stars,
    annotated_corr_matrix,
    compute_vif,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_series():
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(loc=0, scale=1, size=200), name="x")


@pytest.fixture
def small_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "a": rng.normal(0, 1, 100),
        "b": rng.normal(5, 2, 100),
        "c": rng.normal(-1, 0.5, 100),
    })


# ---------------------------------------------------------------------------
# descriptive_stats
# ---------------------------------------------------------------------------

class TestDescriptiveStats:
    def test_keys_present(self, normal_series):
        result = descriptive_stats(normal_series)
        for key in ["mean", "median", "std", "variance", "min", "max",
                    "skewness", "kurtosis", "IQR", "CV (%)", "Q1", "Q3"]:
            assert key in result, f"Missing key: {key}"

    def test_mean_approx(self, normal_series):
        result = descriptive_stats(normal_series)
        assert abs(result["mean"]) < 0.3  # ~N(0,1) with 200 samples

    def test_iqr_positive(self, normal_series):
        result = descriptive_stats(normal_series)
        assert result["IQR"] > 0

    def test_handles_nans(self):
        s = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        result = descriptive_stats(s)
        assert result["count"] == 4

    def test_zero_mean_cv_nan(self):
        s = pd.Series([0.0, 0.0, 0.0])
        result = descriptive_stats(s)
        assert np.isnan(result["CV (%)"])


# ---------------------------------------------------------------------------
# descriptive_table
# ---------------------------------------------------------------------------

class TestDescriptiveTable:
    def test_returns_dataframe(self, small_df):
        result = descriptive_table(small_df, ["a", "b", "c"])
        assert isinstance(result, pd.DataFrame)
        assert set(result.index) == {"a", "b", "c"}

    def test_empty_cols(self, small_df):
        result = descriptive_table(small_df, [])
        assert result.empty


# ---------------------------------------------------------------------------
# normality_tests
# ---------------------------------------------------------------------------

class TestNormalityTests:
    def test_returns_dataframe(self, normal_series):
        result = normality_tests(normal_series)
        assert isinstance(result, pd.DataFrame)
        assert "Test" in result.columns
        assert "p-value" in result.columns
        assert "Pass (α=0.05)" in result.columns

    def test_shapiro_present_small(self):
        s = pd.Series(np.random.default_rng(1).normal(0, 1, 50))
        result = normality_tests(s)
        shapiro_row = result[result["Test"] == "Shapiro-Wilk"]
        assert len(shapiro_row) == 1
        assert shapiro_row.iloc[0]["Pass (α=0.05)"] != "—"

    def test_shapiro_skipped_large(self):
        s = pd.Series(np.random.default_rng(2).normal(0, 1, 5001))
        result = normality_tests(s)
        shapiro_row = result[result["Test"] == "Shapiro-Wilk"]
        assert len(shapiro_row) == 1
        assert shapiro_row.iloc[0]["Pass (α=0.05)"] == "—"
        assert "5,001" in shapiro_row.iloc[0]["Note"] or "5001" in shapiro_row.iloc[0]["Note"]

    def test_too_few_obs(self):
        s = pd.Series([1.0, 2.0])
        result = normality_tests(s)
        assert "too few" in result.iloc[0]["Note"]

    def test_four_tests_returned_small(self):
        s = pd.Series(np.random.default_rng(3).normal(0, 1, 100))
        result = normality_tests(s)
        assert len(result) == 4  # SW, DP, KS, AD


# ---------------------------------------------------------------------------
# outlier_summary
# ---------------------------------------------------------------------------

class TestOutlierSummary:
    def test_known_outlier_detected(self):
        data = list(np.random.default_rng(5).normal(0, 1, 100)) + [1000.0]
        df = pd.DataFrame({"x": data})
        result = outlier_summary(df, ["x"])
        assert result.iloc[0]["IQR outliers (high)"] >= 1

    def test_no_outliers_clean_data(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = outlier_summary(df, ["x"])
        assert result.iloc[0]["IQR total"] == 0

    def test_columns_present(self, small_df):
        result = outlier_summary(small_df, ["a", "b"])
        for col in ["IQR outliers (low)", "IQR outliers (high)", "Z-score |z|>3"]:
            assert col in result.columns


class TestOutlierRows:
    def test_iqr_returns_outlier(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 1000]})
        flagged = get_iqr_outlier_rows(df, "x")
        assert 1000 in flagged["x"].values

    def test_zscore_returns_outlier(self):
        data = list(np.random.default_rng(7).normal(0, 1, 200)) + [50.0]
        df = pd.DataFrame({"x": data})
        flagged = get_zscore_outlier_rows(df, "x")
        assert 50.0 in flagged["x"].values

    def test_zscore_constant_col(self):
        df = pd.DataFrame({"x": [5.0, 5.0, 5.0, 5.0]})
        flagged = get_zscore_outlier_rows(df, "x")
        assert flagged.empty


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

class TestCorrelation:
    def test_pearson_diagonal_one(self, small_df):
        corr = correlation_matrix(small_df, ["a", "b", "c"], "Pearson")
        np.testing.assert_array_almost_equal(np.diag(corr.values), [1.0, 1.0, 1.0])

    def test_pvalue_diagonal_nan(self, small_df):
        pv = pvalue_matrix(small_df, ["a", "b", "c"], "Pearson")
        assert np.isnan(pv.loc["a", "a"])

    def test_pvalue_symmetric(self, small_df):
        pv = pvalue_matrix(small_df, ["a", "b", "c"], "Pearson")
        assert abs(pv.loc["a", "b"] - pv.loc["b", "a"]) < 1e-10

    def test_spearman(self, small_df):
        corr = correlation_matrix(small_df, ["a", "b"], "Spearman")
        assert corr.shape == (2, 2)

    def test_kendall(self, small_df):
        corr = correlation_matrix(small_df, ["a", "b"], "Kendall")
        assert corr.shape == (2, 2)


class TestSignificanceStars:
    def test_three_stars(self):
        assert significance_stars(0.0001) == "***"

    def test_two_stars(self):
        assert significance_stars(0.005) == "**"

    def test_one_star(self):
        assert significance_stars(0.03) == "*"

    def test_no_stars(self):
        assert significance_stars(0.1) == ""

    def test_nan_returns_empty(self):
        assert significance_stars(float("nan")) == ""


# ---------------------------------------------------------------------------
# VIF
# ---------------------------------------------------------------------------

class TestVIF:
    def test_vif_returns_dataframe(self, small_df):
        result = compute_vif(small_df)
        assert isinstance(result, pd.DataFrame)
        assert "Predictor" in result.columns
        assert "VIF" in result.columns

    def test_vif_values_positive(self, small_df):
        result = compute_vif(small_df)
        assert (result["VIF"].dropna() > 0).all()

    def test_vif_independent_cols_near_one(self):
        rng = np.random.default_rng(99)
        df = pd.DataFrame({
            "x1": rng.normal(0, 1, 500),
            "x2": rng.normal(0, 1, 500),
            "x3": rng.normal(0, 1, 500),
        })
        result = compute_vif(df)
        # Independent columns should have VIF close to 1
        assert (result["VIF"] < 2).all()
