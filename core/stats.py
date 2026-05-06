"""
core/stats.py — Pure statistical computation functions.
All functions accept plain pandas/numpy inputs and return plain Python/numpy objects.
No Streamlit imports here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.stats import (
    shapiro,
    normaltest,
    kstest,
    anderson,
)
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def descriptive_stats(series: pd.Series) -> dict:
    """Return a dict of descriptive statistics for a numeric series.
    Returns NaN-filled dict when the series is empty after dropping NaNs.
    """
    nan = float("nan")
    s = series.dropna()
    if len(s) == 0:
        return {
            "count": 0,
            "mean": nan, "median": nan, "std": nan, "variance": nan,
            "min": nan, "max": nan, "skewness": nan, "kurtosis": nan,
            "IQR": nan, "CV (%)": nan, "Q1": nan, "Q3": nan,
        }
    q1 = float(np.percentile(s, 25))
    q3 = float(np.percentile(s, 75))
    mean = float(s.mean())
    return {
        "count": int(s.count()),
        "mean": mean,
        "median": float(s.median()),
        "std": float(s.std()),
        "variance": float(s.var()),
        "min": float(s.min()),
        "max": float(s.max()),
        "skewness": float(s.skew()),
        "kurtosis": float(s.kurt()),
        "IQR": q3 - q1,
        "CV (%)": float(s.std() / mean * 100) if mean != 0 else nan,
        "Q1": q1,
        "Q3": q3,
    }


def descriptive_table(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Return a DataFrame of descriptive stats for all numeric columns."""
    rows = []
    for col in numeric_cols:
        row = descriptive_stats(df[col])
        row["column"] = col
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).set_index("column")
    return result


# ---------------------------------------------------------------------------
# Normality tests
# ---------------------------------------------------------------------------

def normality_tests(series: pd.Series) -> pd.DataFrame:
    """
    Run all applicable normality tests on a numeric series.
    Returns a DataFrame with columns: Test, Statistic, p-value, Pass (α=0.05), Note.

    Guards:
    - Shapiro-Wilk is skipped if n >= 5000.
    - All tests skip if n < 3.
    """
    s = series.dropna().astype(float).values
    n = len(s)
    alpha = 0.05
    rows = []

    if n < 3:
        return pd.DataFrame(
            [{"Test": "—", "Statistic": None, "p-value": None,
              "Pass (α=0.05)": None, "Note": f"n={n} — too few observations"}]
        )

    # Shapiro-Wilk
    if n < 5000:
        stat, p = shapiro(s)
        rows.append({
            "Test": "Shapiro-Wilk",
            "Statistic": round(stat, 6),
            "p-value": round(p, 6),
            "Pass (α=0.05)": "✓" if p > alpha else "✗",
            "Note": "",
        })
    else:
        rows.append({
            "Test": "Shapiro-Wilk",
            "Statistic": None,
            "p-value": None,
            "Pass (α=0.05)": "—",
            "Note": f"Skipped — n={n:,} ≥ 5,000",
        })

    # D'Agostino-Pearson K²
    try:
        stat, p = normaltest(s)
        rows.append({
            "Test": "D'Agostino-Pearson K²",
            "Statistic": round(stat, 6),
            "p-value": round(p, 6),
            "Pass (α=0.05)": "✓" if p > alpha else "✗",
            "Note": "",
        })
    except Exception as e:
        rows.append({"Test": "D'Agostino-Pearson K²", "Statistic": None,
                     "p-value": None, "Pass (α=0.05)": "—", "Note": str(e)})

    # Kolmogorov-Smirnov vs. normal
    try:
        mu, sigma = s.mean(), s.std(ddof=1)
        stat, p = kstest(s, "norm", args=(mu, sigma))
        rows.append({
            "Test": "Kolmogorov-Smirnov",
            "Statistic": round(stat, 6),
            "p-value": round(p, 6),
            "Pass (α=0.05)": "✓" if p > alpha else "✗",
            "Note": "vs. fitted normal",
        })
    except Exception as e:
        rows.append({"Test": "Kolmogorov-Smirnov", "Statistic": None,
                     "p-value": None, "Pass (α=0.05)": "—", "Note": str(e)})

    # Anderson-Darling
    try:
        result = anderson(s, dist="norm")
        # Use the 5% critical value (index 2)
        idx = 2  # 5% significance level
        stat_ad = result.statistic
        cv = result.critical_values[idx]
        sl = result.significance_level[idx]
        rows.append({
            "Test": "Anderson-Darling",
            "Statistic": round(stat_ad, 6),
            "p-value": f"CV={cv:.4f} @ {sl}%",
            "Pass (α=0.05)": "✓" if stat_ad < cv else "✗",
            "Note": "compared to critical value",
        })
    except Exception as e:
        rows.append({"Test": "Anderson-Darling", "Statistic": None,
                     "p-value": None, "Pass (α=0.05)": "—", "Note": str(e)})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def outlier_summary(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    Returns a DataFrame summarising outlier counts per numeric column
    using IQR and Z-score methods.
    """
    rows = []
    for col in numeric_cols:
        s = df[col].dropna()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_low = int((s < lower).sum())
        iqr_high = int((s > upper).sum())
        z_scores = np.abs((s - s.mean()) / s.std(ddof=1)) if s.std(ddof=1) > 0 else pd.Series(0, index=s.index)
        z_out = int((z_scores > 3).sum())
        rows.append({
            "Column": col,
            "IQR outliers (low)": iqr_low,
            "IQR outliers (high)": iqr_high,
            "IQR total": iqr_low + iqr_high,
            "Z-score |z|>3": z_out,
            "IQR lower bound": round(lower, 4),
            "IQR upper bound": round(upper, 4),
        })
    return pd.DataFrame(rows)


def get_iqr_outlier_rows(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = df[col].dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
    return df[mask]


def get_zscore_outlier_rows(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = df[col]
    std = s.std(ddof=1)
    if std == 0:
        return df.iloc[0:0]
    z = np.abs((s - s.mean()) / std)
    return df[z > 3]


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def correlation_matrix(df: pd.DataFrame, numeric_cols: list[str], method: str) -> pd.DataFrame:
    return df[numeric_cols].corr(method=method.lower())


def pvalue_matrix(df: pd.DataFrame, numeric_cols: list[str], method: str) -> pd.DataFrame:
    """Compute pairwise correlation p-values for all numeric column pairs."""
    cols = numeric_cols
    n_cols = len(cols)
    pvals = pd.DataFrame(np.ones((n_cols, n_cols)), index=cols, columns=cols)
    m = method.lower()
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            x = df[cols[i]].dropna()
            y = df[cols[j]].dropna()
            common = x.index.intersection(y.index)
            x, y = x.loc[common].values, y.loc[common].values
            if len(x) < 3:
                p = 1.0
            elif m == "pearson":
                _, p = scipy_stats.pearsonr(x, y)
            elif m == "spearman":
                _, p = scipy_stats.spearmanr(x, y)
            elif m == "kendall":
                _, p = scipy_stats.kendalltau(x, y)
            else:
                p = 1.0
            pvals.iloc[i, j] = p
            pvals.iloc[j, i] = p
    for c in cols:
        pvals.loc[c, c] = np.nan
    return pvals


def significance_stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def annotated_corr_matrix(
    corr: pd.DataFrame, pvals: pd.DataFrame
) -> pd.DataFrame:
    """Return a string DataFrame combining correlation value and significance stars."""
    result = corr.copy().astype(object)
    for i in result.index:
        for j in result.columns:
            val = corr.loc[i, j]
            if i == j or np.isnan(val):
                result.loc[i, j] = "—"
            else:
                p = pvals.loc[i, j]
                stars = significance_stars(p)
                result.loc[i, j] = f"{val:.2f}{stars}"
    return result


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------

def compute_vif(X_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for each predictor column.
    X_df should be the design matrix WITHOUT the constant column.
    """
    cols = list(X_df.columns)
    vif_data = []
    arr = X_df.values.astype(float)
    for i, col in enumerate(cols):
        try:
            vif = variance_inflation_factor(arr, i)
        except Exception:
            vif = float("nan")
        vif_data.append({"Predictor": col, "VIF": round(vif, 4)})
    return pd.DataFrame(vif_data)


# ---------------------------------------------------------------------------
# T-Tests
# ---------------------------------------------------------------------------

def cohens_d_two_sample(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two independent samples (pooled SD)."""
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled_std)


def cohens_d_one_sample(s: np.ndarray, mu0: float) -> float:
    """Cohen's d for one-sample t-test."""
    sd = s.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float((s.mean() - mu0) / sd)


def cohens_d_paired(diff: np.ndarray) -> float:
    """Cohen's d for paired t-test (mean diff / SD of diffs)."""
    sd = diff.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(diff.mean() / sd)


def run_independent_ttest(
    df: pd.DataFrame, value_col: str, group_col: str
) -> dict:
    """
    Independent-samples t-test (or Welch's if variances unequal).
    Returns a dict with all relevant statistics.
    """
    groups = [g for g in df[group_col].dropna().unique()]
    if len(groups) != 2:
        raise ValueError(f"Grouping column must have exactly 2 unique values; found {len(groups)}.")
    a = df.loc[df[group_col] == groups[0], value_col].dropna().values.astype(float)
    b = df.loc[df[group_col] == groups[1], value_col].dropna().values.astype(float)

    # Levene's test
    lev_stat, lev_p = scipy_stats.levene(a, b)
    equal_var = lev_p >= 0.05

    t, p = scipy_stats.ttest_ind(a, b, equal_var=equal_var)
    df_val = (len(a) + len(b) - 2) if equal_var else None  # Welch df is fractional
    d = cohens_d_two_sample(a, b)

    # 95% CI on mean difference using scipy
    mean_diff = a.mean() - b.mean()
    if equal_var:
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        df_ci = len(a) + len(b) - 2
    else:
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        # Welch–Satterthwaite df
        df_ci = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2 / (
            (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) +
            (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1)
        )
    t_crit = scipy_stats.t.ppf(0.975, df_ci)
    ci_low, ci_high = mean_diff - t_crit * se, mean_diff + t_crit * se

    # Non-parametric alternative
    u_stat, u_p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")

    # Normality checks
    norm_a = normality_tests(pd.Series(a))
    norm_b = normality_tests(pd.Series(b))

    return {
        "group_labels": [str(groups[0]), str(groups[1])],
        "n": [len(a), len(b)],
        "means": [float(a.mean()), float(b.mean())],
        "sds": [float(a.std(ddof=1)), float(b.std(ddof=1))],
        "test_type": "Student's t" if equal_var else "Welch's t",
        "t_stat": float(t),
        "df": float(df_ci),
        "p_value": float(p),
        "cohens_d": d,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "levene_stat": float(lev_stat),
        "levene_p": float(lev_p),
        "equal_var": equal_var,
        "mwu_stat": float(u_stat),
        "mwu_p": float(u_p),
        "normality_a": norm_a,
        "normality_b": norm_b,
    }


def run_paired_ttest(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """Paired samples t-test."""
    sub = df[[col1, col2]].dropna()
    a = sub[col1].values.astype(float)
    b = sub[col2].values.astype(float)
    diff = a - b

    t, p = scipy_stats.ttest_rel(a, b)
    d = cohens_d_paired(diff)
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    t_crit = scipy_stats.t.ppf(0.975, len(diff) - 1)
    ci_low = diff.mean() - t_crit * se
    ci_high = diff.mean() + t_crit * se

    wil_stat, wil_p = scipy_stats.wilcoxon(a, b)

    return {
        "n": len(diff),
        "mean_diff": float(diff.mean()),
        "sd_diff": float(diff.std(ddof=1)),
        "t_stat": float(t),
        "df": float(len(diff) - 1),
        "p_value": float(p),
        "cohens_d": d,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "wilcoxon_stat": float(wil_stat),
        "wilcoxon_p": float(wil_p),
        "normality": normality_tests(pd.Series(diff)),
    }


def run_onesample_ttest(df: pd.DataFrame, col: str, mu0: float) -> dict:
    """One-sample t-test against hypothesised mean mu0."""
    s = df[col].dropna().values.astype(float)
    t, p = scipy_stats.ttest_1samp(s, mu0)
    d = cohens_d_one_sample(s, mu0)
    se = s.std(ddof=1) / np.sqrt(len(s))
    t_crit = scipy_stats.t.ppf(0.975, len(s) - 1)
    ci_low = s.mean() - t_crit * se
    ci_high = s.mean() + t_crit * se

    return {
        "n": len(s),
        "sample_mean": float(s.mean()),
        "mu0": mu0,
        "t_stat": float(t),
        "df": float(len(s) - 1),
        "p_value": float(p),
        "cohens_d": d,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "normality": normality_tests(pd.Series(s)),
    }


# ---------------------------------------------------------------------------
# ANOVA
# ---------------------------------------------------------------------------

def run_oneway_anova(df: pd.DataFrame, value_col: str, group_col: str) -> dict:
    """
    One-way ANOVA with effect sizes (eta², omega²) and Levene + normality checks.
    Returns dict with anova_table, effect_sizes, group_stats,
    levene result, normality per group, and Kruskal-Wallis result.
    """
    groups_vals = df[group_col].dropna().unique()
    groups = [df.loc[df[group_col] == g, value_col].dropna().values.astype(float)
              for g in groups_vals]

    if len(groups) < 2:
        raise ValueError("Need at least 2 groups for ANOVA.")
    for i, g in enumerate(groups):
        if len(g) < 2:
            raise ValueError(f"Group '{groups_vals[i]}' has fewer than 2 observations.")

    f_stat, p_val = scipy_stats.f_oneway(*groups)

    # SS and effect sizes
    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ss_total = ss_between + ss_within
    n_total = sum(len(g) for g in groups)
    k = len(groups)
    df_between = k - 1
    df_within = n_total - k
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    eta_sq = ss_between / ss_total if ss_total > 0 else float("nan")
    omega_sq = (ss_between - df_between * ms_within) / (ss_total + ms_within) if ss_total > 0 else float("nan")

    anova_table = pd.DataFrame([
        {"Source": "Between groups", "SS": round(ss_between, 6), "df": df_between,
         "MS": round(ms_between, 6), "F": round(f_stat, 6), "p-value": round(p_val, 6)},
        {"Source": "Within groups",  "SS": round(ss_within, 6),  "df": df_within,
         "MS": round(ms_within, 6),  "F": "",               "p-value": ""},
        {"Source": "Total",           "SS": round(ss_total, 6),   "df": n_total - 1,
         "MS": "",                    "F": "",               "p-value": ""},
    ])

    group_stats = pd.DataFrame([{
        "Group": str(groups_vals[i]),
        "n": len(g),
        "Mean": round(g.mean(), 4),
        "SD": round(g.std(ddof=1), 4),
    } for i, g in enumerate(groups)])

    # Levene
    lev_stat, lev_p = scipy_stats.levene(*groups)

    # Normality per group
    normality_per_group = {
        str(groups_vals[i]): normality_tests(pd.Series(g))
        for i, g in enumerate(groups)
    }

    # Kruskal-Wallis
    kw_stat, kw_p = scipy_stats.kruskal(*groups)

    # Tukey HSD (post-hoc) only run on demand — return raw data for page to call
    return {
        "anova_table": anova_table,
        "f_stat": float(f_stat),
        "p_value": float(p_val),
        "eta_sq": float(eta_sq),
        "omega_sq": float(omega_sq),
        "group_stats": group_stats,
        "group_labels": [str(v) for v in groups_vals],
        "groups_data": groups,
        "levene_stat": float(lev_stat),
        "levene_p": float(lev_p),
        "normality_per_group": normality_per_group,
        "kruskal_stat": float(kw_stat),
        "kruskal_p": float(kw_p),
        "n_total": n_total,
        "df_between": df_between,
        "df_within": df_within,
    }


def run_tukey_hsd(df: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
    """Run Tukey HSD post-hoc test. Returns a styled results DataFrame."""
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    sub = df[[value_col, group_col]].dropna()
    result = pairwise_tukeyhsd(sub[value_col], sub[group_col], alpha=0.05)
    df_out = pd.DataFrame(data=result._results_table.data[1:],
                          columns=result._results_table.data[0])
    return df_out


def run_twoway_anova(
    df: pd.DataFrame, value_col: str, factor1: str, factor2: str
) -> dict:
    """
    Two-way ANOVA using statsmodels OLS + anova_lm (Type II SS).
    Returns anova_table dict and group_stats.
    """
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    sub = df[[value_col, factor1, factor2]].dropna().copy()
    # Sanitize column names for formula
    f1 = "F1"
    f2 = "F2"
    dv = "DV"
    sub.columns = [dv, f1, f2]
    formula = f"DV ~ C(F1) + C(F2) + C(F1):C(F2)"
    lm = smf.ols(formula, data=sub).fit()
    table = anova_lm(lm, typ=2)
    table = table.reset_index().rename(columns={"index": "Source"})
    # Restore readable source names
    table["Source"] = table["Source"].str.replace("C(F1)", factor1, regex=False)\
                                      .str.replace("C(F2)", factor2, regex=False)\
                                      .str.replace("C(F1):C(F2)", f"{factor1}:{factor2}", regex=False)
    group_stats = sub.groupby([f1, f2])[dv].agg(["count", "mean", "std"]).reset_index()
    group_stats.columns = [factor1, factor2, "n", "Mean", "SD"]
    return {"anova_table": table, "group_stats": group_stats, "n": len(sub)}


# ---------------------------------------------------------------------------
# Mediation analysis (via Pingouin)
# ---------------------------------------------------------------------------

def run_mediation(
    df: pd.DataFrame,
    x_col: str,
    m_col: str,
    y_col: str,
    covariates: list[str] | None = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Mediation analysis using Pingouin.
    Returns a dict with the path table (DataFrame) and mediation classification.
    Raises ImportError if Pingouin is not installed.
    """
    try:
        import pingouin as pg
    except ImportError:
        raise ImportError(
            "Pingouin is required for mediation analysis. "
            "Install it with: pip install pingouin"
        )

    cols = [x_col, m_col, y_col] + (covariates or [])
    sub = df[cols].dropna()
    if len(sub) < 10:
        raise ValueError("Need at least 10 complete observations for mediation analysis.")

    result = pg.mediation_analysis(
        data=sub,
        x=x_col,
        m=m_col,
        y=y_col,
        covar=covariates if covariates else None,
        n_boot=n_boot,
        seed=seed,
        alpha=0.05,
    )

    # Extract key paths
    paths = {}
    for _, row in result.iterrows():
        paths[row["path"]] = row

    # Indirect effect (ab) CI
    if "ab" in paths:
        ab_row = paths["ab"]
        ci_low = ab_row.get("CI[2.5%]", float("nan"))
        ci_high = ab_row.get("CI[97.5%]", float("nan"))
        indirect_sig = not (
            (isinstance(ci_low, float) and np.isnan(ci_low)) or
            (ci_low <= 0 <= ci_high)
        )
    else:
        ci_low = ci_high = float("nan")
        indirect_sig = False

    # Mediation type classification
    direct_p = float(paths["c'"]["pval"]) if "c'" in paths else 1.0
    if indirect_sig and direct_p >= 0.05:
        med_type = "Full mediation"
    elif indirect_sig and direct_p < 0.05:
        med_type = "Partial mediation"
    else:
        med_type = "No mediation"

    return {
        "path_table": result,
        "paths": paths,
        "indirect_sig": indirect_sig,
        "indirect_ci": (float(ci_low), float(ci_high)),
        "mediation_type": med_type,
        "n": len(sub),
    }


# ---------------------------------------------------------------------------
# Repeated-measures ANOVA
# ---------------------------------------------------------------------------

def run_rm_anova(
    df: pd.DataFrame,
    value_col: str,
    within_col: str,
    subject_col: str,
) -> dict:
    """
    Repeated-measures ANOVA via Pingouin.

    Parameters
    ----------
    df          : DataFrame with at least value_col, within_col, subject_col
    value_col   : Numeric dependent variable column
    within_col  : Categorical within-subject factor column
    subject_col : Column identifying each subject/participant

    Returns dict with:
      anova_table   pd.DataFrame — pingouin rm_anova output
      sphericity    dict         — Mauchly's test result
      post_hoc      pd.DataFrame — pairwise t-tests (paired, FDR-corrected)
      n_subjects    int
      n_levels      int
      warnings      list[str]
    """
    try:
        import pingouin as pg
    except ImportError as exc:
        raise ImportError(
            "Pingouin is required for repeated-measures ANOVA. "
            "Install it with: pip install pingouin>=0.5.4"
        ) from exc

    cols = [subject_col, within_col, value_col]
    sub = df[cols].dropna()
    if sub.empty:
        raise ValueError("No complete observations after dropping NaNs.")

    n_subjects = sub[subject_col].nunique()
    n_levels = sub[within_col].nunique()

    if n_levels < 2:
        raise ValueError("Within-subject factor must have at least 2 levels.")
    if n_subjects < 3:
        raise ValueError("Need at least 3 subjects for repeated-measures ANOVA.")

    warnings: list[str] = []

    # Mauchly's sphericity test (only meaningful with ≥ 3 levels)
    sphericity: dict = {}
    if n_levels >= 3:
        try:
            sph = pg.sphericity(data=sub, dv=value_col, within=within_col, subject=subject_col)
            # pingouin returns a named tuple: (spher, W, chi2, dof, pval)
            sphericity = {
                "sphericity": bool(sph.spher),
                "W": float(sph.W),
                "chi2": float(sph.chi2),
                "dof": int(sph.dof),
                "pval": float(sph.pval),
            }
            if not sph.spher:
                warnings.append(
                    f"Mauchly's test of sphericity violated (W={sph.W:.3f}, "
                    f"p={sph.pval:.4f}). Greenhouse-Geisser correction applied automatically."
                )
        except Exception:
            sphericity = {}

    # Run RM-ANOVA; correction='auto' applies GG when sphericity is violated
    anova_table = pg.rm_anova(
        data=sub,
        dv=value_col,
        within=within_col,
        subject=subject_col,
        correction="auto",
        detailed=True,
    )

    # Post-hoc pairwise paired t-tests with FDR correction
    post_hoc = pg.pairwise_tests(
        data=sub,
        dv=value_col,
        within=within_col,
        subject=subject_col,
        padjust="fdr_bh",
    )

    return {
        "anova_table": anova_table,
        "sphericity": sphericity,
        "post_hoc": post_hoc,
        "n_subjects": n_subjects,
        "n_levels": n_levels,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Multilevel Mediation (2-1-1 model)
# ---------------------------------------------------------------------------

def run_multilevel_mediation(
    df: pd.DataFrame,
    outcome: str,
    x_col: str,
    mediators: list[str],
    cluster_col: str,
    covariates: list[str] | None = None,
    level1_predictors: list[str] | None = None,
    n_boot: int = 500,
    seed: int = 42,
) -> dict:
    """
    Multilevel mediation analysis (2-1-1 model) using mixed-effects models.

    For each mediator M_i:
      Path a_i : M_i ~ X + covariates          (LMM, random intercept by cluster)
      Path b_i : coefficient on M_i in Y model  (from the shared outcome LMM)
      Path c'  : X → Y controlling for all M_i  (from the shared outcome LMM)
      Indirect : a_i * b_i  (bootstrap 95 % CI, cluster-resampling)

    Parameters
    ----------
    outcome             Numeric outcome column (Y)
    x_col               Level-2 predictor (X, e.g. treatment/group)
    mediators           One or more mediator columns (Level-2 aggregates)
    cluster_col         Column identifying clusters (subjects / groups)
    covariates          Level-2 covariate columns (optional)
    level1_predictors   Additional within-cluster predictors of Y (optional)
    n_boot              Number of cluster-bootstrap iterations (default 500)
    seed                Random seed for reproducibility

    Returns
    -------
    dict with keys:
      path_a          dict[med → {coef, se, pval, ci_low, ci_high}]
      path_b          dict[med → {coef, se, pval, ci_low, ci_high}]
      path_c_prime    {coef, se, pval, ci_low, ci_high}
      indirect        dict[med → {point, ci_low, ci_high, sig}]
      total_indirect  float
      total_effect    float
      total_ci        (float, float)
      path_table      pd.DataFrame  (tidy summary)
      n_obs           int
      n_clusters      int
      n_boot_ok       int  (successful bootstrap iterations)
      warnings        list[str]
    """
    import re
    import statsmodels.formula.api as smf

    covariates         = list(covariates or [])
    level1_predictors  = list(level1_predictors or [])
    warnings_out: list[str] = []

    # ── Column set & completeness ────────────────────────────────────────────
    all_needed = (
        [outcome, x_col, cluster_col]
        + mediators
        + covariates
        + level1_predictors
    )
    missing_cols = [c for c in all_needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in data: {missing_cols}")

    sub = df[all_needed].dropna().copy()
    n_obs      = len(sub)
    n_clusters = sub[cluster_col].nunique()

    if n_obs < 10:
        raise ValueError("Need at least 10 complete observations.")
    if n_clusters < 5:
        raise ValueError(
            f"Need at least 5 clusters; found {n_clusters}. "
            "Check the cluster column or apply less restrictive filters."
        )

    # ── Rename columns to safe identifiers for statsmodels formulas ──────────
    all_cols_used = list(dict.fromkeys(all_needed))   # deduplicated, ordered
    safe: dict[str, str] = {}
    for i, c in enumerate(all_cols_used):
        s = re.sub(r"[^a-zA-Z0-9]", "_", c)
        if not s[0].isalpha():
            s = "v" + s
        safe[c] = f"{s}_{i}"

    rev_safe = {v: k for k, v in safe.items()}   # safe → original

    sub_safe = sub.rename(columns=safe)

    s_outcome  = safe[outcome]
    s_x        = safe[x_col]
    s_cluster  = safe[cluster_col]
    s_mediators = [safe[m] for m in mediators]
    s_covars    = [safe[c] for c in covariates]
    s_l1preds   = [safe[c] for c in level1_predictors]

    # ── Helper: fit LMM and extract coef/se/pval for a predictor ────────────
    def _fit_lmm(formula: str, groups, key: str) -> tuple:
        """Returns (coef, se, pval, ci_low, ci_high) for `key` in fitted LMM."""
        m = smf.mixedlm(formula, sub_safe, groups=groups).fit(
            reml=False, disp=False
        )
        coef  = float(m.params[key])
        se    = float(m.bse[key])
        pval  = float(m.pvalues[key])
        ci    = m.conf_int().loc[key]
        return coef, se, pval, float(ci[0]), float(ci[1])

    # ── Path a : M_i ~ X + covariates ───────────────────────────────────────
    rhs_a = " + ".join([s_x] + s_covars) if s_covars else s_x
    path_a: dict[str, dict] = {}
    for med_orig, s_med in zip(mediators, s_mediators):
        try:
            coef, se, pval, ci_lo, ci_hi = _fit_lmm(
                f"{s_med} ~ {rhs_a}",
                sub_safe[s_cluster],
                s_x,
            )
            path_a[med_orig] = dict(coef=coef, se=se, pval=pval,
                                    ci_low=ci_lo, ci_high=ci_hi)
        except Exception as e:
            raise ValueError(f"Path a model for '{med_orig}' failed: {e}") from e

    # ── Path b + c' : Y ~ X + M_1 + ... + M_k + covariates + L1 preds ──────
    rhs_y_parts = [s_x] + s_mediators + s_covars + s_l1preds
    rhs_y = " + ".join(rhs_y_parts)
    try:
        y_model = smf.mixedlm(
            f"{s_outcome} ~ {rhs_y}", sub_safe, groups=sub_safe[s_cluster]
        ).fit(reml=False, disp=False)
    except Exception as e:
        raise ValueError(f"Outcome model failed: {e}") from e

    def _extract(key):
        c   = float(y_model.params[key])
        se  = float(y_model.bse[key])
        p   = float(y_model.pvalues[key])
        ci  = y_model.conf_int().loc[key]
        return dict(coef=c, se=se, pval=p, ci_low=float(ci[0]), ci_high=float(ci[1]))

    path_c_prime = _extract(s_x)
    path_b: dict[str, dict] = {
        med_orig: _extract(s_med)
        for med_orig, s_med in zip(mediators, s_mediators)
    }

    # ── Point-estimate indirect effects ──────────────────────────────────────
    indirect: dict[str, dict] = {}
    for med in mediators:
        ab = path_a[med]["coef"] * path_b[med]["coef"]
        indirect[med] = {"point": ab, "ci_low": float("nan"),
                         "ci_high": float("nan"), "sig": False}

    total_indirect = sum(v["point"] for v in indirect.values())
    total_effect   = path_c_prime["coef"] + total_indirect

    # ── Cluster bootstrap for indirect-effect CIs ────────────────────────────
    rng      = np.random.default_rng(seed)
    clusters = sub_safe[s_cluster].unique()
    n_cls    = len(clusters)

    boot_ab:    dict[str, list[float]] = {m: [] for m in mediators}
    boot_total: list[float] = []
    n_boot_ok  = 0

    if n_clusters < 10:
        warnings_out.append(
            f"Only {n_clusters} clusters — bootstrap CIs may be unstable. "
            "Interpret with caution."
        )

    for _ in range(n_boot):
        samp_cls = rng.choice(clusters, size=n_cls, replace=True)
        chunks   = []
        for new_id, orig_cls in enumerate(samp_cls):
            chunk = sub_safe[sub_safe[s_cluster] == orig_cls].copy()
            chunk[s_cluster] = new_id
            chunks.append(chunk)
        bdf = pd.concat(chunks, ignore_index=True)

        try:
            # Path a per mediator
            b_a: dict[str, float] = {}
            for med_orig, s_med in zip(mediators, s_mediators):
                bm = smf.mixedlm(
                    f"{s_med} ~ {rhs_a}", bdf, groups=bdf[s_cluster]
                ).fit(reml=False, disp=False)
                b_a[med_orig] = float(bm.params[s_x])

            # Outcome model
            bmy = smf.mixedlm(
                f"{s_outcome} ~ {rhs_y}", bdf, groups=bdf[s_cluster]
            ).fit(reml=False, disp=False)

            b_c_prime = float(bmy.params[s_x])
            b_tot_ind = 0.0
            for med_orig, s_med in zip(mediators, s_mediators):
                ab = b_a[med_orig] * float(bmy.params[s_med])
                boot_ab[med_orig].append(ab)
                b_tot_ind += ab
            boot_total.append(b_c_prime + b_tot_ind)
            n_boot_ok += 1
        except Exception:
            continue   # skip failed iterations silently

    # Percentile CIs
    alpha = 0.05
    for med in mediators:
        arr = np.array(boot_ab[med])
        if len(arr) >= 20:
            lo = float(np.percentile(arr, 100 * alpha / 2))
            hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
            indirect[med].update(ci_low=lo, ci_high=hi,
                                 sig=not (lo <= 0 <= hi))

    arr_tot = np.array(boot_total)
    total_ci = (
        (float(np.percentile(arr_tot, 2.5)),
         float(np.percentile(arr_tot, 97.5)))
        if len(arr_tot) >= 20 else (float("nan"), float("nan"))
    )

    if n_boot_ok < n_boot * 0.5:
        warnings_out.append(
            f"Only {n_boot_ok}/{n_boot} bootstrap iterations converged. "
            "CIs may be unreliable — try fewer covariates or more observations."
        )

    # ── Tidy path table ───────────────────────────────────────────────────────
    rows = []
    for med in mediators:
        a = path_a[med]
        rows.append({
            "Path": f"a  ({x_col} → {med})",
            "Coef": round(a["coef"], 4),
            "SE": round(a["se"], 4),
            "p": round(a["pval"], 4),
            "CI low": round(a["ci_low"], 4),
            "CI high": round(a["ci_high"], 4),
        })
    for med in mediators:
        b = path_b[med]
        rows.append({
            "Path": f"b  ({med} → {outcome})",
            "Coef": round(b["coef"], 4),
            "SE": round(b["se"], 4),
            "p": round(b["pval"], 4),
            "CI low": round(b["ci_low"], 4),
            "CI high": round(b["ci_high"], 4),
        })
    cp = path_c_prime
    rows.append({
        "Path": f"c' (direct: {x_col} → {outcome})",
        "Coef": round(cp["coef"], 4),
        "SE": round(cp["se"], 4),
        "p": round(cp["pval"], 4),
        "CI low": round(cp["ci_low"], 4),
        "CI high": round(cp["ci_high"], 4),
    })
    path_table = pd.DataFrame(rows)

    return {
        "path_a":        path_a,
        "path_b":        path_b,
        "path_c_prime":  path_c_prime,
        "indirect":      indirect,
        "total_indirect":total_indirect,
        "total_effect":  total_effect,
        "total_ci":      total_ci,
        "path_table":    path_table,
        "n_obs":         n_obs,
        "n_clusters":    n_clusters,
        "n_boot_ok":     n_boot_ok,
        "warnings":      warnings_out,
        # ── echo back for UI labels ──
        "outcome":       outcome,
        "x_col":         x_col,
        "mediators":     mediators,
        "cluster_col":   cluster_col,
    }
