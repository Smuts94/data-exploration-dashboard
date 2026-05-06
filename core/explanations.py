"""
Educational content module for EDA Dashboard.
Provides glossaries, assumption guides, and interpretation templates.
Helps students understand statistical concepts and results.
"""

import pandas as pd

# ============================================================================
# STATISTICAL GLOSSARY
# ============================================================================

STAT_GLOSSARY = {
    # Descriptive Statistics
    "mean": (
        "**Mean** is the average of all values. It is sensitive to outliers, "
        "so if you have extreme values, the mean may not represent the typical value well."
    ),
    "median": (
        "**Median** is the middle value when data is sorted. It is robust to outliers, "
        "making it a better choice for skewed data."
    ),
    "std": (
        "**Standard Deviation (SD)** measures how spread out the data is. "
        "Larger SD = more variability. About 68% of data falls within 1 SD of the mean."
    ),
    "variance": (
        "**Variance** is SD squared. It's harder to interpret than SD but useful in formulas. "
        "Larger variance = more spread."
    ),
    "iqr": (
        "**Interquartile Range (IQR)** is Q3 - Q1. It contains the middle 50% of data. "
        "Used for outlier detection: values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR] are flagged."
    ),
    "cv": (
        "**Coefficient of Variation (CV)** = SD / mean. It's a standardized measure of spread. "
        "Useful for comparing variability across variables with different scales."
    ),
    "skewness": (
        "**Skewness** measures asymmetry in the distribution.\n"
        "- Skewness ≈ 0: symmetric distribution\n"
        "- Skewness > 1: right-skewed (tail to right)\n"
        "- Skewness < -1: left-skewed (tail to left)\n"
        "- |Skewness| > 1 often suggests non-normal distribution."
    ),
    "kurtosis": (
        "**Kurtosis** measures the 'thickness' of the tails.\n"
        "- Kurtosis ≈ 3: normal distribution (excess = 0)\n"
        "- Kurtosis > 3: heavy tails (leptokurtic)\n"
        "- Kurtosis < 3: light tails (platykurtic)\n"
        "Excess kurtosis > 1 suggests potential outliers."
    ),
    
    # Hypothesis Testing
    "p_value": (
        "**P-value** is the probability of observing your result (or more extreme) "
        "if the null hypothesis were true. "
        "Convention: p < 0.05 → reject null hypothesis (result is 'significant'). "
        "P-values are NOT the probability that your hypothesis is true."
    ),
    "alpha": (
        "**Alpha (α)** is your significance level (usually 0.05). "
        "If p-value < α, you reject the null hypothesis. "
        "α = 0.05 means you accept a 5% risk of a false positive."
    ),
    "effect_size": (
        "**Effect size** measures the practical magnitude of a difference, "
        "independent of sample size. "
        "A large p-value with huge sample size might have tiny effect size (not practically meaningful). "
        "Examples: Cohen's d (t-tests), η² (ANOVA), r (correlation)."
    ),
    
    # Correlation
    "correlation": (
        "**Correlation** measures the strength and direction of a linear relationship.\n"
        "- r = +1: perfect positive relationship\n"
        "- r = 0: no linear relationship\n"
        "- r = -1: perfect negative relationship\n"
        "Correlation does NOT imply causation."
    ),
    "pearson": (
        "**Pearson correlation** measures linear relationships between continuous variables. "
        "Assumes data are approximately normally distributed. "
        "Sensitive to outliers and assumes linearity."
    ),
    "spearman": (
        "**Spearman correlation** is rank-based (doesn't assume normality). "
        "Better for non-normal data or when you suspect non-linear monotonic relationships. "
        "Robust to outliers."
    ),
    "kendall": (
        "**Kendall's τ (Tau)** is another rank-based correlation. "
        "Similar to Spearman but often preferred for smaller samples. "
        "More computationally intensive."
    ),
    
    # Regression
    "r_squared": (
        "**R² (R-squared)** is the coefficient of determination. "
        "It represents the proportion of variance in Y explained by X (0–1). "
        "R² = 0.7 means your model explains 70% of the variation. "
        "Higher is better, but context matters."
    ),
    "adj_r_squared": (
        "**Adjusted R²** penalizes adding more predictors. "
        "It can decrease even if R² increases, helping you avoid overfitting. "
        "Use this to compare models with different numbers of predictors."
    ),
    "f_statistic": (
        "**F-statistic** tests whether your model as a whole is significant. "
        "It compares explained variance to unexplained variance. "
        "Large F and small p-value → model explains significant variation."
    ),
    "aic": (
        "**Akaike Information Criterion (AIC)** balances model fit with complexity. "
        "Lower AIC is better. Use to compare non-nested models. "
        "Penalizes adding predictors more harshly than adjusted R²."
    ),
    "bic": (
        "**Bayesian Information Criterion (BIC)** is similar to AIC. "
        "Lower BIC is better. BIC penalizes complexity more than AIC. "
        "Often used in model selection."
    ),
    "vif": (
        "**Variance Inflation Factor (VIF)** measures multicollinearity. "
        "VIF = 1: no correlation with other predictors\n"
        "VIF > 5: concerning (predictors are correlated)\n"
        "VIF > 10: severe multicollinearity (consider removing predictors)"
    ),
    "durbin_watson": (
        "**Durbin-Watson** tests for autocorrelation in residuals. "
        "Range: 0–4. Value ≈ 2 means no autocorrelation. "
        "< 2: positive autocorrelation; > 2: negative autocorrelation."
    ),
    
    # ANOVA & Post-Hoc
    "eta_squared": (
        "**η² (Eta-squared)** is effect size for ANOVA. "
        "Measures proportion of variance explained by group membership. "
        "Small ≈ 0.01, Medium ≈ 0.06, Large ≈ 0.14."
    ),
    "omega_squared": (
        "**ω² (Omega-squared)** is a less biased effect size than η² for ANOVA. "
        "More accurate with smaller samples."
    ),
    "cohens_d": (
        "**Cohen's d** is effect size for t-tests. "
        "Measures standardized difference between means.\n"
        "Small ≈ 0.2, Medium ≈ 0.5, Large ≈ 0.8."
    ),
    "tukey_hsd": (
        "**Tukey HSD** is post-hoc test for pairwise comparisons after ANOVA. "
        "Compares all pairs of groups. "
        "Only run if omnibus ANOVA is significant (p < 0.05)."
    ),
}


# ============================================================================
# ASSUMPTION EXPLANATIONS
# ============================================================================

ASSUMPTIONS = {
    "normality": {
        "name": "Normality",
        "explanation": (
            "**Normality:** Data should approximately follow a normal (bell-shaped) distribution. "
            "This assumption is important for t-tests, ANOVA, and linear regression."
        ),
        "how_to_check": [
            "Histogram: look for bell-shaped curve",
            "Q-Q plot: points should lie close to the diagonal line",
            "Shapiro-Wilk test: p > 0.05 suggests normality",
            "D'Agostino-Pearson test: p > 0.05 suggests normality",
        ],
        "what_if_violated": (
            "If data is non-normal: (1) Transform data (log, square root), "
            "(2) Use non-parametric alternatives (Mann-Whitney U, Kruskal-Wallis), "
            "or (3) check if violation is severe (robust tests work with moderate violations)."
        ),
    },
    
    "equal_variance": {
        "name": "Homogeneity of Variance",
        "explanation": (
            "**Equal Variance:** The spread of data should be similar across groups. "
            "Important for independent t-tests and ANOVA."
        ),
        "how_to_check": [
            "Levene's test: p > 0.05 suggests equal variance",
            "Visualize: box plots of groups should have similar heights",
            "Check ratio of largest SD to smallest SD (should be < 2)",
        ],
        "what_if_violated": (
            "If variances are unequal: Use Welch's t-test (auto-corrects) or Welch's ANOVA. "
            "These are nearly as powerful as standard tests but don't assume equal variance."
        ),
    },
    
    "independence": {
        "name": "Independence",
        "explanation": (
            "**Independence:** Observations should be independent of each other. "
            "Violating this (repeated measures, clustered data) requires special tests."
        ),
        "how_to_check": [
            "Understand your data collection: are rows independent?",
            "Repeated measures (same subject multiple times)? Use paired tests or mixed models.",
            "Clustered data (students in classrooms)? Use multilevel modeling.",
        ],
        "what_if_violated": (
            "If data are paired: Use paired t-test or repeated-measures ANOVA. "
            "If clustered: Use mixed models or multilevel mediation."
        ),
    },
    
    "linearity": {
        "name": "Linearity",
        "explanation": (
            "**Linearity:** The relationship between X and Y should be linear (not curved). "
            "Important for linear regression and Pearson correlation."
        ),
        "how_to_check": [
            "Scatter plot: look for linear pattern (not curved)",
            "Residuals vs Fitted plot: residuals should be randomly scattered (not curved)",
        ],
        "what_if_violated": (
            "If relationship is curved: (1) Transform X or Y (log, polynomial), "
            "(2) Use polynomial regression, or (3) Use Spearman correlation (rank-based, doesn't assume linearity)."
        ),
    },
    
    "homoscedasticity": {
        "name": "Homoscedasticity (Constant Variance)",
        "explanation": (
            "**Homoscedasticity:** Spread of residuals should be constant across fitted values. "
            "If spread changes, predictions are less reliable."
        ),
        "how_to_check": [
            "Residuals vs Fitted plot: spread of points should be roughly constant",
            "Scale-Location plot: should be roughly horizontal",
        ],
        "what_if_violated": (
            "If variance increases with fitted values: Use robust standard errors or "
            "weighted least squares (WLS) regression."
        ),
    },
    
    "no_multicollinearity": {
        "name": "No Multicollinearity",
        "explanation": (
            "**Multicollinearity:** Predictors should not be highly correlated with each other. "
            "High correlation makes it hard to isolate individual effects."
        ),
        "how_to_check": [
            "VIF (Variance Inflation Factor): all should be < 5 (ideally < 3)",
            "Correlation matrix: no pairs should have |r| > 0.8",
        ],
        "what_if_violated": (
            "If VIF > 5: Remove or combine correlated predictors. "
            "Or use regularization (ridge regression, LASSO)."
        ),
    },
}


# ============================================================================
# TEST SELECTION GUIDE
# ============================================================================

TEST_SELECTION_GUIDE = """
## Which Test Should I Use?

### Comparing 2 Groups

**Do you have paired data?** (same subjects measured twice)
- YES → Use **Paired t-test**
- NO → Use **Independent samples t-test**

**Is your data normal?** (check with Shapiro-Wilk or Q-Q plot)
- YES → Use parametric test (above)
- NO → Use non-parametric: **Mann-Whitney U** (independent) or **Wilcoxon signed-rank** (paired)

### Comparing 3+ Groups

**Do you have one independent variable (IV) or two?**
- ONE IV → Use **One-way ANOVA** (parametric) or **Kruskal-Wallis** (non-parametric)
- TWO IVs → Use **Two-way ANOVA** (tests main effects + interaction)

**Is your data normal?**
- YES → Use ANOVA
- NO → Use Kruskal-Wallis (one-way only)

**Are measurements repeated?** (same subjects multiple times)
- YES → Use **Repeated-measures ANOVA** (assumes sphericity)
- NO → Use standard ANOVA

### Examining Relationships

**Do you want to measure correlation?**
- Linear relationship, normal data → **Pearson correlation**
- Non-normal data or rank-based → **Spearman correlation**

**Do you want to predict one variable from another?**
- Continuous outcome → **Linear Regression** (OLS)

**Do you want to test an indirect effect?** (X → M → Y)
- YES → **Mediation Analysis** (Pingouin)

---

**Remember:** Always check assumptions BEFORE running the test!
"""


# ============================================================================
# INTERPRETATION TEMPLATES
# ============================================================================

def interpret_normality_tests(norm_df, column_name, n_samples=None):
    """
    Generate plain-English interpretation of normality test results.
    
    Args:
        norm_df: DataFrame with normality test results
        column_name: Name of tested variable
        n_samples: Sample size (for Shapiro-Wilk interpretation)
    
    Returns:
        str: Interpretation text
    """
    # Count how many tests reject normality (p < 0.05). The Anderson-Darling
    # row stores a string in this column ("CV=... @ 5%"), so coerce first.
    if "p-value" in norm_df:
        p_numeric = pd.to_numeric(norm_df["p-value"], errors="coerce")
        non_normal_count = int((p_numeric < 0.05).sum())
    else:
        non_normal_count = 0
    total_tests = len(norm_df)
    
    if non_normal_count == 0:
        return (
            f"✓ **{column_name} appears normally distributed.** "
            f"All tests (p > 0.05) support normality. "
            f"Safe to use parametric tests (t-test, ANOVA, linear regression)."
        )
    elif non_normal_count < total_tests / 2:
        return (
            f"⚠ **{column_name} shows mixed evidence for normality.** "
            f"Some tests suggest non-normality. Consider: "
            f"(1) Transform data (log, sqrt), "
            f"(2) Use non-parametric alternatives (Mann-Whitney U, Kruskal-Wallis), or "
            f"(3) Check if violation is severe."
        )
    else:
        return (
            f"✗ **{column_name} appears non-normally distributed.** "
            f"Most tests reject normality (p < 0.05). "
            f"**Recommendation:** Use non-parametric tests (Mann-Whitney U, Kruskal-Wallis, Spearman). "
            f"Or transform the data and re-test."
        )


def interpret_ttest_result(t_stat, p_value, cohens_d, group1_mean, group2_mean, group_names=None):
    """Generate plain-English interpretation of t-test results."""
    sig_level = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))
    sig_text = (
        "**SIGNIFICANT**" if p_value < 0.05 
        else "**NOT SIGNIFICANT**"
    )
    
    effect_size_interpret = (
        "negligible" if abs(cohens_d) < 0.2
        else ("small" if abs(cohens_d) < 0.5 else ("medium" if abs(cohens_d) < 0.8 else "large"))
    )
    
    g1_name = group_names[0] if group_names else "Group 1"
    g2_name = group_names[1] if group_names else "Group 2"
    
    return f"""
**T-Test Result Interpretation:**

- **Result:** {sig_text} {sig_level}
- **P-value:** {p_value:.4f} (probability of this result if no real difference)
- **Effect size (Cohen's d):** {cohens_d:.3f} ({effect_size_interpret})
- **Mean difference:** {group1_mean:.2f} vs {group2_mean:.2f} ({abs(group1_mean - group2_mean):.2f} units)

**Plain English:** 
The mean difference between {g1_name} and {g2_name} is {effect_size_interpret}.
With p = {p_value:.4f}, this result is {'statistically significant (likely a real difference)' if p_value < 0.05 else 'not statistically significant (may be due to chance)'}.
    """


def interpret_anova_result(f_stat, p_value, eta_squared, omega_squared, num_groups):
    """Generate plain-English interpretation of ANOVA results."""
    sig_text = "**SIGNIFICANT**" if p_value < 0.05 else "**NOT SIGNIFICANT**"
    effect_size_text = (
        "small" if eta_squared < 0.06
        else ("medium" if eta_squared < 0.14 else "large")
    )
    
    return f"""
**ANOVA Result Interpretation:**

- **Result:** {sig_text}
- **F-statistic:** {f_stat:.3f}
- **P-value:** {p_value:.4f}
- **Effect size (η²):** {eta_squared:.4f} ({effect_size_text})
- **Effect size (ω², less biased):** {omega_squared:.4f}

**Plain English:**
The differences between {num_groups} groups are {'statistically significant' if p_value < 0.05 else 'not statistically significant'}.
The effect size is {effect_size_text}, meaning groups differ by a {effect_size_text} amount.
    """


def interpret_correlation_result(r_value, p_value, method="Pearson"):
    """Generate plain-English interpretation of correlation results."""
    # Interpret strength
    abs_r = abs(r_value)
    if abs_r < 0.3:
        strength = "weak"
    elif abs_r < 0.7:
        strength = "moderate"
    else:
        strength = "strong"
    
    direction = "positive" if r_value > 0 else "negative"
    sig_text = "statistically significant" if p_value < 0.05 else "not statistically significant"
    
    return f"""
**{method} Correlation Interpretation:**

- **Correlation (r):** {r_value:.4f}
- **P-value:** {p_value:.4f} ({sig_text})
- **Strength:** {strength.upper()} {direction} correlation

**Plain English:**
The two variables have a {strength} {direction} relationship.
As one increases, the other tends to {'increase' if r_value > 0 else 'decrease'}.
The relationship is {sig_text} (p = {p_value:.4f}).

**Important:** Correlation does NOT imply causation!
    """


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_glossary_term(term_key):
    """Retrieve glossary entry for a term."""
    return STAT_GLOSSARY.get(term_key, f"No explanation available for '{term_key}'.")


def get_assumption_info(assumption_key):
    """Retrieve full assumption information."""
    return ASSUMPTIONS.get(assumption_key, None)


def render_assumption_checklist(test_name):
    """Return assumption checklist for a specific test."""
    checklist_map = {
        "independent_ttest": ["normality", "equal_variance", "independence"],
        "paired_ttest": ["normality", "independence"],
        "oneway_anova": ["normality", "equal_variance", "independence"],
        "twoway_anova": ["normality", "equal_variance", "independence"],
        "pearson_correlation": ["linearity", "no_multicollinearity"],
        "linear_regression": ["linearity", "independence", "homoscedasticity", "no_multicollinearity", "normality"],
    }
    
    return checklist_map.get(test_name, [])
