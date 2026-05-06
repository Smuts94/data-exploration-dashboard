import streamlit as st

st.set_page_config(
    page_title="EDA Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("EDA Dashboard")
st.sidebar.caption("Local exploratory data analysis for PhD research.")

st.title("Welcome to the EDA Dashboard")
st.caption("A fully local, statistically rigorous exploratory data analysis tool built for PhD research.")

st.markdown("---")

pages = [
    (
        "0 · Data Upload",
        "Start here.",
        [
            "Upload a `.csv`, `.tsv`, or `.xlsx` file",
            "Preview dataset shape, column types, missing values, and duplicates",
            "Override inferred column types (Numeric / Categorical / DateTime)",
            "Designate a **Study column** (single-select filter) and a **Group column** (multi-select filter) that drive all analysis pages",
        ],
    ),
    (
        "1 · Variable Selection",
        "Reduce your dataset before any analysis.",
        [
            "Browse all columns with type, non-null count, unique values, and variance",
            "Use **Quick select** to keep the top N numeric columns by variance, numeric-only, or all",
            "Manually adjust which columns to carry forward in grouped selectors",
            "Applying the selection resets the Data Filter and restricts all downstream pages to the chosen columns",
        ],
    ),
    (
        "2 · Data Filter",
        "Subset rows before analysis.",
        [
            "Range sliders for numeric columns",
            "Multi-select dropdowns for categorical columns",
            "Date range pickers for DateTime columns",
            "Live row count and % of data retained; reset button and filtered CSV export",
        ],
    ),
    (
        "3 · Descriptive Stats",
        "Understand each variable individually.",
        [
            "Full descriptive statistics table — mean, median, SD, variance, skewness, kurtosis, IQR, CV",
            "Interactive histogram + KDE and Q-Q plot per variable",
            "Normality tests: Shapiro-Wilk (n < 5,000), D'Agostino-Pearson, Kolmogorov-Smirnov, Anderson-Darling",
            "Outlier summary (IQR method and Z-score method) with row-level drill-down",
            "Value counts and bar chart for categorical variables",
            "Per-visual group split with cascade reset from page-level defaults",
        ],
    ),
    (
        "4 · Correlation",
        "Explore relationships between variables.",
        [
            "Annotated correlation heatmap (Pearson / Spearman / Kendall) with significance stars",
            "Interactive scatter plot explorer with optional colour and size encoding",
            "Seaborn pairplot rendered as image (gated behind a button; sampled if n > 100 k)",
            "Grouped pivot table of means and medians by a categorical variable",
            "Exportable correlation matrix and p-value matrix",
        ],
    ),
    (
        "5 · Regression",
        "Fit and diagnose OLS linear models.",
        [
            "Select dependent variable and one or more predictors; optional intercept",
            "Full `statsmodels` OLS summary with R², Adj. R², F p-value, AIC, BIC, Durbin-Watson, condition number",
            "Coefficient plot with 95% confidence intervals",
            "Variance Inflation Factor (VIF) table with severity warnings",
            "Four residual diagnostic plots: Residuals vs Fitted, Q-Q, Scale-Location, Leverage/Cook's distance",
            "Run per group via the group-split selector",
        ],
    ),
    (
        "6 · Statistical Tests",
        "Formal hypothesis testing with assumption checks.",
        [
            "**T-Tests** — Independent samples (with Levene's test + Welch correction), Paired samples, One-sample; Cohen's d; Mann-Whitney U / Wilcoxon fallbacks if normality violated",
            "**One-way ANOVA** — F-table, η² and ω² effect sizes, Tukey HSD post-hoc, Kruskal-Wallis fallback, bar chart of group means ± SD",
            "**Two-way ANOVA** — main effects + interaction term (Type II SS via statsmodels)",
            "**Repeated-measures ANOVA** — Mauchly's sphericity test, Greenhouse-Geisser correction, FDR-corrected pairwise post-hoc (Pingouin)",
            "**Mediation analysis** — paths a, b, c, c′ and indirect effect (ab) with bootstrap CI, mediation type classification, and Plotly path diagram",
            "All major results exportable as CSV",
        ],
    ),
]

for title, subtitle, bullets in pages:
    with st.container(border=True):
        st.markdown(f"### {title}")
        st.caption(subtitle)
        for b in bullets:
            st.markdown(f"- {b}")

st.markdown("---")
st.info("Upload a file on **0 · Data Upload** to get started, then proceed through the pages in order.")
