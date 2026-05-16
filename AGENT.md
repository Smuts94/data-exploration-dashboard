# AGENT.md — Local EDA Dashboard for PhD Research

## Purpose

Build a locally hosted, interactive dashboard that allows a researcher to upload
a dataset and immediately conduct rigorous exploratory data analysis (EDA).
The app must prioritize statistical depth over aesthetics. The entire app runs
locally with no external API calls, no authentication, and no telemetry.

---

## Tech Stack

| Layer | Library | Reason |
|---|---|---|
| UI / Dashboard | Streamlit | Fast, Python-native, local hosting out of the box |
| Data wrangling | Pandas, NumPy | Standard, robust |
| Statistical tests | SciPy, Statsmodels, Pingouin | Normality, correlation significance, regression, ANOVA, t-tests, mediation |
| Visualizations | Plotly (interactive), Seaborn (static exports) | Interactivity for exploration |
| Regression | Statsmodels OLS | Coefficients, p-values, CIs, AIC, BIC — full inferential output |

> **Alternative**: If interactivity requirements grow complex, swap Streamlit for Dash (Plotly).

---

## Folder Structure

```
project/
├── AGENT.md
├── app.py                  # Entry point, sidebar title, welcome page
├── requirements.txt
├── pages/
│   ├── 0_upload.py
│   ├── 1_univariate.py
│   ├── 2_correlation.py
│   ├── 3_filter.py
│   ├── 4_regression.py
│   └── 5_statistical_tests.py
├── core/
│   ├── loader.py           # File parsing, type inference, validation
│   ├── stats.py            # All statistical computations (pure functions)
│   ├── plots.py            # All plot-generating functions
│   ├── state.py            # Streamlit session_state management
│   ├── sidebar.py          # Global study/group/variable filter sidebar component
│   └── group_utils.py      # Per-visual local group selector helper
└── tests/
    └── test_stats.py       # Unit tests for statistical functions
```

---

## State Management

Use `st.session_state` as the single source of truth across all pages.

```python
# Keys to maintain across pages
st.session_state["raw_df"]           # Original uploaded dataframe — never mutated
st.session_state["filtered_df"]      # Dataframe after Page 3 range/category/date filters
st.session_state["col_types"]        # User-overridden column type map
st.session_state["filters"]          # Active filter config dict (Page 3)

# Study / group / variable context (set on Upload page, refined on every analysis page)
st.session_state["study_col"]        # Column designated as the Study identifier (or None)
st.session_state["group_col"]        # Column designated as the Group identifier (or None)
st.session_state["selected_study"]   # Single study value currently active (str | None)
st.session_state["selected_groups"]  # List of group values currently selected (2+ for comparison)
st.session_state["selected_vars"]    # List of numeric columns selected for analysis
```

**Two-layer filtering model:**

- **Layer 1 — Page 3 filters** (`filtered_df`): range sliders, multiselect, and date pickers that
  broadly subset the dataset. Persisted in `session_state["filtered_df"]`.
- **Layer 2 — Global sidebar** (`analysis_df`): study, group, and variable selectors rendered in
  the sidebar on every analysis page. They further subset `filtered_df` for the current view.
  `analysis_df` is computed on the fly from `filtered_df` — it is **not** stored in
  `session_state` (it is a local variable returned by `sidebar.render()`).

- `raw_df` is never mutated after upload
- All analysis pages use `analysis_df`, never `raw_df` or `filtered_df` directly

---

## App Structure & Features

Build as a multi-page Streamlit app using the `pages/` directory convention.
Streamlit will automatically render a sidebar with navigation links to each page.
`app.py` is the entry point and handles the sidebar title and any global configuration.

### Global Sidebar — Study / Group / Variable Filters (`core/sidebar.py`)

Rendered at the top of the sidebar on every analysis page (Pages 1–4) via
`core.sidebar.render(filtered_df, col_types)`. Returns `analysis_df` — the
working DataFrame for that page.

**Study filter** (shown only if `study_col` is set):
- **Single-select `st.selectbox`** — only one study is active at a time
- All unique values in the study column are shown as options
- Default: first study in sorted order (or the previously selected study if still valid)
- Selecting a study rows-filters `filtered_df` to that study only — every visual on the
  page reflects exactly that study's data
- Rationale: studies are independent experimental units; mixing them would conflate data

**Group filter** (shown only if `group_col` is set):
- Multiselect of all unique values in the group column
- Default: all groups selected
- Minimum 1 group required; warn if 0 selected
- When exactly 2 groups are selected, plots that support it add a group-coloured
  overlay or split view (e.g. histogram KDE per group, scatter colour by group)

### Per-Visual Local Group Selector (`core/group_utils.py`)

In addition to the global sidebar group filter, **every table and plot** on each analysis
page exposes its own local "Split by group" selectbox directly above the visual.

- Implemented in `core/group_utils.py` via `local_group_selector(df, col_types, key)`
- Returns `(group_col, group_vals, subsets)` — one DataFrame subset per group value
- When a group column is chosen, the visual renders inside `st.tabs()` — one tab per
  group value — so the researcher sees that visual independently for each group
- When "— All data —" is selected (default), the visual renders on the full `analysis_df`
  as before, with no tabs
- The local selector is completely independent of the global sidebar group filter;
  a researcher can have different group splits on different visuals simultaneously

**Variable filter**:
- Multiselect of all numeric columns currently active in `col_types`
- Default: all numeric columns selected
- Controls which columns appear in descriptive tables and correlation matrix
- Does not affect axes that the user explicitly picks from a dropdown

**Active filter summary**: always show a compact badge/caption in the sidebar
listing current study name, groups, and variable count so the researcher knows
exactly what data is in view.

### Page 0 — Upload & Data Preview (`0_upload.py`)

- File uploader accepting `.csv`, `.xlsx`, `.tsv`
- Auto-detect delimiter and encoding on load
- On successful upload, populate `session_state["raw_df"]` and `session_state["filtered_df"]`
- Display:
  - Dataset shape (rows × columns)
  - Column names, inferred dtypes, and first 10 rows
  - Missing value summary: count and % per column, shown as a sortable table
  - Duplicate row count with option to preview duplicate rows
- Allow user to override column types via a selectbox per column
  (options: Numeric, Categorical, DateTime) — store overrides in `session_state["col_types"]`
- Show a warning if any column has > 50 unique values and is typed as Categorical
- **Study / Group column designation** (new):
  - Two additional selectboxes: "Study column" and "Group column"
  - Options: "— none —" plus all Categorical columns
  - Stored in `session_state["study_col"]` and `session_state["group_col"]`
  - These drive the global sidebar filters on all analysis pages

### Page 1 — Univariate Statistics (`1_univariate.py`)

Calls `sidebar.render()` → operates on `analysis_df`.

**For numeric columns** (restricted to `selected_vars`):

- Full descriptive statistics table per column:
  Mean, median, std, variance, min, max, skewness, kurtosis, IQR,
  coefficient of variation (CV)
- Distribution plot: histogram + KDE overlay (Plotly) — column selectable via dropdown
  - If exactly 2 groups are selected, overlay one KDE per group on the same plot (colour-coded)
- Q-Q plot per selected column (Plotly or Matplotlib)
- Normality tests — run all applicable tests and display results in a single table
  with test statistic, p-value, and pass/fail at α = 0.05:
  - Shapiro-Wilk → only if n < 5000
  - D'Agostino-Pearson K²
  - Kolmogorov-Smirnov (tested against normal distribution)
  - Anderson-Darling
  - If n ≥ 5000, skip Shapiro-Wilk and display a notice explaining why
- Outlier summary table:
  - IQR method: count of outliers below Q1 - 1.5×IQR and above Q3 + 1.5×IQR
  - Z-score method: count of rows with |z| > 3
  - Allow user to click through to inspect the flagged rows

**For categorical columns:**

- Value counts table + horizontal bar chart (Plotly)
- Mode and unique value count displayed as metrics
- Cardinality warning if unique values > 50

### Page 2 — Bivariate & Correlation Analysis (`2_correlation.py`)

Calls `sidebar.render()` → operates on `analysis_df`.

- **Correlation matrix:**
  - User selects method: Pearson, Spearman, or Kendall via radio button
  - Display a single annotated heatmap (Plotly) with correlation coefficients and
    significance stars appended directly to each cell value (e.g. `0.83***`)
  - Star legend: `*` p<0.05, `**` p<0.01, `***` p<0.001
- **Scatter plot explorer:**
  - User picks X column and Y column from dropdowns (numeric only)
  - Optional: color encoding by a categorical column, size encoding by a numeric column
  - If `group_col` is set and 2+ groups are selected, default the color encoding to the group column
  - Display Pearson r and p-value for the selected pair above the chart
- **Pairplot:**
  - User selects a subset of numeric columns (warn and cap at 8 columns for performance)
  - Render via Seaborn `pairplot()` → save to `BytesIO` buffer → display with `st.image()`
  - Require a "Generate Pairplot" button press — do not auto-render on page load
  - If n > 100k rows, sample 10k rows for the pairplot and notify the user
- **Grouped statistics:**
  - User selects one categorical column as the grouping variable
  - Display a pivot table of mean and median for all numeric columns per group

### Page 3 — Data Filtering & Subsetting (`3_filter.py`)

Calls `sidebar.render()` to show current study/group context (read-only display on this page —
the study/group selectors are shown but the page's own filter panel is the primary control).
Page 3 operates on `raw_df` and writes to `session_state["filtered_df"]`.

- Dynamic filter panel built from column types in `session_state["col_types"]`:
  - Numeric columns → range slider (min/max from data)
  - Categorical columns → multi-select dropdown (all values selected by default)
  - DateTime columns → date range picker (if any detected)
- Display live row count and % of original data retained as filters are adjusted
- On filter change, update `session_state["filtered_df"]` immediately
- "Reset All Filters" button restores `filtered_df` to `raw_df`
- "Download Filtered Dataset" button exports current `filtered_df` as CSV
- Display a preview table of the filtered dataset (first 20 rows)
- If filtered dataset is empty, show a clear warning and block downstream pages

### Page 4 — Linear Regression (`4_regression.py`)

Calls `sidebar.render()` → operates on `analysis_df`.

- User selects:
  - Dependent variable (Y): numeric columns only
  - Independent variables (X): one or more numeric columns (multi-select)
  - Checkbox: include intercept (default: yes)
- Require a "Run Regression" button press — do not auto-run on page load
- Run OLS via `statsmodels.OLS` — not scikit-learn
- Display:
  - Full `model.summary()` rendered cleanly in a code block or via `st.text()`
  - Key metrics called out explicitly as `st.metric()` cards:
    R², Adjusted R², F-statistic p-value, AIC, BIC, Durbin-Watson, condition number
  - Multicollinearity warning if condition number > 30
  - Coefficient plot: point estimates + 95% confidence intervals (Plotly)
  - VIF table: variance inflation factor per predictor
    (warn if any VIF > 5, flag as severe if > 10)
  - Residual diagnostic plots (all four, displayed in a 2×2 grid):
    - Residuals vs. Fitted
    - Q-Q plot of residuals
    - Scale-Location (√|residuals| vs. fitted)
    - Residuals vs. Leverage with Cook's distance contours

### Page 5 — Statistical Tests (`5_statistical_tests.py`)

Calls `sidebar.render()` → operates on `analysis_df`.

Three test families, each in its own expander/section. All tests require explicit
"Run" button press — do not auto-execute on page load.

#### T-Tests

- User selects test type via radio button:
  - **Independent samples t-test** — compare means of two separate groups
  - **Paired samples t-test** — compare two numeric columns row-by-row (paired observations)
  - **One-sample t-test** — compare a column's mean against a user-supplied hypothesised mean
- For independent/one-sample: user selects the numeric column(s) and (for independent) the
  grouping categorical column; warn if the grouping column has ≠ 2 unique values in the
  current filter context
- Levene's test for equality of variances runs automatically before the independent t-test;
  if Levene p < 0.05, switch to Welch's t-test and surface a notice
- Display: t-statistic, degrees of freedom, p-value, Cohen's d effect size, 95% CI on
  the mean difference, and a plain-English interpretation sentence
- If normality assumption is violated (Shapiro-Wilk or D'Agostino p < 0.05 for either
  group), offer the non-parametric alternative:
  - Mann-Whitney U (independent) or Wilcoxon signed-rank (paired), and display its
    results alongside the parametric results

#### ANOVA

- User selects test type via radio button:
  - **One-way ANOVA** — one numeric DV, one categorical IV (≥ 2 levels)
  - **Two-way ANOVA** — one numeric DV, two categorical IVs (main effects + interaction)
  - **Repeated-measures ANOVA** — one numeric DV, one within-subject factor column,
    one subject-ID column (requires Pingouin)
- For one-way and two-way: use `scipy.stats.f_oneway` (one-way) or `statsmodels` OLS
  with `anova_lm` (two-way)
- Post-hoc tests (one-way and two-way, run automatically when overall F is significant):
  - Tukey HSD via `statsmodels.stats.multicomp.pairwise_tukeyhsd`
  - Display as a styled table with mean difference, p-adj, confidence interval,
    and a "Reject H₀" boolean column
- Assumptions checks (displayed in a collapsible section):
  - Normality per group: Shapiro-Wilk (if n < 5000) shown per group
  - Homogeneity of variances: Levene's test
- Display: ANOVA table (SS, df, MS, F, p), η² (eta-squared) and ω² (omega-squared)
  effect-size metrics as `st.metric()` cards, and a bar chart of group means ± 1 SD
  (Plotly)
- If normality or homogeneity assumption is violated, offer Kruskal-Wallis as the
  non-parametric alternative (one-way only) and display its results

#### Mediation Analysis (Section 6)

- Uses Pingouin `pg.mediation_analysis()` under the hood
- User selects:
  - X (independent variable, numeric)
  - M (mediator variable, numeric)
  - Y (dependent variable, numeric)
  - Optional: covariate columns (multiselect, numeric)
  - Number of bootstrap samples: slider 500–5000 (default 1000)
- Require "Run Mediation" button press
- Display:
  - Path table: coefficients and p-values for paths a, b, c, c' (direct), and indirect (ab)
  - Bootstrap 95% CI for the indirect effect (ab); if CI excludes zero, label as
    "Significant mediation"
  - Mediation type classification: full mediation, partial mediation, or no mediation
    — determined from whether c' is significant and CI of ab excludes zero
  - A path diagram rendered as a simple Plotly figure (four nodes: X → M → Y, X → Y,
    with coefficients annotated on each arrow)

#### Multilevel Mediation — 2-1-1 (Section 7)

- Tests indirect effects via one or more mediators in nested/clustered data using Linear Mixed Models (LMM)
- Implemented in `core/stats.py` as `run_multilevel_mediation()`; uses `statsmodels.formula.api.mixedlm`
- User selects:
  - Cluster / Subject ID column (categorical or numeric ID)
  - Outcome Y (numeric)
  - Predictor X (numeric or categorical, Level-2)
  - One or more mediators M (numeric, tested independently)
  - Optional: Level-2 covariates (multiselect)
  - Optional: Level-1 (within-cluster) predictors of Y (multiselect)
  - Bootstrap iterations: slider 100–2000 (default 500)
- Bootstrap resamples **clusters** (not rows) to preserve within-cluster structure
- All column names are renamed to safe Python identifiers before fitting, then mapped back for display
- Require "Run Multilevel Mediation" button press
- Display:
  - Summary metrics: n_obs, n_clusters, bootstrap convergence count
  - Path table with all path coefficients and significance stars
  - Indirect effects table with 95% bootstrap CIs and significance badge (CI excludes 0)
  - Total effect metric + 95% CI
  - Plotly path diagram: X (blue, left) → mediators (orange, centre) → Y (green, right);
    solid arrows for a/b paths, dotted arrow for direct effect c'
  - Warnings for low cluster count or low bootstrap convergence rate
- Gated behind explicit button press; warns about runtime

---

## `requirements.txt`

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.26.0
scipy>=1.12.0
statsmodels>=0.14.0
plotly>=5.20.0
seaborn>=0.13.0
matplotlib>=3.8.0
openpyxl>=3.1.0
pingouin>=0.5.4
```

---

## Key Implementation Constraints

- **Fully local** — no external API calls, no telemetry, no authentication
- **Never mutate `raw_df`** — all filtering and transformations operate on copies
- **Two-layer filtering**: Page 3 writes `filtered_df`; `sidebar.render()` returns `analysis_df`
  as a further subset — analysis pages always use `analysis_df`
- **Normality test guards:**
  - Never run Shapiro-Wilk on n ≥ 5000 — enforce this in `stats.py`, not in the page
  - Always display which tests were skipped and why
- **Performance guards:**
  - Pairplot and regression require explicit button press — never auto-run
  - If n > 100k rows, sample for pairplot and Shapiro-Wilk; always notify the user
  - Cap pairplot column selection at 8 columns
- Use `statsmodels.OLS` for regression — never sklearn — researchers need
  inferential statistics (p-values, CIs, AIC, BIC), not just predictions
- Mediation analysis uses Pingouin — wrap import in try/except and surface a clear
  install message if Pingouin is missing
- ANOVA post-hoc (Tukey HSD) only runs when the omnibus F-test p < 0.05
- Seaborn pairplot rendering: render to matplotlib figure → save to `BytesIO`
  → display via `st.image()` — do not attempt to make it interactive
- **Error handling:** every page must handle gracefully:
  - No file uploaded yet → redirect message to Page 0
  - Empty filtered dataset → clear warning, block regression page
  - Wrong column types selected → informative error, no crash
  - Singular matrix in OLS → catch and surface the error clearly
- No hardcoded column names — all logic must be fully dynamic based on
  whatever dataset the user uploads
- Do not use deprecated `st.experimental_*` APIs
- Use the `pages/` directory convention for multi-page routing — do not use `st.navigation()` or `st.tabs()` for top-level page separation

---

## What the Agent Must NOT Do

- Do not add authentication of any kind
- Do not make any external network requests
- Do not add ML beyond OLS linear regression (no clustering, classification, or
  dimensionality reduction)
- Do not auto-run computationally heavy operations on page load (pairplot,
  regression) — always gate behind a button
- Do not swallow exceptions silently — all errors must surface to the user
  with a clear, plain-English message
- Do not hardcode any dataset-specific logic, column names, or assumptions

---

---

## Student Education Enhancement Strategy

**Goal:** Transform the dashboard from a powerful analysis tool into a **learning tool** that helps students understand not just *what* statistics mean, but *why* they matter and *what to do* with them.

### New Module: `core/explanations.py`

This module centralizes all educational content:

```python
# Terminology glossary (key → explanation)
STAT_GLOSSARY = {
    "mean": "Average of all values. Sensitive to outliers.",
    "median": "Middle value when sorted. Robust to outliers.",
    "skewness": "Measures asymmetry (|skew| > 1 suggests non-normal).",
    "kurtosis": "Measures tail heaviness (>3 suggests heavy tails).",
    "iqr": "Interquartile range (Q3 - Q1). Contains middle 50% of data.",
    "cv": "Coefficient of variation (SD/mean). Compare spread across scales.",
    "p_value": "Probability that result occurred by chance if null is true. p<0.05 is conventional threshold.",
    "effect_size": "Practical magnitude of difference, independent of sample size.",
    "correlation": "Strength & direction of linear relationship (-1 to +1).",
    "r_squared": "Proportion of variance explained (0–1). Higher is better.",
    "vif": "Variance Inflation Factor. >5 suggests multicollinearity.",
    # ... more terms
}

# Test assumptions (test_name → list of assumption names)
ASSUMPTIONS_BY_TEST = {
    "independent_ttest": ["normality", "equal_variance", "independence"],
    "paired_ttest": ["normality", "independence"],
    "oneway_anova": ["normality", "homogeneity_of_variance", "independence"],
    "correlation": ["linearity", "scale", "no_extreme_outliers"],
    # ... etc
}

# Assumption explanations (assumption → what it means & how to check)
ASSUMPTION_CHECKS = {
    "normality": {
        "meaning": "Data approximately follows normal distribution.",
        "check": "Shapiro-Wilk, Q-Q plot, histogram shape.",
        "fix": "Transform data (log, sqrt) or use non-parametric test.",
    },
    # ... etc
}

# Interpretation templates (test → template string)
RESULT_TEMPLATES = {
    "independent_ttest": "...",
    # ... etc
}

# Helper functions
def get_glossary_entry(term: str) -> str:
    """Retrieve explanation for a statistical term."""
    pass

def get_assumption_warning(test: str, violated: list[str]) -> str:
    """Generate warning about violated assumptions."""
    pass

def get_result_interpretation(test: str, **results) -> str:
    """Generate plain-English interpretation of test results."""
    pass
```

### 3_Descriptive_Stats.py — Enhancements

**Add education blocks:**
- Before descriptive table: info box explaining which metrics to focus on
- After normality tests: interpretation guide & what to do if non-normal
- Outlier section: explain IQR vs Z-score methods & when to use each

Example:
```python
with st.expander("📚 Understanding Descriptive Stats", expanded=False):
    st.markdown("""
    **Mean vs Median:** Use median if data is skewed or has outliers.
    
    **Variance & SD:** Measure spread; higher = more variability.
    
    **Skewness:** |skew| > 1 suggests non-normal.
    - Positive: tail to the right
    - Negative: tail to the left
    
    **Normality matters for:** t-tests, ANOVA, linear regression.
    """)
```

### 4_Correlation.py — Enhancements

- Add info box after correlation heatmap explaining:
  - What Pearson / Spearman / Kendall mean & when to use each
  - How to interpret correlation strength (0–0.3 weak, 0.3–0.7 moderate, etc.)
  - What p-value means (reject null = correlation ≠ 0, not causation)
  
- Scatter plot: add interpretation guide ("positive / negative / no relationship")

### 5_Regression.py — Enhancements

- Pre-regression: assumption checklist (normality of residuals, homoscedasticity, etc.)
- Post-regression: interpretation template showing:
  - What R² means
  - How to read coefficient significance
  - VIF warnings & what multicollinearity means
  - Diagnostic plot guidance (what to look for in each plot)

### 6_Statistical_Tests.py — Enhancements

- **Pre-test assumption section:** show checklist of required assumptions
- **Post-test result block:** plain-English interpretation + effect size guidance
- Add a "Test Selection Guide" expander at top showing decision tree:
  ```
  Comparing 2 groups?
  ├─ Paired data → Paired t-test
  ├─ Normal + equal variance → Independent t-test  
  └─ Non-normal → Mann-Whitney U
  ```

### Implementation: Progressive Disclosure

All educational content uses collapsible expanders to avoid overwhelming advanced users:
```python
with st.expander("📚 Learn more about [concept]", expanded=False):
    st.markdown(explanation)
```

### CSV Export Enhancement

When users download results, include a metadata header:
```python
def enhance_export(df: pd.DataFrame, test_name: str, **metadata) -> pd.DataFrame:
    """Prepend metadata & interpretation guide to exported CSV."""
    header = f"# {test_name}\n# Test Description: ...\n# Assumptions: ...\n# Interpretation: ...\n"
    # Return df with metadata
```

---

## Reproducible Code Export (R & Python)

**Goal:** Every analysis result is accompanied by downloadable, standalone **R** and
**Python** scripts that reproduce it on the student's own machine. The dashboard
always computes in Python; R is an **export-only artifact** — no R runtime is
required to run the app, and `requirements.txt` is unchanged.

### Modules

- `core/codegen.py` — pure module (**no Streamlit import**). Given a `Provenance`
  and a list of analysis specs, emits standalone `.py` and `.R` script strings.
- `core/export_ui.py` — Streamlit glue: builds `Provenance` from `session_state`,
  renders the per-result export expander, and wires the whole-session export.

### Provenance — the layers a script must replay

A faithful script reproduces the *analysed subset*, not the raw upload:

1. **Load** — `read.csv` / `pd.read_csv` of a `DATA_FILE` constant the student edits
2. **Page 2 filters** — numeric `between`, categorical `isin`, datetime range;
   every filter keeps NaN rows, matching `2_Data_Filter.py`
3. **Sidebar study selection** — `df[df[study] == value]`
4. **Sidebar group selection** — `df[df[group].isin([...])]`

Column names are emitted with `df[["name"]]` (R) / `df[VAR]` (Python) indexing so
names containing spaces survive. Variable-Selection column projection is *not*
reproduced — dropping columns changes no statistic.

### Accuracy tiers

- **Tier 1 — R reproduces the dashboard exactly:** descriptives, outliers,
  normality, correlation, t-tests, one-way & two-way ANOVA, Tukey HSD, OLS + VIF.
- **Tier 2 — R is canonical but diverges:** repeated-measures ANOVA, mediation,
  multilevel mediation. Cause: statsmodels `mixedlm` vs `lme4`/`lmerTest`, and
  bootstrap RNG differing across languages. Each Tier 2 R script carries a header
  `NOTE:` explaining the expected divergence; bootstrap analyses fix a seed.

### State

`session_state["analysis_log"]` — an ordered list of `{"kind", "params"}` dicts,
appended (deduplicated, capped at 50) whenever an analysis is run. Drives the
whole-session export. Reset on upload. See `core/state.py` — `log_analysis`,
`get_analysis_log`, `clear_analysis_log`.

### UI

- **Per-result:** an "⬇ Reproduce this in R / Python" expander below each result
  with two download buttons. Rendered via `export_ui.render_export(kind, params, key=…)`.
- **Whole-session:** a sidebar button (`export_ui.render_session_export`) emits one
  script reproducing load → filters → every analysis in `analysis_log`.

### Constraints

- `core/codegen.py` must **not** import Streamlit — it stays unit-testable.
- Generated scripts must be standalone: they must not import the app's `core`
  package. Templates inline the scipy / statsmodels / R calls.
- `tests/test_codegen.py` executes every generated Python script against a
  fixture dataset — the guard against template syntax/logic drift.

---

## Definition of Done

**Core Functionality (existing):**
- [ ] User can upload CSV or Excel and see a full data summary in under 3 seconds
- [ ] Study and group columns can be designated on the Upload page
- [ ] Global sidebar filters (study, group, variables) appear on all analysis pages
- [ ] Selecting a study subset correctly rows-filters the data for that page
- [ ] Selecting exactly 2 groups overlays per-group KDE on histogram and defaults scatter colour to group
- [ ] Variable filter correctly restricts descriptive tables and correlation matrix
- [ ] All four normality tests run with correct n-size guards enforced in `stats.py`
- [ ] Correlation heatmap displays both coefficients and p-value significance stars
- [ ] Pairplot renders correctly via `st.image()` and is gated behind a button
- [ ] Filters update `session_state["filtered_df"]` and persist to the regression page
- [ ] OLS output matches statsmodels reference output exactly — no rounding or omission of fields
- [ ] All four residual diagnostic plots are rendered
- [ ] VIF table is displayed with correct threshold warnings
- [ ] T-test runs parametric and (if assumptions violated) non-parametric tests; Cohen's d shown
- [ ] ANOVA displays F-table, effect sizes (η², ω²), and Tukey HSD post-hoc when p < 0.05
- [ ] Mediation analysis shows all four paths, bootstrap CI, and mediation classification
- [ ] Path diagram rendered as Plotly figure with annotated coefficients
- [ ] App launches with `streamlit run app.py` with zero additional configuration
- [ ] All pages handle an empty or unfiltered state without crashing

**Education Enhancement (NEW):**
- [ ] `core/explanations.py` created with glossary, assumptions, & interpretation guides
- [ ] All statistical terms have accessible explanations in STAT_GLOSSARY
- [ ] Descriptive Stats page has education blocks for mean/median, normality, outliers
- [ ] Correlation page explains Pearson/Spearman/Kendall & correlation strength
- [ ] Regression page shows assumption checklist & VIF/multicollinearity guidance
- [ ] Statistical Tests page has test selection guide & result interpretation templates
- [ ] All education content uses collapsible expanders (progressive disclosure)
- [ ] Assumption violations surface warning messages with remediation advice
- [ ] Students can understand outputs without external documentation

**Reproducible Code Export (NEW):**
- [ ] `core/codegen.py` created — pure, no Streamlit import — with `Provenance`,
      `python_script()`, `r_script()`, and a per-analysis template `REGISTRY`
- [ ] `core/export_ui.py` created — `render_export()` and `render_session_export()`
- [ ] `session_state["analysis_log"]` added to `core/state.py`, reset on upload
- [ ] Every result on Pages 3–6 has an export expander with working `.R` + `.py` downloads
- [ ] Generated scripts replay the filter/study/group provenance and run standalone
- [ ] Tier 2 R scripts carry a header `NOTE:` and a fixed seed
- [ ] "Export full session" in the sidebar emits one cumulative script
- [ ] `tests/test_codegen.py` executes every generated Python script on a fixture
