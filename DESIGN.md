# Design Document

Architecture and design rationale for the EDA Dashboard. For a feature overview see [`README.md`](README.md); for the user-facing description of each page see `app.py`.

## Goals

A locally hosted, interactive Streamlit app that lets a researcher upload a dataset and run rigorous exploratory data analysis. Statistical depth is prioritized over aesthetics. The app runs fully offline — no external API calls, no authentication, no telemetry.

## Tech Stack

| Layer | Library | Reason |
|---|---|---|
| UI / Dashboard | Streamlit | Python-native, local hosting out of the box |
| Data wrangling | pandas, NumPy | Standard, robust |
| Statistical tests | SciPy, statsmodels, Pingouin | Normality, correlation, regression, ANOVA, t-tests, mediation |
| Visualizations | Plotly (interactive), Seaborn (static exports) | Interactivity for exploration |
| Regression | statsmodels OLS | Coefficients, p-values, CIs, AIC, BIC — full inferential output, not just predictions |

## Folder Structure

```
.
├── app.py                          # Entry point, sidebar title, landing page
├── requirements.txt
├── README.md
├── DESIGN.md
├── .streamlit/config.toml
├── pages/
│   ├── 0_Data_Upload.py
│   ├── 1_Variable_Selection.py
│   ├── 2_Data_Filter.py
│   ├── 3_Descriptive_Stats.py
│   ├── 4_Correlation.py
│   ├── 5_Regression.py
│   └── 6_Statistical_Tests.py
├── core/
│   ├── loader.py                   # File parsing, type inference
│   ├── stats.py                    # All statistical computations (pure functions)
│   ├── plots.py                    # All plot-generating functions
│   ├── state.py                    # Streamlit session_state management
│   ├── sidebar.py                  # Global study / group / variable filter component
│   ├── group_utils.py              # Per-visual local group selector helper
│   ├── explanations.py             # Education content: glossary, assumptions, interpretations
│   ├── codegen.py                  # Reproducible R & Python script generation (pure)
│   └── export_ui.py                # Streamlit glue for the code-export buttons
└── tests/
    ├── test_stats.py               # Unit tests for statistical functions
    └── test_codegen.py             # Tests for reproducible code export
```

## State Management

`st.session_state` is the single source of truth across all pages. Keys are defined as constants in `core/state.py`.

```python
st.session_state["raw_df"]            # Original uploaded DataFrame — never mutated
st.session_state["selected_cols"]     # Columns kept after Variable Selection
st.session_state["filtered_df"]       # DataFrame after Data Filter page
st.session_state["col_types"]         # Column type map (Numeric / Categorical / DateTime)
st.session_state["filters"]           # Active filter config

# Study / group / variable context
st.session_state["study_col"]         # Column designated as Study identifier
st.session_state["group_col"]         # Column designated as Group identifier
st.session_state["selected_study"]    # Single active study value
st.session_state["selected_groups"]   # Selected group values (multiselect)
st.session_state["selected_vars"]     # Numeric columns selected for analysis

st.session_state["analysis_log"]      # Ordered log of analyses run — drives code export
```

### Two-layer filtering model

- **Layer 1 — Data Filter page** (`filtered_df`): range sliders, multiselects, and date pickers that broadly subset the dataset. Persisted in session state.
- **Layer 2 — Global sidebar** (`analysis_df`): study / group / variable selectors rendered on every analysis page. Further subsets `filtered_df` for the current view. **Not** stored in session state — it is a local variable returned by `sidebar.render()` on each page render.

Invariants:
- `raw_df` is never mutated after upload.
- All analysis pages operate on `analysis_df`, never on `raw_df` or `filtered_df` directly.

### Global sidebar (`core/sidebar.py`)

Rendered at the top of the sidebar on every analysis page via `core.sidebar.render(filtered_df, col_types)`. Returns `analysis_df`.

- **Study filter** (shown only if `study_col` is set): single-select; rows-filters to that study only. Studies are independent experimental units; mixing them would conflate data.
- **Group filter** (shown only if `group_col` is set): multiselect; default all selected. When exactly 2 groups are selected, plots that support it add a per-group overlay or split view.
- **Variable filter**: multiselect of numeric columns currently active in `col_types`; default all selected. Controls which columns appear in descriptive tables and the correlation matrix.

A compact summary in the sidebar always shows the current study, groups, and variable count so the researcher knows exactly what data is in view.

### Per-visual local group selector (`core/group_utils.py`)

In addition to the global group filter, every table and plot on each analysis page exposes its own local "Split by group" selectbox above the visual. When a group column is chosen, the visual renders inside `st.tabs()` — one tab per group value. Default ("— All data —") renders on the full `analysis_df` with no tabs. The local selector is independent of the global sidebar group filter.

## Pages

### 0 — Data Upload

- File uploader for `.csv`, `.tsv`, `.xlsx`. Delimiter and encoding auto-detected.
- On upload, populates `raw_df`, `filtered_df`, and `col_types`.
- Displays shape, dtypes, first 10 rows, missing value summary, duplicate count.
- Per-column type override (Numeric / Categorical / DateTime).
- Cardinality warning if a Categorical column has > 50 unique values.
- Designate **Study column** and **Group column** from the categorical columns; these drive the global sidebar filters.

### 1 — Variable Selection

Reduces the dataset before any analysis runs.

- Browse all columns with type, non-null count, unique values, and variance.
- Quick-select presets: top N numeric columns by variance, numeric-only, all.
- Manual adjustment via grouped selectors.
- Applying a selection resets the Data Filter and restricts all downstream pages to the chosen columns.

### 2 — Data Filter

Operates on `raw_df` (restricted to `selected_cols`) and writes to `filtered_df`.

- Dynamic filter panel built from `col_types`: numeric → range slider, categorical → multiselect, datetime → date range picker.
- Live row count and % of original data retained.
- Reset button restores `filtered_df` to the working DataFrame.
- Download Filtered Dataset as CSV.
- Empty filtered dataset blocks downstream pages with a clear warning.

### 3 — Descriptive Stats

Operates on `analysis_df`. The univariate analysis page.

For numeric columns (restricted to `selected_vars`):
- Descriptive table: mean, median, SD, variance, min, max, skewness, kurtosis, IQR, CV.
- Histogram + KDE (Plotly). With exactly 2 groups selected, overlays one KDE per group.
- Q-Q plot per column.
- Normality tests in a single table (statistic, p-value, pass/fail at α = 0.05): Shapiro-Wilk (only if n < 5,000), D'Agostino-Pearson, Kolmogorov-Smirnov, Anderson-Darling.
- Outlier summary: IQR method (`Q1 − 1.5·IQR`, `Q3 + 1.5·IQR`) and Z-score method (`|z| > 3`); drill-down to flagged rows.

For categorical columns: value counts table + horizontal bar chart, mode and unique count, cardinality warning if > 50.

### 4 — Correlation

- Annotated heatmap (Pearson / Spearman / Kendall, selectable) with coefficients and significance stars: `*` p<0.05, `**` p<0.01, `***` p<0.001.
- Scatter plot explorer with optional color and size encoding. If `group_col` is set and 2+ groups are selected, color defaults to the group column. Pearson r and p-value displayed above the chart.
- Seaborn pairplot rendered to a `BytesIO` buffer and displayed via `st.image()`. Gated behind a button. Capped at 8 columns; if n > 100k, samples 10k rows and notifies.
- Grouped pivot table of mean and median for all numeric columns per categorical group.

### 5 — Regression

OLS via `statsmodels.OLS`. Never sklearn — researchers need inferential statistics, not just predictions.

- Select dependent variable, one or more predictors, optional intercept.
- Gated behind a Run Regression button.
- Full `model.summary()` plus `st.metric()` cards for R², Adjusted R², F p-value, AIC, BIC, Durbin-Watson, condition number.
- Multicollinearity warning if condition number > 30.
- Coefficient plot with 95% CIs.
- VIF table per predictor (warn if > 5, severe if > 10).
- Four residual diagnostic plots: Residuals vs Fitted, Q-Q, Scale-Location, Residuals vs Leverage with Cook's distance.

### 6 — Statistical Tests

Three test families, each in its own section. All tests gated behind a Run button.

**T-Tests** — Independent samples (with Levene's test → automatic Welch correction if variances unequal), Paired samples, One-sample. Cohen's d effect size, 95% CI on the mean difference, plain-English interpretation. Mann-Whitney U / Wilcoxon signed-rank fallbacks shown alongside parametric results when normality fails.

**ANOVA** — One-way (`scipy.stats.f_oneway`), Two-way (`statsmodels` OLS + `anova_lm`, Type II SS), Repeated-measures (Pingouin) with Mauchly's sphericity test and Greenhouse-Geisser correction. η² and ω² effect sizes. Tukey HSD post-hoc runs automatically when omnibus F is significant. Bar chart of group means ± 1 SD. Kruskal-Wallis fallback offered when assumptions fail (one-way only).

**Mediation Analysis** — Pingouin `pg.mediation_analysis()`. Path table for a, b, c, c′, indirect (ab) with bootstrap 95% CI. Mediation type classification (full / partial / none). Plotly path diagram with annotated coefficients. Bootstrap samples slider (500–5000, default 1000).

**Multilevel Mediation (2-1-1)** — `core/stats.py::run_multilevel_mediation` using `statsmodels.formula.api.mixedlm`. Bootstrap resamples *clusters*, not rows, to preserve within-cluster structure. All column names safely renamed to Python identifiers internally. Path table, indirect effects with bootstrap CIs, total effect, Plotly path diagram. Warnings for low cluster count or low bootstrap convergence.

## Implementation Constraints

- **Fully local** — no external network requests of any kind.
- **Never mutate `raw_df`** — all filtering operates on copies.
- **Two-layer filtering** — analysis pages always use `analysis_df`.
- **Normality test guards** — Shapiro-Wilk never runs at n ≥ 5,000 (enforced in `stats.py`); skipped tests are surfaced with a reason.
- **Performance guards** — Pairplot and regression require explicit button presses; sampling for n > 100k with notification; pairplot capped at 8 columns.
- **Inferential output** — `statsmodels` for regression, never sklearn. Mediation via Pingouin (import wrapped in try/except).
- **Post-hoc gating** — Tukey HSD only runs when omnibus F p < 0.05.
- **Errors surface to the user** — no silent except-pass; all errors render with a plain-English message.
- **No hardcoded column names** — all logic is dynamic on the uploaded dataset.
- **No `st.experimental_*` APIs** and no top-level `st.navigation()` — uses the `pages/` directory convention for routing.

Out of scope: authentication, ML beyond OLS (no clustering, classification, dimensionality reduction), auto-running heavy operations on page load.

## Education Enhancement (`core/explanations.py`)

Centralizes educational content so the dashboard doubles as a learning tool:

- `STAT_GLOSSARY` — plain-English explanations for statistical terms (mean, skewness, p-value, R², VIF, etc.).
- `ASSUMPTIONS_BY_TEST` — required assumptions per test name.
- `ASSUMPTION_CHECKS` — meaning, how to check, and how to fix each assumption.
- `RESULT_TEMPLATES` — interpretation templates per test.
- Helpers: `get_glossary_entry`, `get_assumption_warning`, `get_result_interpretation`.

All educational content is rendered inside collapsible expanders (progressive disclosure) so it does not crowd advanced users. Pre-test sections show the assumption checklist; post-test sections show plain-English interpretation and effect-size guidance. Statistical Tests page includes a decision-tree expander for test selection.

## CSV Export

Result downloads include a metadata header (test name, description, assumptions, interpretation) prepended to the CSV so an exported file is interpretable on its own.

## Reproducible Code Export (R & Python)

Every analysis result downloads as a standalone **R** and **Python** script that reproduces it on the student's own machine. The dashboard always computes in Python; R is an export-only artifact — **no R runtime is required to run the app**, and `requirements.txt` is unchanged.

- `core/codegen.py` — a pure module (no Streamlit import, so it stays unit-testable). Given a `Provenance` (the data file plus the active filter / study / group selections) and a list of analysis specs, it emits standalone `.py` and `.R` script strings. Each script replays the filter chain so it reproduces the *analysed subset*, not the raw upload, and exposes a `DATA_FILE` constant the student points at their own file. Column names are emitted with explicit indexing so names containing spaces survive.
- `core/export_ui.py` — Streamlit glue: `render_export()` renders the per-result download expander on Pages 3–6; `render_session_export()` (in the sidebar) emits one cumulative script for the whole session, driven by `session_state["analysis_log"]`.

**Accuracy tiers.** Tier 1 analyses (descriptives, outliers, normality, correlation, t-tests, one-way & two-way ANOVA, OLS) reproduce the dashboard exactly. Tier 2 analyses (repeated-measures ANOVA, mediation, multilevel mediation) are canonical in R but diverge slightly — `lme4` / `lmerTest` vs statsmodels `mixedlm`, and bootstrap RNG differing across languages — so each Tier 2 R block carries a header `NOTE` and a fixed seed.

Generated scripts stay standalone — they never import the app's `core` package; templates inline the scipy / statsmodels / R calls. `tests/test_codegen.py` executes every generated Python script against a fixture dataset to guard against template drift.
