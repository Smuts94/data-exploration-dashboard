# EDA Dashboard

[![Tests](https://github.com/Smuts94/data-exploration-dashboard/actions/workflows/tests.yml/badge.svg)](https://github.com/Smuts94/data-exploration-dashboard/actions/workflows/tests.yml)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=Smuts94/data-exploration-dashboard&branch=main&mainModule=app.py)

A locally hosted, statistically rigorous exploratory data analysis dashboard built for PhD research. Upload a dataset and immediately work through descriptive stats, correlation, regression, and formal hypothesis testing — with assumption checks and effect sizes throughout.

## Features

- **Upload** CSV / TSV / XLSX with auto-detected delimiter and encoding; override inferred column types
- **Variable Selection** — keep top N by variance or pick manually before any analysis
- **Filtering** — range sliders, multiselects, date pickers; live row count and CSV export
- **Descriptive Stats** — mean / median / SD / skew / kurtosis / IQR / CV; histogram + KDE; Q-Q plot; four normality tests (Shapiro-Wilk, D'Agostino-Pearson, Kolmogorov-Smirnov, Anderson-Darling) with sample-size guards; outlier detection (IQR + Z-score)
- **Correlation** — annotated Pearson / Spearman / Kendall heatmap with significance stars; scatter explorer; Seaborn pairplot
- **Regression** — `statsmodels` OLS with full inferential output; coefficient plot with 95% CIs; VIF table; four residual diagnostic plots
- **Statistical Tests** — t-tests (independent / paired / one-sample) with Levene + Welch + non-parametric fallbacks; one-way / two-way / repeated-measures ANOVA with Tukey HSD; mediation (Pingouin) and multilevel 2-1-1 mediation with bootstrap CIs

See [`DESIGN.md`](DESIGN.md) for architecture, state management, and per-page design rationale.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

## Run tests

```bash
pip install pytest
python -m pytest tests/
```

## Deploy to Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app** → pick `Smuts94/data-exploration-dashboard`, branch `main`, main file `app.py`.
3. In **Advanced settings** set Python version to **3.12** (3.13 also works).
4. Click **Deploy**.

The first build takes ~3–5 minutes (installs `scipy`, `statsmodels`, `pingouin`). Every push to `main` redeploys automatically.

## Tech stack

Streamlit · pandas · NumPy · SciPy · statsmodels · Pingouin · Plotly · Seaborn · matplotlib

## Constraints

Fully local by design — no external API calls, no telemetry, no authentication. `raw_df` is never mutated; all analysis runs on copies via a two-layer filter model (`filtered_df` → `analysis_df`).
