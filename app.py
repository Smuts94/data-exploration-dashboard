import streamlit as st

st.set_page_config(
    page_title="EDA Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.theme import inject_theme, page_header, feature_card

inject_theme()

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("EDA Dashboard")
st.sidebar.caption("Local exploratory data analysis for PhD research.")

# ── Hero ───────────────────────────────────────────────────────────────────
page_header(
    eyebrow="EDA Dashboard",
    title="Statistically rigorous data exploration.",
    lede=(
        "A fully local Streamlit app for PhD research — descriptive stats, "
        "correlation, regression, and hypothesis testing without sending a "
        "single byte off your machine."
    ),
)

# ── Pages grid ─────────────────────────────────────────────────────────────
pages = [
    ("Step 0", "Data Upload",
     "Upload a CSV, TSV, or XLSX file. Inspect shape, types, and missingness, "
     "then designate the study and group columns that drive every analysis page."),
    ("Step 1", "Variable Selection",
     "Reduce the dataset to the columns you care about before any analysis runs. "
     "Quick-select by variance or pick manually."),
    ("Step 2", "Data Filter",
     "Subset rows with range sliders, multiselects, and date pickers. "
     "Live row count and filtered CSV export."),
    ("Step 3", "Descriptive Stats",
     "Per-variable summary statistics, distributions, normality tests "
     "(Shapiro / D'Agostino / KS / Anderson-Darling), and outlier drill-down."),
    ("Step 4", "Correlation",
     "Annotated heatmap with significance stars, scatter explorer, "
     "Seaborn pairplot, and grouped pivot tables."),
    ("Step 5", "Regression",
     "OLS via statsmodels — full inferential output, coefficient plot with 95% CIs, "
     "VIF, and four residual diagnostics."),
    ("Step 6", "Statistical Tests",
     "T-tests, ANOVA (one/two-way and repeated measures), and mediation analysis "
     "with assumption checks and non-parametric fallbacks."),
]

st.markdown("##### Pages")
for i in range(0, len(pages), 2):
    cols = st.columns(2, gap="medium")
    for col, item in zip(cols, pages[i : i + 2]):
        with col:
            feature_card(*item)

st.write("")
st.info("Open **0 · Data Upload** in the sidebar to get started, then walk through the pages in order.")
