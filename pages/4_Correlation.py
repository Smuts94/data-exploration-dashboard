import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.state import (
    init_state, require_upload, require_nonempty_filtered,
    get_filtered_df, get_col_types, get_group_col, get_selected_vars,
)
from core.sidebar import render_sidebar
from core.stats import correlation_matrix, pvalue_matrix
from core.plots import correlation_heatmap, scatter_plot, pairplot_image
from core.group_utils import universal_filter, local_group_selector, render_group_layout, download_csv
from core.explanations import interpret_correlation_result
from core.export_ui import render_export
import scipy.stats as scipy_stats

st.set_page_config(page_title="Correlation", layout="wide")
init_state()

st.title("4 · Bivariate & Correlation Analysis")

if not require_upload():
    st.stop()
if not require_nonempty_filtered():
    st.stop()

filtered_df = get_filtered_df()
col_types   = get_col_types()

analysis_df, selected_groups = render_sidebar(filtered_df, col_types)

if analysis_df.empty:
    st.error("No data remaining after applying the current filters.")
    st.stop()

group_col     = get_group_col()
selected_vars = get_selected_vars()

all_numeric  = [c for c, t in col_types.items() if t == "Numeric"     and c in analysis_df.columns]
numeric_cols = [c for c in selected_vars if c in analysis_df.columns] if selected_vars else all_numeric
cat_cols     = [c for c, t in col_types.items() if t == "Categorical" and c in analysis_df.columns]

if len(numeric_cols) < 2:
    st.warning("Need at least 2 numeric columns for correlation analysis.")
    st.stop()

# ---------------------------------------------------------------------------
# Local selector key registry
# ---------------------------------------------------------------------------
LOCAL_VAR_KEYS = ["scatter_x", "scatter_y"]
LOCAL_GRP_KEYS = ["corr_hm_grp", "scatter_grp", "pairplot_grp", "grpstat_grp"]

# ---------------------------------------------------------------------------
# Universal page-level filter
# ---------------------------------------------------------------------------
default_var, default_grp_col = universal_filter(
    analysis_df, col_types,
    page_prefix="p2",
    local_var_keys=LOCAL_VAR_KEYS,
    local_grp_keys=LOCAL_GRP_KEYS,
)


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------
def _corr_heatmap(df, label, cols, method):
    if len(df) < 3:
        st.warning(f"Too few rows ({len(df)}) for group **{label}**.")
        return
    corr  = correlation_matrix(df, cols, method)
    pvals = pvalue_matrix(df, cols, method)
    st.plotly_chart(correlation_heatmap(corr, pvals), use_container_width=True)
    st.caption(f"n = {len(df):,}" + (f" — **{label}**" if label else ""))
    with st.expander("Raw correlation values"):
        corr_export = correlation_matrix(df, cols, method)
        st.dataframe(corr_export.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1).format("{:.4f}"),
                     use_container_width=True)
        download_csv(corr_export.reset_index(), f"Download correlation matrix{' — ' + label if label else ''}", f"corr_matrix{'_' + label if label else ''}.csv", key=f"dl_corr{'_' + label if label else ''}")
    with st.expander("P-value matrix"):
        pval_export = pvalue_matrix(df, cols, method)
        st.dataframe(pval_export.style.format("{:.4f}"), use_container_width=True)
        download_csv(pval_export.reset_index(), f"Download p-value matrix{' — ' + label if label else ''}", f"pval_matrix{'_' + label if label else ''}.csv", key=f"dl_pval{'_' + label if label else ''}")


def _scatter(df, label, x_col, y_col, color_arg, size_arg):
    sub = df[[x_col, y_col]].dropna()
    if len(sub) >= 3:
        r, p = scipy_stats.pearsonr(sub[x_col], sub[y_col])
        st.caption(
            f"Pearson r = **{r:.4f}** · p = **{p:.4e}**"
            + (f" — **{label}**" if label else "")
        )
        
        # Add interpretation
        with st.expander("💡 Interpretation", expanded=False):
            interpretation = interpret_correlation_result(r, p, method="Pearson")
            st.markdown(interpretation)
    
    st.plotly_chart(scatter_plot(df, x_col, y_col, color_arg, size_arg),
                    use_container_width=True)


# ===========================================================================
# Correlation Matrix
# ===========================================================================
st.header("Correlation Matrix")

with st.expander("📚 Choosing a Correlation Method", expanded=False):
    st.markdown("""
### Which Correlation Method Should I Use?

**Pearson Correlation**
- Measures **linear** relationships between continuous variables
- Assumes data are approximately **normally distributed**
- Sensitive to outliers
- **Use when:** Data is roughly normal and you expect a linear relationship

**Spearman Correlation**
- **Rank-based** (doesn't assume normality)
- Measures **monotonic** relationships (doesn't have to be linear)
- Robust to outliers
- **Use when:** Data is non-normal or you have outliers

**Kendall's τ (Tau)**
- Another rank-based correlation
- Similar to Spearman but often preferred for **smaller samples**
- More computationally intensive
- **Use when:** Sample size < 50 or for robustness

### Interpreting Correlation Strength

- **|r| < 0.3:** Weak correlation
- **0.3 ≤ |r| < 0.7:** Moderate correlation
- **|r| ≥ 0.7:** Strong correlation

**Sign:** 
- **Positive (+):** As X increases, Y tends to increase
- **Negative (-):** As X increases, Y tends to decrease

**Important:** Correlation ≠ Causation! Just because X and Y are correlated doesn't mean X causes Y.
    """)

method = st.radio("Correlation method", ["Pearson", "Spearman", "Kendall"],
                   horizontal=True, key="corr_method")

group_col_corr, group_vals_corr, subsets_corr = local_group_selector(
    analysis_df, col_types, key="corr_hm_grp", default_col=default_grp_col
)
if group_col_corr:
    render_group_layout(
        group_vals_corr, subsets_corr, _corr_heatmap,
        cols=numeric_cols, method=method,
    )
else:
    _corr_heatmap(analysis_df, "", numeric_cols, method)

st.caption("Stars: * p<0.05 · ** p<0.01 · *** p<0.001. Diagonal excluded.")

render_export("correlation", {"columns": numeric_cols, "method": method}, key="exp_corr")

# ===========================================================================
# Scatter Plot Explorer
# ===========================================================================
st.header("Scatter Plot Explorer")

_xi = numeric_cols.index(default_var) if default_var in numeric_cols else 0
_yi_opts = [c for c in numeric_cols if c != numeric_cols[_xi]] or numeric_cols
_yi = _yi_opts.index(default_var) if default_var in _yi_opts else 0

c1, c2, c3, c4 = st.columns(4)
x_col = c1.selectbox("Variable of interest (X axis)", numeric_cols, index=_xi, key="scatter_x")
y_col = c2.selectbox("Variable of interest (Y axis)",
                     [c for c in numeric_cols if c != x_col] or numeric_cols,
                     key="scatter_y")

color_options     = ["—"] + cat_cols
color_default     = group_col if (group_col and group_col in cat_cols and len(selected_groups) >= 2) else "—"
color_default_idx = color_options.index(color_default) if color_default in color_options else 0
color_col = c3.selectbox("Color encoding (optional)", color_options,
                          index=color_default_idx, key="scatter_color")
size_col  = c4.selectbox("Size encoding (optional)", ["—"] + numeric_cols, key="scatter_size")

color_arg = None if color_col == "—" else color_col
size_arg  = None if size_col  == "—" else size_col

group_col_scat, group_vals_scat, subsets_scat = local_group_selector(
    analysis_df, col_types, key="scatter_grp", default_col=default_grp_col
)
if group_col_scat:
    render_group_layout(
        group_vals_scat, subsets_scat, _scatter,
        x_col=x_col, y_col=y_col, color_arg=color_arg, size_arg=size_arg,
    )
else:
    _scatter(analysis_df, "", x_col, y_col, color_arg, size_arg)

# ===========================================================================
# Pairplot
# ===========================================================================
st.header("Pairplot")
MAX_PAIR_COLS = 8
if len(numeric_cols) > MAX_PAIR_COLS:
    st.warning(f"More than {MAX_PAIR_COLS} numeric columns — selection capped at {MAX_PAIR_COLS}.")

pair_cols = st.multiselect(
    "Variables of interest (max 8)",
    numeric_cols,
    default=numeric_cols[:min(5, len(numeric_cols))],
    key="pair_cols",
)

group_col_pair, group_vals_pair, subsets_pair = local_group_selector(
    analysis_df, col_types, key="pairplot_grp", default_col=default_grp_col
)

if len(pair_cols) > MAX_PAIR_COLS:
    st.error(f"Please select at most {MAX_PAIR_COLS} columns.")
elif len(pair_cols) < 2:
    st.info("Select at least 2 variables.")
else:
    SAMPLE_THRESHOLD = 100_000

    def _pairplot(df, label):
        plot_df = df
        if len(df) > SAMPLE_THRESHOLD:
            plot_df = df.sample(10_000, random_state=42)
            st.info(f"Group '{label}': {len(df):,} rows — sampled 10,000.")
        with st.spinner(f"Rendering pairplot{' for ' + label if label else ''}…"):
            try:
                img_bytes = pairplot_image(plot_df, pair_cols)
                st.image(img_bytes, use_column_width=True)
            except Exception as e:
                st.error(f"Could not render pairplot: {e}")

    if group_col_pair:
        if st.button("Generate Pairplot", type="primary", key="gen_pair"):
            render_group_layout(group_vals_pair, subsets_pair, _pairplot)
    else:
        plot_df = analysis_df
        if len(analysis_df) > SAMPLE_THRESHOLD:
            plot_df = analysis_df.sample(10_000, random_state=42)
            st.info(f"Dataset has {len(analysis_df):,} rows — sampled 10,000 for pairplot.")
        hue_col = group_col if (group_col and group_col in plot_df.columns and len(selected_groups) >= 1) else None
        if st.button("Generate Pairplot", type="primary", key="gen_pair_all"):
            with st.spinner("Rendering pairplot…"):
                try:
                    img_bytes = pairplot_image(plot_df, pair_cols, hue_col=hue_col)
                    st.image(img_bytes, use_column_width=True)
                    if hue_col:
                        st.caption(f"Coloured by **{hue_col}**")
                except Exception as e:
                    st.error(f"Could not render pairplot: {e}")

# ===========================================================================
# Grouped Statistics
# ===========================================================================
st.header("Grouped Statistics")
if not cat_cols:
    st.info("No categorical columns available for grouping.")
else:
    default_group = group_col if group_col in cat_cols else cat_cols[0]
    grp = st.selectbox("Group by", cat_cols,
                       index=cat_cols.index(default_group),
                       key="group_col_sel")

    def _grp_stats(df, label):
        try:
            pivot = (
                df.groupby(grp)[numeric_cols]
                .agg(["mean", "median"])
                .round(4)
            )
            pivot.columns = [f"{col} ({agg})" for col, agg in pivot.columns]
            st.dataframe(pivot, use_container_width=True)
            suffix = f"_{label}" if label else ""
            download_csv(pivot.reset_index(), f"Download grouped stats{' — ' + label if label else ''}", f"grouped_stats{suffix}.csv", key=f"dl_grpstat{suffix}")
        except Exception as e:
            st.error(f"Could not compute grouped statistics: {e}")

    group_col_gs, group_vals_gs, subsets_gs = local_group_selector(
        analysis_df, col_types, key="grpstat_grp", default_col=default_grp_col
    )
    if group_col_gs:
        render_group_layout(group_vals_gs, subsets_gs, _grp_stats)
    else:
        _grp_stats(analysis_df, "")
