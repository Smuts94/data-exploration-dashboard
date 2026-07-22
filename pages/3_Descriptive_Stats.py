import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.state import (
    init_state, require_upload, require_nonempty_filtered,
    get_filtered_df, get_col_types, get_selected_vars,
)
from core.sidebar import render_sidebar
from core.stats import (
    descriptive_table, normality_tests,
    outlier_summary, get_iqr_outlier_rows, get_zscore_outlier_rows,
)
from core.plots import histogram_kde, qq_plot, categorical_bar
from core.group_utils import universal_filter, local_group_selector, render_group_layout, download_csv
from core.explanations import STAT_GLOSSARY, interpret_normality_tests
from core.export_ui import render_export

st.set_page_config(page_title="Descriptive Stats", layout="wide")
init_state()

st.title("3 · Descriptive Stats")

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

selected_vars = get_selected_vars()
all_numeric   = [c for c, t in col_types.items() if t == "Numeric"      and c in analysis_df.columns]
numeric_cols  = [c for c in selected_vars if c in analysis_df.columns] if selected_vars else all_numeric
cat_cols      = [c for c, t in col_types.items() if t == "Categorical"  and c in analysis_df.columns]

# ---------------------------------------------------------------------------
# Local selector key registry — used by universal_filter cascade
# ---------------------------------------------------------------------------
LOCAL_VAR_KEYS = ["uni_dist_col", "uni_norm_col", "uni_out_col"]
LOCAL_GRP_KEYS = ["uni_desc_grp", "uni_dist_grp", "uni_norm_grp",
                  "uni_out_grp",  "uni_outrow_grp", "uni_cat_grp"]

# ---------------------------------------------------------------------------
# Universal page-level filter
# ---------------------------------------------------------------------------
if numeric_cols:
    default_var, default_grp_col = universal_filter(
        analysis_df, col_types,
        page_prefix="p1",
        local_var_keys=LOCAL_VAR_KEYS,
        local_grp_keys=LOCAL_GRP_KEYS,
    )
else:
    default_var, default_grp_col = None, None


# ===========================================================================
# Numeric section
# ===========================================================================
if numeric_cols:
    st.header("Numeric Columns")

    # ── Descriptive Stats ────────────────────────────────────────────────────
    st.subheader("Descriptive Statistics")

    with st.expander("📚 Understanding Descriptive Statistics", expanded=False):
        st.markdown("""
### What Do These Statistics Mean?

**Measures of Center:**
- **Mean:** Average of all values. Sensitive to outliers.
- **Median:** Middle value when sorted. Robust to outliers (use if data is skewed).
- **Mode:** Most frequently occurring value.

**Measures of Spread:**
- **Std (SD):** Standard deviation. Shows how spread out the data is. 
  - About 68% of data falls within 1 SD of the mean (for normal data).
  - Larger SD = more variability.
  
- **Variance:** SD squared. Used in many statistical formulas.

- **IQR (Interquartile Range):** Q3 - Q1. Contains the middle 50% of data.
  - Robust to outliers (unlike SD).
  - Used for outlier detection.

- **CV (Coefficient of Variation):** SD / mean. Standardized measure of spread.
  - Useful for comparing variability across variables with different scales.

**Measures of Shape:**
- **Skewness:** Measures asymmetry.
  - Skewness ≈ 0: symmetric
  - Skewness > 1: right-skewed (tail to right)
  - Skewness < -1: left-skewed (tail to left)
  - |Skewness| > 1 suggests **non-normal distribution**

- **Kurtosis:** Measures tail heaviness. This table reports **excess kurtosis**
  (0 = normal, not the raw/Pearson convention where 3 = normal).
  - Excess kurtosis ≈ 0: normal distribution
  - Excess kurtosis > 0: heavy tails (watch for outliers)
  - Excess kurtosis > 1 suggests potential outliers

**Key Takeaway:** 
If your data is skewed or has high kurtosis, consider transformations or non-parametric tests.
        """)


    def _desc_table(df, label):
        desc = descriptive_table(df, numeric_cols)
        if desc.empty:
            st.info(f"No data for group **{label}**.")
        else:
            st.dataframe(desc.style.format(precision=4), use_container_width=True)
            suffix = f"_{label}" if label else ""
            download_csv(desc, f"Download descriptive stats{' — ' + label if label else ''}", f"descriptive_stats{suffix}.csv", key=f"dl_desc{suffix}")

    group_col_d, group_vals_d, subsets_d = local_group_selector(
        analysis_df, col_types, key="uni_desc_grp", default_col=default_grp_col
    )
    if group_col_d:
        render_group_layout(group_vals_d, subsets_d, _desc_table)
    else:
        _desc_table(analysis_df, "")

    render_export("descriptive_stats", {"columns": numeric_cols}, key="exp_desc")

    # ── Distribution & Q-Q ───────────────────────────────────────────────────
    st.subheader("Distribution Plot")

    # Default index for "Variable of interest" selectbox
    _var_idx = numeric_cols.index(default_var) if default_var in numeric_cols else 0
    selected_col = st.selectbox(
        "Variable of interest",
        numeric_cols,
        index=_var_idx,
        key="uni_dist_col",
    )

    group_col_dist, group_vals_dist, subsets_dist = local_group_selector(
        analysis_df, col_types, key="uni_dist_grp", default_col=default_grp_col
    )

    if group_col_dist:
        n_grp = len(group_vals_dist)
        if n_grp == 2:
            col_l, col_r = st.columns(2)
            for col_ctx, lbl, sub in zip([col_l, col_r], group_vals_dist, subsets_dist):
                with col_ctx:
                    st.caption(f"**{lbl}**")
                    st.plotly_chart(
                        histogram_kde(sub[selected_col], selected_col),
                        use_container_width=True,
                    )
                    st.plotly_chart(
                        qq_plot(sub[selected_col], f"Q-Q — {selected_col}"),
                        use_container_width=True,
                    )
        else:
            st.plotly_chart(
                histogram_kde(
                    analysis_df[selected_col], selected_col,
                    group_series=analysis_df[group_col_dist],
                    group_labels=group_vals_dist,
                ),
                use_container_width=True,
            )
            st.markdown("**Q-Q plots per group**")
            def _qq(df, label):
                st.plotly_chart(
                    qq_plot(df[selected_col], f"Q-Q — {selected_col} ({label})"),
                    use_container_width=True,
                )
            render_group_layout(group_vals_dist, subsets_dist, _qq)
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(histogram_kde(analysis_df[selected_col], selected_col),
                            use_container_width=True)
        with c2:
            st.plotly_chart(qq_plot(analysis_df[selected_col], selected_col),
                            use_container_width=True)

    # ── Normality Tests ──────────────────────────────────────────────────────
    st.subheader("Normality Tests")

    with st.expander("📚 Why Test for Normality?", expanded=False):
        st.markdown("""
### Understanding Normality Tests

Many statistical tests assume your data is **normally distributed** (bell-shaped):
- T-tests, ANOVA, and linear regression all rely on normality
- Violating this assumption can affect the validity of your conclusions

**Interpretation Guide:**
- **p-value > 0.05:** Data appears normally distributed (fail to reject null)
- **p-value < 0.05:** Data appears non-normally distributed (reject null)

**Tests Shown:**
- **Shapiro-Wilk** (best for n < 5,000): Most powerful normality test
- **D'Agostino-Pearson:** Tests skewness AND kurtosis together
- **Kolmogorov-Smirnov:** Compares data to a normal distribution
- **Anderson-Darling:** Emphasizes deviations in the tails

**What if Data is Non-Normal?**
1. **Transform:** Apply log, square root, or Box-Cox transformation
2. **Use non-parametric tests:** Mann-Whitney U (instead of t-test), Kruskal-Wallis (instead of ANOVA)
3. **Accept violation:** Some tests are robust to moderate normality violations, especially with large n

**Visual Checks:**
- **Histogram:** Should be roughly bell-shaped
- **Q-Q plot:** Points should lie close to the diagonal line
        """)


    _norm_idx = numeric_cols.index(default_var) if default_var in numeric_cols else 0
    norm_col = st.selectbox(
        "Variable of interest",
        numeric_cols,
        index=_norm_idx,
        key="uni_norm_col",
    )

    def _norm_block(df, label):
        n = df[norm_col].dropna().shape[0]
        if n == 0:
            st.info(f"No data for group **{label}**.")
            return
        if n >= 5000:
            st.info(f"n = {n:,} ≥ 5,000 — Shapiro-Wilk skipped.")
        norm_df = normality_tests(df[norm_col])
        st.dataframe(norm_df, use_container_width=True, hide_index=True)
        
        # Add interpretation
        with st.expander("💡 Interpretation", expanded=False):
            interpretation = interpret_normality_tests(norm_df, norm_col, n)
            st.markdown(interpretation)
        
        suffix = f"_{label}" if label else ""
        download_csv(norm_df, f"Download normality results{' — ' + label if label else ''}", f"normality_{norm_col}{suffix}.csv", key=f"dl_norm_{norm_col}{suffix}")

    group_col_norm, group_vals_norm, subsets_norm = local_group_selector(
        analysis_df, col_types, key="uni_norm_grp", default_col=default_grp_col
    )
    if group_col_norm:
        render_group_layout(group_vals_norm, subsets_norm, _norm_block)
    else:
        n = analysis_df[norm_col].dropna().shape[0]
        if n >= 5000:
            st.info(f"n = {n:,} ≥ 5,000 — Shapiro-Wilk will be skipped.")
        norm_df_all = normality_tests(analysis_df[norm_col])
        st.dataframe(norm_df_all, use_container_width=True, hide_index=True)
        
        # Add interpretation
        with st.expander("💡 Interpretation", expanded=False):
            interpretation = interpret_normality_tests(norm_df_all, norm_col, n)
            st.markdown(interpretation)
        
        download_csv(norm_df_all, "Download normality results", f"normality_{norm_col}.csv", key=f"dl_norm_{norm_col}_all")

    render_export("normality", {"column": norm_col}, key="exp_norm")

    # ── Outlier Summary ──────────────────────────────────────────────────────
    st.subheader("Outlier Summary")

    with st.expander("📚 Understanding Outliers", expanded=False):
        st.markdown("""
### What Are Outliers?

Outliers are data points that are unusually far from the rest of the data. 
They can result from:
- **Legitimate extreme values** (naturally rare but real)
- **Measurement errors** (should be removed)
- **Data entry mistakes** (should be corrected)

**Two Detection Methods:**

**1. IQR Method (Interquartile Range)**
- **Rule:** Values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR] are flagged
- **Advantage:** Robust, doesn't assume normality
- **Use when:** Data is skewed or non-normal

**2. Z-Score Method**
- **Rule:** Values with |z| > 3 are flagged (more than 3 SDs from mean)
- **Advantage:** Parametric, considers distribution
- **Limitation:** Assumes data is roughly normal
- **Use when:** Data is approximately normal

**What to Do With Outliers:**
1. **Investigate:** Are they real or errors?
2. **Keep/Report:** If legitimate, keep and report them
3. **Remove/Transform:** If errors, remove or use robust methods (median, IQR)
4. **Analyze Separately:** Run analysis with and without outliers to see impact

**Note:** Removing outliers changes your results. Always report both versions!
        """)


    def _outlier_table(df, label):
        out_df = outlier_summary(df, numeric_cols)
        st.dataframe(out_df, use_container_width=True, hide_index=True)
        suffix = f"_{label}" if label else ""
        download_csv(out_df, f"Download outlier summary{' — ' + label if label else ''}", f"outlier_summary{suffix}.csv", key=f"dl_out{suffix}")

    group_col_out, group_vals_out, subsets_out = local_group_selector(
        analysis_df, col_types, key="uni_out_grp", default_col=default_grp_col
    )
    if group_col_out:
        render_group_layout(group_vals_out, subsets_out, _outlier_table)
    else:
        _outlier_table(analysis_df, "")

    st.markdown("**Inspect outlier rows**")
    _out_idx = numeric_cols.index(default_var) if default_var in numeric_cols else 0
    outlier_col = st.selectbox(
        "Variable of interest",
        numeric_cols,
        index=_out_idx,
        key="uni_out_col",
    )
    method = st.radio("Detection method", ["IQR", "Z-score (|z| > 3)"],
                      horizontal=True, key="uni_out_method")

    def _outlier_rows(df, label):
        flagged = (get_iqr_outlier_rows(df, outlier_col) if method == "IQR"
                   else get_zscore_outlier_rows(df, outlier_col))
        st.caption(f"{len(flagged):,} flagged rows")
        if not flagged.empty:
            st.dataframe(flagged, use_container_width=True)
        else:
            st.success("No outliers detected with this method.")

    group_col_orow, group_vals_orow, subsets_orow = local_group_selector(
        analysis_df, col_types, key="uni_outrow_grp", default_col=default_grp_col
    )
    if group_col_orow:
        render_group_layout(group_vals_orow, subsets_orow, _outlier_rows)
    else:
        _outlier_rows(analysis_df, "")

    render_export("outliers", {"columns": numeric_cols}, key="exp_out")

else:
    st.info("No numeric columns found. Check column type settings on the Upload page.")


# ===========================================================================
# Categorical section
# ===========================================================================
if cat_cols:
    st.header("Categorical Columns")
    cat_sel = st.selectbox("Variable of interest", cat_cols, key="uni_cat_col")

    def _cat_block(df, label):
        s = df[cat_sel]
        n_unique = s.nunique()
        c1, c2 = st.columns(2)
        c1.metric("Mode", str(s.mode().iloc[0]) if not s.mode().empty else "—")
        c2.metric("Unique values", n_unique)
        if n_unique > 50:
            st.warning(f"High cardinality: {n_unique:,} unique values.")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(
                s.value_counts().rename_axis("Value").reset_index(name="Count"),
                use_container_width=True, hide_index=True,
            )
        with col2:
            st.plotly_chart(
                categorical_bar(s, f"{cat_sel}" + (f" — {label}" if label else "")),
                use_container_width=True,
            )

    group_col_cat, group_vals_cat, subsets_cat = local_group_selector(
        analysis_df, col_types, key="uni_cat_grp", default_col=default_grp_col
    )
    if group_col_cat:
        render_group_layout(group_vals_cat, subsets_cat, _cat_block)
    else:
        _cat_block(analysis_df, "")

elif not numeric_cols:
    st.info("No columns found. Please upload a dataset and configure column types on the Upload page.")
