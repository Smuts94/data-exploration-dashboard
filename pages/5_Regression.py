import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.state import (
    init_state, require_upload, require_nonempty_filtered,
    get_filtered_df, get_col_types, get_selected_vars,
)
from core.sidebar import render_sidebar
from core.stats import compute_vif
from core.plots import residual_diagnostic_plots, coefficient_plot
from core.group_utils import universal_filter, local_group_selector, render_group_layout, download_csv
from core.explanations import STAT_GLOSSARY, ASSUMPTIONS

import statsmodels.api as sm

st.set_page_config(page_title="Regression", layout="wide")
init_state()

st.title("5 · Linear Regression (OLS)")

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
all_numeric   = [c for c, t in col_types.items() if t == "Numeric" and c in analysis_df.columns]
numeric_cols  = [c for c in selected_vars if c in analysis_df.columns] if selected_vars else all_numeric

if len(numeric_cols) < 2:
    st.warning("Need at least 2 numeric columns to run a regression.")
    st.stop()

# ---------------------------------------------------------------------------
# Local selector key registry
# ---------------------------------------------------------------------------
LOCAL_VAR_KEYS = ["reg_y", "reg_x"]
LOCAL_GRP_KEYS = ["reg_grp"]

# ---------------------------------------------------------------------------
# Universal page-level filter
# ---------------------------------------------------------------------------
default_var, default_grp_col = universal_filter(
    analysis_df, col_types,
    page_prefix="p4",
    local_var_keys=LOCAL_VAR_KEYS,
    local_grp_keys=LOCAL_GRP_KEYS,
)

# ---------------------------------------------------------------------------
# Variable selection
# ---------------------------------------------------------------------------
st.subheader("Variable Selection")
_y_idx = numeric_cols.index(default_var) if default_var in numeric_cols else 0

c1, c2, c3 = st.columns([1, 2, 1])
y_col = c1.selectbox("Dependent variable (Y)", numeric_cols, index=_y_idx, key="reg_y")
x_options = [c for c in numeric_cols if c != y_col]
x_cols = c2.multiselect(
    "Independent variables (X)",
    x_options,
    default=x_options[:min(3, len(x_options))],
    key="reg_x",
)
include_const = c3.checkbox("Include intercept", value=True, key="reg_const")

if not x_cols:
    st.info("Select at least one independent variable.")
    st.stop()

# ---------------------------------------------------------------------------
# Per-group split selector
# ---------------------------------------------------------------------------
st.markdown("**Run regression per group** _(optional)_")
group_col_reg, group_vals_reg, subsets_reg = local_group_selector(
    analysis_df, col_types, key="reg_grp", default_col=default_grp_col
)

# ---------------------------------------------------------------------------
# Run button
# ---------------------------------------------------------------------------
with st.expander("📋 Regression Assumptions", expanded=False):
    st.markdown("""
### Linear Regression Assumptions (Check These!)

1. **Linearity:** Relationship between X and Y should be linear
   - Check: Scatter plots, residuals vs fitted plot
   
2. **Independence:** Observations should be independent
   - Check: Study design (are rows truly independent?)
   
3. **Homoscedasticity:** Residuals should have constant variance
   - Check: Residuals vs fitted plot (should show random scatter, not funnel)
   
4. **Normality of Residuals:** Residuals should be approximately normal
   - Check: Q-Q plot (points on diagonal = good), histogram of residuals
   
5. **No Multicollinearity:** Predictors shouldn't be highly correlated
   - Check: VIF values (all < 5 is good)
   - Check: Correlation matrix of predictors

**What if assumptions are violated?**
- **Non-linearity:** Transform variables or use polynomial regression
- **Non-normal residuals:** Use robust standard errors or transformations
- **Heteroscedasticity:** Use weighted least squares or robust SEs
- **Multicollinearity:** Remove correlated predictors or use regularization
    """)

if not st.button("Run Regression", type="primary"):
    st.info("Configure variables above and click **Run Regression**.")
    st.stop()


# ---------------------------------------------------------------------------
# OLS runner
# ---------------------------------------------------------------------------
def _run_ols(df: pd.DataFrame, label: str = "") -> None:
    tag = f" — **{label}**" if label else ""

    try:
        sub = df[[y_col] + x_cols].dropna()
        if sub.empty:
            st.error(f"No complete rows after dropping NaNs{tag}.")
            return
        if len(sub) <= len(x_cols) + 1:
            st.warning(
                f"Too few observations (n={len(sub)}) for {len(x_cols)} predictors{tag}. Skipping."
            )
            return

        Y = sub[y_col]
        X = sub[x_cols]
        if include_const:
            X = sm.add_constant(X, has_constant="add")

        result = sm.OLS(Y, X).fit()
    except np.linalg.LinAlgError as e:
        st.error(
            f"Singular matrix{tag} — design matrix not invertible. "
            f"Two predictors may be perfectly collinear.\n\nDetails: {e}"
        )
        return
    except Exception as e:
        st.error(f"Regression failed{tag}: {e}")
        return

    st.success(f"n = {int(result.nobs):,}{tag}")
    _tag_safe = label.replace(" ", "_") if label else "all"

    st.subheader("Model Summary Metrics")
    cond_num = float(result.condition_number)
    mcols = st.columns(7)
    mcols[0].metric("R²",            f"{result.rsquared:.4f}")
    mcols[1].metric("Adj. R²",       f"{result.rsquared_adj:.4f}")
    mcols[2].metric("F p-value",     f"{result.f_pvalue:.4e}")
    mcols[3].metric("AIC",           f"{result.aic:.2f}")
    mcols[4].metric("BIC",           f"{result.bic:.2f}")
    mcols[5].metric("Durbin-Watson", f"{sm.stats.stattools.durbin_watson(result.resid):.4f}")
    mcols[6].metric("Condition #",   f"{cond_num:.1f}")
    
    # Add interpretation guide
    with st.expander("💡 Understanding These Metrics", expanded=False):
        st.markdown(f"""
### Model Performance Metrics

**R² = {result.rsquared:.4f}**
- Proportion of variance in Y explained by your model (0–1)
- R² = 0.7 means your model explains 70% of the variation
- Higher is better, but context matters
- Warning: R² always increases when you add predictors (even useless ones)

**Adjusted R² = {result.rsquared_adj:.4f}**
- Penalizes adding more predictors (avoids overfitting)
- Use this to compare models with different numbers of predictors
- Can decrease even if R² increases

**F p-value = {result.f_pvalue:.4e}**
- Tests if your model as a whole is significant
- p < 0.05 → model explains significant variation
- If p ≥ 0.05 → your predictors don't explain variation

**AIC = {result.aic:.2f} and BIC = {result.bic:.2f}**
- Compare models: **lower is better**
- Balance between fit and complexity
- Useful for comparing non-nested models

**Durbin-Watson = {sm.stats.stattools.durbin_watson(result.resid):.4f}**
- Tests for autocorrelation in residuals
- Range: 0–4, value ≈ 2 is good
- < 2: positive autocorrelation, > 2: negative autocorrelation
- Usually matters for time series data

**Condition Number = {cond_num:.1f}**
- Measures multicollinearity
- < 30: OK, 30–100: moderate concern, > 100: severe
        """)

    if cond_num > 30:
        st.warning(
            f"Condition number = {cond_num:.1f} > 30 — potential multicollinearity. "
            "Interpret coefficients with caution."
        )

    with st.expander("Full OLS Summary"):
        st.code(str(result.summary()), language="text")
        coef_export = result.summary2().tables[1].reset_index().rename(columns={"index": "Predictor"})
        download_csv(coef_export, f"Download coefficients{tag}", f"coef_{_tag_safe}.csv", key=f"dl_coef_{_tag_safe}")

    st.subheader("Coefficient Plot (95% CI)")
    st.plotly_chart(coefficient_plot(result), use_container_width=True)

    st.subheader("Variance Inflation Factors (VIF)")
    vif_df = compute_vif(sub[x_cols])

    def _vif_flag(v):
        if pd.isna(v): return "—"
        if v > 10:     return "SEVERE (>10)"
        if v > 5:      return "Warning (>5)"
        return "OK"

    vif_df["Status"] = vif_df["VIF"].apply(_vif_flag)
    st.dataframe(vif_df, use_container_width=True, hide_index=True)
    
    # Add interpretation
    with st.expander("💡 Understanding VIF", expanded=False):
        st.markdown("""
### Variance Inflation Factor (VIF)

**What it measures:** How much multicollinearity inflates the variance of a coefficient

**Interpretation:**
- **VIF = 1:** No correlation with other predictors (ideal)
- **1 < VIF < 5:** Acceptable multicollinearity
- **VIF = 5–10:** Moderate concern; consider removing predictors
- **VIF > 10:** Severe multicollinearity; predictors are highly correlated

**What to do if VIF is high:**
1. **Investigate:** Are predictors measuring the same thing?
2. **Remove:** Drop one of the correlated predictors
3. **Combine:** Create an index or principal component
4. **Accept:** Sometimes necessary; interpret with caution

**Example:** If VIF(X₁) = 6, then the variance of X₁'s coefficient is 6 times 
larger than if X₁ were uncorrelated with other predictors.
        """)
    
    _tag_safe = label.replace(" ", "_") if label else "all"
    download_csv(vif_df, f"Download VIF table{tag}", f"vif_{_tag_safe}.csv", key=f"dl_vif_{_tag_safe}")

    severe = vif_df[vif_df["VIF"] > 10]
    warned = vif_df[(vif_df["VIF"] > 5) & (vif_df["VIF"] <= 10)]
    if not severe.empty:
        st.error(f"Severe multicollinearity (VIF > 10): {', '.join(severe['Predictor'].tolist())}")
    elif not warned.empty:
        st.warning(f"Moderate multicollinearity (VIF > 5): {', '.join(warned['Predictor'].tolist())}")

    st.subheader("Residual Diagnostic Plots")
    
    with st.expander("📚 Reading Residual Plots", expanded=False):
        st.markdown("""
### Understanding the Four Diagnostic Plots

**1. Residuals vs Fitted**
- **What it shows:** Residuals (prediction errors) vs predicted values
- **Look for:** Random scatter around zero
- **Red flag:** Curved pattern (suggests non-linearity) or funnel shape (heteroscedasticity)
- **Good:** Points randomly scattered, horizontal red line is flat

**2. Q-Q Plot (Normal Probability)**
- **What it shows:** How closely residuals follow a normal distribution
- **Look for:** Points close to the diagonal line
- **Red flag:** Points deviate significantly from the line (esp. at the tails = outliers)
- **Good:** Points follow the line closely

**3. Scale-Location (√|Residuals| vs Fitted)**
- **What it shows:** Whether residuals have constant variance
- **Look for:** Horizontal red line with even scatter
- **Red flag:** Line slopes up or down (heteroscedasticity) or funnel shape
- **Good:** Flat line with roughly equal scatter

**4. Residuals vs Leverage (Cook's Distance)**
- **What it shows:** Influential outliers and high-leverage points
- **Look for:** Most points near zero, within dashed lines (Cook's distance ≈ 0.5 or 1)
- **Red flag:** Points outside dashed lines (influential outliers)
- **Good:** No points far outside contours, no high-leverage outliers

**If violations are present:**
- **Non-linearity:** Transform X or Y, add polynomial terms
- **Heteroscedasticity:** Use robust SEs or weighted LS
- **Non-normal residuals:** Transform Y or use robust methods
        """)
    
    try:
        st.plotly_chart(residual_diagnostic_plots(result), use_container_width=True)
    except Exception as e:
        st.error(f"Could not render diagnostic plots: {e}")


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
if group_col_reg:
    render_group_layout(group_vals_reg, subsets_reg, _run_ols)
else:
    _run_ols(analysis_df)
