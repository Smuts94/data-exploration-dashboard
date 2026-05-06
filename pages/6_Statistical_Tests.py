"""
Page 5 — Statistical Tests
T-Tests, ANOVA (one-way, two-way), and Mediation Analysis.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.state import init_state, require_upload, require_nonempty_filtered, get_filtered_df, get_col_types
from core.sidebar import render_sidebar
from core.group_utils import download_csv
from core.explanations import TEST_SELECTION_GUIDE

from core.stats import (
    run_independent_ttest,
    run_paired_ttest,
    run_onesample_ttest,
    run_oneway_anova,
    run_tukey_hsd,
    run_twoway_anova,
    run_rm_anova,
    run_mediation,
    run_multilevel_mediation,
)

st.set_page_config(page_title="Statistical Tests", layout="wide")
init_state()

st.title("6 · Statistical Tests")

with st.expander("📚 Test Selection Guide", expanded=False):
    st.markdown(TEST_SELECTION_GUIDE)

if not require_upload():
    st.stop()
if not require_nonempty_filtered():
    st.stop()

filtered_df = get_filtered_df()
col_types = get_col_types()

analysis_df, selected_groups = render_sidebar(filtered_df, col_types)

if analysis_df.empty:
    st.error("No data remaining after applying the current filters.")
    st.stop()

numeric_cols = [c for c, t in col_types.items() if t == "Numeric" and c in analysis_df.columns]
cat_cols = [c for c, t in col_types.items() if t == "Categorical" and c in analysis_df.columns]

if not numeric_cols:
    st.warning("No numeric columns available. Check your column type settings on the Upload page.")
    st.stop()


# ===========================================================================
# Helper utilities
# ===========================================================================

def _pval_label(p: float) -> str:
    if p < 0.001:
        return f"{p:.2e} ***"
    if p < 0.01:
        return f"{p:.4f} **"
    if p < 0.05:
        return f"{p:.4f} *"
    return f"{p:.4f}"


def _normality_violated(norm_df: pd.DataFrame) -> bool:
    """Return True if any valid normality test rejects H0."""
    for _, row in norm_df.iterrows():
        pv = row.get("p-value", None)
        if isinstance(pv, (float, int)) and not np.isnan(float(pv)):
            if float(pv) < 0.05:
                return True
    return False


# ===========================================================================
# Section 1 — T-Tests
# ===========================================================================

st.header("T-Tests")

ttest_type = st.radio(
    "Test type",
    ["Independent samples", "Paired samples", "One-sample"],
    horizontal=True,
    key="ttest_type",
)

# ---- Independent samples ---------------------------------------------------
if ttest_type == "Independent samples":
    if not cat_cols:
        st.warning("No categorical columns available for grouping.")
        st.stop()
    c1, c2 = st.columns(2)
    ind_val = c1.selectbox("Numeric variable", numeric_cols, key="ind_val")
    ind_grp = c2.selectbox("Grouping column (must have exactly 2 levels)", cat_cols, key="ind_grp")

    grp_counts = analysis_df[ind_grp].dropna().nunique()
    if grp_counts != 2:
        st.warning(
            f"'{ind_grp}' has {grp_counts} unique values in the current filter context. "
            "Please ensure exactly 2 groups are visible (use the sidebar or Page 3 filters)."
        )

    if st.button("Run Independent T-Test", type="primary", key="run_ind"):
        try:
            res = run_independent_ttest(analysis_df, ind_val, ind_grp)
        except Exception as e:
            st.error(f"T-test failed: {e}")
            st.stop()

        g0, g1 = res["group_labels"]
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Test", res["test_type"])
        col_b.metric("t-statistic", f"{res['t_stat']:.4f}")
        col_c.metric("p-value", _pval_label(res["p_value"]))
        col_d.metric("Cohen's d", f"{res['cohens_d']:.4f}")

        st.caption(
            f"95% CI on mean difference ({g0} − {g1}): "
            f"[{res['ci_low']:.4f}, {res['ci_high']:.4f}]   |   "
            f"df = {res['df']:.2f}"
        )

        direction = "greater" if res["means"][0] > res["means"][1] else "less"
        sig = res["p_value"] < 0.05
        st.info(
            f"**Interpretation**: The mean of '{g0}' ({res['means'][0]:.4f} ± {res['sds'][0]:.4f}) "
            f"is {'significantly' if sig else 'not significantly'} {direction} than "
            f"'{g1}' ({res['means'][1]:.4f} ± {res['sds'][1]:.4f}), "
            f"{res['test_type']} t({res['df']:.1f}) = {res['t_stat']:.4f}, "
            f"p = {res['p_value']:.4f}."
        )

        _ind_export = pd.DataFrame([{
            "Test": res["test_type"], "Column": ind_val, "Group col": ind_grp,
            "Group A": g0, "Group B": g1,
            "Mean A": res["means"][0], "SD A": res["sds"][0],
            "Mean B": res["means"][1], "SD B": res["sds"][1],
            "t-stat": res["t_stat"], "df": res["df"], "p-value": res["p_value"],
            "CI low": res["ci_low"], "CI high": res["ci_high"],
            "Cohen's d": res["cohens_d"], "Significant (α=0.05)": sig,
        }])
        download_csv(_ind_export, "Download t-test results", "ttest_independent.csv", key="dl_ttest_ind")

        if not res["equal_var"]:
            st.warning(
                f"Levene's test: F = {res['levene_stat']:.4f}, p = {res['levene_p']:.4f} — "
                "variances are unequal; Welch's correction applied automatically."
            )

        # Non-parametric
        norm_violated = (
            _normality_violated(res["normality_a"]) or
            _normality_violated(res["normality_b"])
        )
        if norm_violated:
            st.warning(
                "Normality assumption may be violated for one or both groups. "
                "Mann-Whitney U results are shown below as a non-parametric alternative."
            )
            st.markdown(
                f"**Mann-Whitney U**: U = {res['mwu_stat']:.2f}, "
                f"p = {_pval_label(res['mwu_p'])}"
            )

        with st.expander("Normality tests per group"):
            st.markdown(f"**{g0}** (n={res['n'][0]})")
            st.dataframe(res["normality_a"], hide_index=True, use_container_width=True)
            st.markdown(f"**{g1}** (n={res['n'][1]})")
            st.dataframe(res["normality_b"], hide_index=True, use_container_width=True)

# ---- Paired samples --------------------------------------------------------
elif ttest_type == "Paired samples":
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for a paired t-test.")
        st.stop()
    c1, c2 = st.columns(2)
    paired_col1 = c1.selectbox("Column 1", numeric_cols, key="paired_c1")
    paired_col2 = c2.selectbox("Column 2", [c for c in numeric_cols if c != paired_col1], key="paired_c2")

    if st.button("Run Paired T-Test", type="primary", key="run_paired"):
        try:
            res = run_paired_ttest(analysis_df, paired_col1, paired_col2)
        except Exception as e:
            st.error(f"Paired t-test failed: {e}")
            st.stop()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("n (pairs)", res["n"])
        c2.metric("t-statistic", f"{res['t_stat']:.4f}")
        c3.metric("p-value", _pval_label(res["p_value"]))
        c4.metric("Cohen's d", f"{res['cohens_d']:.4f}")

        st.caption(
            f"Mean difference: {res['mean_diff']:.4f} ± {res['sd_diff']:.4f}   |   "
            f"95% CI: [{res['ci_low']:.4f}, {res['ci_high']:.4f}]   |   "
            f"df = {res['df']:.0f}"
        )

        sig = res["p_value"] < 0.05
        st.info(
            f"**Interpretation**: The paired mean difference "
            f"({paired_col1} − {paired_col2}) is "
            f"{'statistically significant' if sig else 'not statistically significant'}, "
            f"t({res['df']:.0f}) = {res['t_stat']:.4f}, p = {res['p_value']:.4f}."
        )

        if _normality_violated(res["normality"]):
            st.warning(
                "Normality of differences may be violated. "
                "Wilcoxon signed-rank results shown as non-parametric alternative."
            )
            st.markdown(
                f"**Wilcoxon signed-rank**: W = {res['wilcoxon_stat']:.2f}, "
                f"p = {_pval_label(res['wilcoxon_p'])}"
            )

        with st.expander("Normality test on differences"):
            st.dataframe(res["normality"], hide_index=True, use_container_width=True)

# ---- One-sample ------------------------------------------------------------
else:
    c1, c2 = st.columns(2)
    os_col = c1.selectbox("Column", numeric_cols, key="os_col")
    mu0 = c2.number_input("Hypothesised mean (H₀: μ = ?)", value=0.0, key="os_mu0")

    if st.button("Run One-Sample T-Test", type="primary", key="run_os"):
        try:
            res = run_onesample_ttest(analysis_df, os_col, mu0)
        except Exception as e:
            st.error(f"One-sample t-test failed: {e}")
            st.stop()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("n", res["n"])
        c2.metric("t-statistic", f"{res['t_stat']:.4f}")
        c3.metric("p-value", _pval_label(res["p_value"]))
        c4.metric("Cohen's d", f"{res['cohens_d']:.4f}")

        st.caption(
            f"Sample mean: {res['sample_mean']:.4f}   |   H₀: μ = {mu0}   |   "
            f"95% CI: [{res['ci_low']:.4f}, {res['ci_high']:.4f}]   |   df = {res['df']:.0f}"
        )

        sig = res["p_value"] < 0.05
        st.info(
            f"**Interpretation**: The sample mean ({res['sample_mean']:.4f}) is "
            f"{'significantly' if sig else 'not significantly'} different from {mu0}, "
            f"t({res['df']:.0f}) = {res['t_stat']:.4f}, p = {res['p_value']:.4f}."
        )

        with st.expander("Normality test"):
            st.dataframe(res["normality"], hide_index=True, use_container_width=True)


st.divider()

# ===========================================================================
# Section 2 — ANOVA
# ===========================================================================

st.header("ANOVA")

anova_type = st.radio(
    "ANOVA type",
    ["One-way", "Two-way", "Repeated-measures"],
    horizontal=True,
    key="anova_type",
)

if not cat_cols:
    st.warning("No categorical columns available. ANOVA requires at least one categorical grouping variable.")
else:
    # ---- One-way ANOVA -------------------------------------------------------
    if anova_type == "One-way":
        c1, c2 = st.columns(2)
        ow_dv = c1.selectbox("Dependent variable (numeric)", numeric_cols, key="ow_dv")
        ow_iv = c2.selectbox("Factor / grouping column (categorical)", cat_cols, key="ow_iv")

        if st.button("Run One-Way ANOVA", type="primary", key="run_ow"):
            try:
                res = run_oneway_anova(analysis_df, ow_dv, ow_iv)
            except Exception as e:
                st.error(f"ANOVA failed: {e}")
                st.stop()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("F-statistic", f"{res['f_stat']:.4f}")
            c2.metric("p-value", _pval_label(res["p_value"]))
            c3.metric("η² (eta²)", f"{res['eta_sq']:.4f}")
            c4.metric("ω² (omega²)", f"{res['omega_sq']:.4f}")

            sig = res["p_value"] < 0.05
            st.info(
                f"**Interpretation**: {'Significant' if sig else 'No significant'} "
                f"difference in '{ow_dv}' across levels of '{ow_iv}', "
                f"F({res['df_between']}, {res['df_within']}) = {res['f_stat']:.4f}, "
                f"p = {res['p_value']:.4f}."
            )

            st.subheader("ANOVA Table")
            st.dataframe(res["anova_table"], hide_index=True, use_container_width=True)
            download_csv(res["anova_table"], "Download ANOVA table", f"anova_oneway_{ow_dv}.csv", key="dl_ow_anova")

            st.subheader("Group Descriptive Statistics")
            st.dataframe(res["group_stats"], hide_index=True, use_container_width=True)

            # Bar chart of group means ± SD
            fig = go.Figure()
            for _, row in res["group_stats"].iterrows():
                fig.add_trace(go.Bar(
                    name=str(row["Group"]),
                    x=[str(row["Group"])],
                    y=[row["Mean"]],
                    error_y=dict(type="data", array=[row["SD"]], visible=True),
                ))
            fig.update_layout(
                title=f"Group Means ± 1 SD: {ow_dv} by {ow_iv}",
                yaxis_title=ow_dv,
                xaxis_title=ow_iv,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Post-hoc Tukey HSD
            if sig:
                st.subheader("Post-hoc: Tukey HSD")
                try:
                    tukey_df = run_tukey_hsd(analysis_df, ow_dv, ow_iv)
                    st.dataframe(tukey_df, hide_index=True, use_container_width=True)
                    download_csv(tukey_df, "Download Tukey HSD results", f"tukey_{ow_dv}.csv", key="dl_tukey")
                except Exception as e:
                    st.error(f"Tukey HSD failed: {e}")

            # Non-parametric alternative
            with st.expander("Assumptions & non-parametric alternative"):
                st.markdown(
                    f"**Levene's test** (homogeneity of variance): "
                    f"F = {res['levene_stat']:.4f}, p = {_pval_label(res['levene_p'])}"
                )
                if res["levene_p"] < 0.05:
                    st.warning("Levene's test is significant — variances may not be homogeneous.")

                st.markdown(
                    f"**Kruskal-Wallis** (non-parametric alternative): "
                    f"H = {res['kruskal_stat']:.4f}, p = {_pval_label(res['kruskal_p'])}"
                )

                st.markdown("**Normality per group (Shapiro-Wilk / D'Agostino):**")
                for grp_label, norm_df in res["normality_per_group"].items():
                    with st.expander(f"Group: {grp_label}"):
                        st.dataframe(norm_df, hide_index=True, use_container_width=True)

    # ---- Two-way ANOVA -------------------------------------------------------
    elif anova_type == "Two-way":
        if len(cat_cols) < 2:
            st.warning("Two-way ANOVA requires at least 2 categorical columns.")
        else:
            c1, c2, c3 = st.columns(3)
            tw_dv = c1.selectbox("Dependent variable (numeric)", numeric_cols, key="tw_dv")
            tw_f1 = c2.selectbox("Factor 1 (categorical)", cat_cols, key="tw_f1")
            tw_f2 = c3.selectbox("Factor 2 (categorical)", [c for c in cat_cols if c != tw_f1], key="tw_f2")

            if st.button("Run Two-Way ANOVA", type="primary", key="run_tw"):
                try:
                    res = run_twoway_anova(analysis_df, tw_dv, tw_f1, tw_f2)
                except Exception as e:
                    st.error(f"Two-way ANOVA failed: {e}")
                    st.stop()

                st.metric("n (complete cases)", res["n"])
                st.subheader("ANOVA Table (Type II SS)")
                st.dataframe(res["anova_table"], hide_index=True, use_container_width=True)

                st.subheader("Cell Means")
                st.dataframe(res["group_stats"], hide_index=True, use_container_width=True)

    # ---- Repeated-measures ANOVA ---------------------------------------------
    else:
        st.info(
            "Repeated-measures ANOVA requires a **subject ID column** (identifying each "
            "participant) and a **within-subject factor column** (e.g. time point, condition). "
            "Both can be any column designated as Categorical on the Upload page."
        )
        if len(cat_cols) < 2:
            st.warning("Repeated-measures ANOVA requires at least 2 categorical columns "
                       "(within-subject factor + subject ID).")
        else:
            c1, c2, c3 = st.columns(3)
            rm_dv = c1.selectbox("Dependent variable (numeric)", numeric_cols, key="rm_dv")
            rm_within = c2.selectbox("Within-subject factor (categorical)", cat_cols, key="rm_within")
            rm_subject = c3.selectbox(
                "Subject ID column (categorical)",
                [c for c in cat_cols if c != rm_within],
                key="rm_subject",
            )

            if st.button("Run Repeated-Measures ANOVA", type="primary", key="run_rm"):
                try:
                    res = run_rm_anova(analysis_df, rm_dv, rm_within, rm_subject)
                except ImportError as e:
                    st.error(str(e))
                    st.stop()
                except Exception as e:
                    st.error(f"Repeated-measures ANOVA failed: {e}")
                    st.stop()

                for w in res["warnings"]:
                    st.warning(w)

                c1, c2 = st.columns(2)
                c1.metric("Subjects (n)", res["n_subjects"])
                c2.metric("Within-subject levels", res["n_levels"])

                st.subheader("RM-ANOVA Table")
                st.dataframe(res["anova_table"], hide_index=True, use_container_width=True)

                # Sphericity
                if res["sphericity"]:
                    sph = res["sphericity"]
                    with st.expander("Mauchly's Test of Sphericity"):
                        passed = sph["sphericity"]
                        st.markdown(
                            f"**W** = {sph['W']:.4f}, **χ²**({sph['dof']}) = {sph['chi2']:.4f}, "
                            f"**p** = {_pval_label(sph['pval'])}"
                        )
                        if passed:
                            st.success("Sphericity assumption met.")
                        else:
                            st.warning(
                                "Sphericity violated — Greenhouse-Geisser correction applied "
                                "automatically in the ANOVA table above."
                            )

                # Post-hoc
                st.subheader("Post-hoc Pairwise Tests (FDR-corrected)")
                st.dataframe(res["post_hoc"], hide_index=True, use_container_width=True)


st.divider()

# ===========================================================================
# Section 3 — Mediation Analysis
# ===========================================================================

st.header("Mediation Analysis")

if len(numeric_cols) < 3:
    st.warning("Mediation analysis requires at least 3 numeric columns (X, M, Y).")
else:
    c1, c2, c3 = st.columns(3)
    med_x = c1.selectbox("X (independent variable)", numeric_cols, key="med_x")
    med_m = c2.selectbox("M (mediator)", [c for c in numeric_cols if c != med_x], key="med_m")
    med_y = c3.selectbox("Y (dependent variable)", [c for c in numeric_cols if c not in (med_x, med_m)], key="med_y")

    cov_options = [c for c in numeric_cols if c not in (med_x, med_m, med_y)]
    covariates = st.multiselect("Covariates (optional)", cov_options, key="med_cov")

    n_boot = st.slider("Bootstrap samples", min_value=500, max_value=5000, value=1000, step=500, key="med_nboot")

    if st.button("Run Mediation Analysis", type="primary", key="run_med"):
        try:
            res = run_mediation(
                analysis_df, med_x, med_m, med_y,
                covariates=covariates if covariates else None,
                n_boot=n_boot,
            )
        except ImportError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Mediation analysis failed: {e}")
            st.stop()

        # Key result banner
        med_type = res["mediation_type"]
        if med_type == "Full mediation":
            st.success(f"Result: **{med_type}** — indirect effect CI excludes zero and direct path is non-significant.")
        elif med_type == "Partial mediation":
            st.warning(f"Result: **{med_type}** — indirect effect CI excludes zero but direct path remains significant.")
        else:
            st.info(f"Result: **{med_type}** — indirect effect CI includes zero.")

        ci_low, ci_high = res["indirect_ci"]
        st.caption(
            f"Bootstrap 95% CI for indirect effect (ab): [{ci_low:.4f}, {ci_high:.4f}]   |   "
            f"n = {res['n']}   |   Bootstrap samples = {n_boot}"
        )

        st.subheader("Path Coefficients")
        st.dataframe(res["path_table"], hide_index=True, use_container_width=True)
        download_csv(res["path_table"], "Download mediation path table", f"mediation_{med_x}_{med_m}_{med_y}.csv", key="dl_mediation")

        # ---- Path diagram (Plotly) ------------------------------------------
        st.subheader("Path Diagram")

        paths = res["paths"]

        def _coef(path_key: str) -> str:
            if path_key in paths:
                coef = paths[path_key].get("coef", float("nan"))
                pval = paths[path_key].get("pval", 1.0)
                stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                return f"{float(coef):.3f}{stars}"
            return "n/a"

        # Node positions (x, y) in a simple layout
        nodes = {
            "X": (0.0, 0.5),
            "M": (0.5, 1.0),
            "Y": (1.0, 0.5),
        }
        node_labels = {"X": med_x, "M": med_m, "Y": med_y}
        node_colors = {"X": "#4C78A8", "M": "#F58518", "Y": "#54A24B"}

        fig = go.Figure()

        # Arrows as annotations
        arrow_defs = [
            ("X", "M", _coef("a"), "top"),
            ("M", "Y", _coef("b"), "top"),
            ("X", "Y", "c={}  c'={}".format(_coef("c"), _coef("c'")), "bottom"),
        ]

        for src, dst, label, text_pos in arrow_defs:
            x0, y0 = nodes[src]
            x1, y1 = nodes[dst]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            offset = 0.07 if text_pos == "top" else -0.07
            fig.add_annotation(
                x=x1, y=y1, ax=x0, ay=y0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=3, arrowsize=1.5, arrowwidth=2,
                arrowcolor="#555",
            )
            fig.add_annotation(
                x=mx, y=my + offset,
                text=f"<b>{label}</b>",
                showarrow=False,
                font=dict(size=12),
                xref="x", yref="y",
            )

        # Node boxes
        for key, (nx, ny) in nodes.items():
            fig.add_shape(
                type="rect",
                x0=nx - 0.12, y0=ny - 0.1, x1=nx + 0.12, y1=ny + 0.1,
                fillcolor=node_colors[key], opacity=0.85,
                line=dict(color="white", width=2),
            )
            fig.add_annotation(
                x=nx, y=ny,
                text=f"<b>{node_labels[key]}</b>",
                showarrow=False,
                font=dict(color="white", size=11),
                xref="x", yref="y",
            )

        fig.update_layout(
            xaxis=dict(range=[-0.2, 1.2], visible=False),
            yaxis=dict(range=[0.2, 1.2], visible=False),
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Significance stars: * p<0.05  ** p<0.01  *** p<0.001  |  c = total effect, c' = direct effect")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Multilevel Mediation (2-1-1)
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.header("7 · Multilevel Mediation (2-1-1)")
st.markdown(
    "Tests indirect effects via one or more mediators in nested/clustered data "
    "using Linear Mixed Models (LMM). Bootstrap CIs resample **clusters** "
    "(not rows) to preserve within-cluster structure. "
    "Requires `statsmodels >= 0.14`."
)

with st.expander("Configure Multilevel Mediation", expanded=True):
    ml_col1, ml_col2 = st.columns(2)

    all_num = numeric_cols  # defined at top of this file
    all_cat = cat_cols

    with ml_col1:
        ml_cluster = st.selectbox(
            "Cluster / Subject ID column",
            options=["— select —"] + all_cat + all_num,
            key="ml_cluster",
            help="Column that identifies each cluster (e.g. participant, school, group ID).",
        )
        ml_y = st.selectbox(
            "Outcome (Y)",
            options=["— select —"] + all_num,
            key="ml_y",
            help="Dependent variable — must be numeric.",
        )
        ml_x = st.selectbox(
            "Predictor (X)",
            options=["— select —"] + all_num + all_cat,
            key="ml_x",
            help="Level-2 independent variable (e.g. treatment group).",
        )

    with ml_col2:
        ml_mediators = st.multiselect(
            "Mediator(s) (M)",
            options=all_num,
            key="ml_mediators",
            help="One or more numeric mediators. Each is tested independently.",
        )
        ml_covariates = st.multiselect(
            "Level-2 covariates (optional)",
            options=[c for c in all_num + all_cat if c not in [ml_y, ml_x]],
            key="ml_covariates",
            help="Cluster-level covariates added to all models.",
        )
        ml_l1_preds = st.multiselect(
            "Level-1 (within-cluster) predictors of Y (optional)",
            options=[c for c in all_num if c not in [ml_y, ml_x]],
            key="ml_l1_preds",
            help="Row-level predictors included in the Y model only.",
        )

    ml_n_boot = st.slider(
        "Bootstrap iterations",
        min_value=100, max_value=2000, value=500, step=100,
        key="ml_n_boot",
        help="More iterations = more stable CIs but slower runtime.",
    )

    st.caption(
        "Runtime note: each mediator requires two LMM fits per bootstrap "
        "iteration. 500 iterations on a typical dataset takes ~10–30 s."
    )

    ml_ready = (
        ml_cluster not in ("— select —", None)
        and ml_y not in ("— select —", None)
        and ml_x not in ("— select —", None)
        and len(ml_mediators) >= 1
        and ml_y != ml_x
    )

    run_ml_med = st.button(
        "Run Multilevel Mediation",
        disabled=not ml_ready,
        type="primary",
        key="run_ml_med_btn",
    )
    if not ml_ready:
        st.caption("Select cluster column, X, Y, and at least one mediator to enable.")

if run_ml_med:
    with st.spinner("Fitting LMMs and bootstrapping indirect effects…"):
        try:
            ml_result = run_multilevel_mediation(
                df=analysis_df,
                outcome=ml_y,
                x_col=ml_x,
                mediators=ml_mediators,
                cluster_col=ml_cluster,
                covariates=ml_covariates if ml_covariates else None,
                level1_predictors=ml_l1_preds if ml_l1_preds else None,
                n_boot=ml_n_boot,
            )
        except Exception as exc:
            st.error(f"Multilevel mediation failed: {exc}")
            st.stop()

    # ── Warnings ──────────────────────────────────────────────────────────
    if ml_result.get("warnings"):
        for w in ml_result["warnings"]:
            st.warning(w)

    # ── Summary metrics ───────────────────────────────────────────────────
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Observations", ml_result["n_obs"])
    mc2.metric("Clusters", ml_result["n_clusters"])
    mc3.metric("Bootstrap converged", f"{ml_result['n_boot_ok']} / {ml_n_boot}")

    # ── Path table ────────────────────────────────────────────────────────
    st.subheader("Path Coefficients")
    path_df = ml_result["path_table"]

    def _star(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    display_path = path_df.copy()
    if "p_value" in display_path.columns:
        display_path["sig"] = display_path["p_value"].apply(_star)

    st.dataframe(display_path, use_container_width=True)
    download_csv(display_path, "Download path table", "ml_med_paths.csv", "dl_ml_paths")

    # ── Indirect effects ──────────────────────────────────────────────────
    st.subheader("Indirect Effects (Bootstrap CIs)")
    indirect_rows = []
    for med, ie in ml_result["indirect"].items():
        ci_lo = ml_result.get("indirect_ci_lo", {}).get(med, float("nan"))
        ci_hi = ml_result.get("indirect_ci_hi", {}).get(med, float("nan"))
        sig = "Yes" if (
            not (pd.isna(ci_lo) or pd.isna(ci_hi))
            and not (ci_lo <= 0 <= ci_hi)
        ) else "No"
        indirect_rows.append({
            "Mediator": med,
            "Indirect effect (a×b)": round(ie, 4),
            "95% CI lower": round(ci_lo, 4),
            "95% CI upper": round(ci_hi, 4),
            "Significant (CI excludes 0)": sig,
        })
    indirect_disp = pd.DataFrame(indirect_rows)
    st.dataframe(indirect_disp, use_container_width=True)
    download_csv(indirect_disp, "Download indirect effects", "ml_med_indirect.csv", "dl_ml_indirect")

    # ── Total effect ──────────────────────────────────────────────────────
    te = ml_result.get("total_effect", float("nan"))
    te_ci = ml_result.get("total_ci", (float("nan"), float("nan")))
    te_col1, te_col2 = st.columns(2)
    te_col1.metric("Total effect (c)", f"{te:.4f}" if not pd.isna(te) else "n/a")
    te_col2.metric(
        "95% CI",
        f"[{te_ci[0]:.4f}, {te_ci[1]:.4f}]"
        if not (pd.isna(te_ci[0]) or pd.isna(te_ci[1])) else "n/a",
    )

    # ── Plotly path diagram ───────────────────────────────────────────────
    st.subheader("Path Diagram")

    n_med = len(ml_mediators)
    # Layout: X at left (0.05, 0.5), mediators stacked centre (0.5, evenly spaced), Y at right (0.95, 0.5)
    x_pos, y_pos_node = 0.05, 0.5
    y_x_node = 0.95
    med_xs = [0.5] * n_med
    med_ys = [(i + 1) / (n_med + 1) for i in range(n_med)]

    node_x = [x_pos] + med_xs + [y_x_node]
    node_y = [y_pos_node] + med_ys + [y_pos_node]
    node_labels_diag = [ml_x] + ml_mediators + [ml_y]
    node_colors_diag = ["#0068C9"] + ["#FF7F0E"] * n_med + ["#2CA02C"]

    # Build arrow annotations
    arrow_annotations = []

    # X → each mediator (path a)
    for i, med in enumerate(ml_mediators):
        a_coef = ml_result["path_a"].get(med, float("nan"))
        arrow_annotations.append(dict(
            ax=x_pos, ay=y_pos_node,
            x=0.5, y=med_ys[i],
            xref="x", yref="y", axref="x", ayref="y",
            text=f"a={a_coef:.3f}" if not pd.isna(a_coef) else "a=?",
            showarrow=True, arrowhead=3, arrowwidth=2, arrowcolor="#555",
            font=dict(size=10),
        ))

    # Each mediator → Y (path b)
    for i, med in enumerate(ml_mediators):
        b_coef = ml_result["path_b"].get(med, float("nan"))
        arrow_annotations.append(dict(
            ax=0.5, ay=med_ys[i],
            x=y_x_node, y=y_pos_node,
            xref="x", yref="y", axref="x", ayref="y",
            text=f"b={b_coef:.3f}" if not pd.isna(b_coef) else "b=?",
            showarrow=True, arrowhead=3, arrowwidth=2, arrowcolor="#555",
            font=dict(size=10),
        ))

    # X → Y direct (path c')
    c_prime = ml_result.get("path_c_prime", float("nan"))
    arrow_annotations.append(dict(
        ax=x_pos, ay=y_pos_node - 0.08,
        x=y_x_node, y=y_pos_node - 0.08,
        xref="x", yref="y", axref="x", ayref="y",
        text=f"c'={c_prime:.3f}" if not pd.isna(c_prime) else "c'=?",
        showarrow=True, arrowhead=3, arrowwidth=2, arrowcolor="#888",
        font=dict(size=10), arrowdash="dot",
    ))

    fig_ml = go.Figure()

    fig_ml.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=48, color=node_colors_diag, opacity=0.9, line=dict(color="white", width=2)),
        text=[f"<b>{lbl}</b>" for lbl in node_labels_diag],
        textposition="middle center",
        textfont=dict(color="white", size=10),
        hoverinfo="text",
    ))

    fig_ml.update_layout(
        annotations=arrow_annotations,
        xaxis=dict(range=[-0.05, 1.05], visible=False),
        yaxis=dict(range=[-0.05, 1.05], visible=False),
        height=max(350, 120 * n_med + 100),
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig_ml, use_container_width=True)
    st.caption(
        "Blue = predictor (X) · Orange = mediator(s) · Green = outcome (Y) · "
        "Solid arrows = a/b paths · Dotted arrow = direct effect c'"
    )
