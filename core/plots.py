"""
core/plots.py — All plot-generating functions.
Returns Plotly figures or Matplotlib figures — no Streamlit calls here.
"""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as scipy_stats
import seaborn as sns


# ---------------------------------------------------------------------------
# Univariate
# ---------------------------------------------------------------------------

def histogram_kde(
    series: pd.Series,
    col_name: str,
    group_series: pd.Series | None = None,
    group_labels: list | None = None,
) -> go.Figure:
    """
    Histogram + KDE overlay.
    If group_series and group_labels are provided, draws one KDE per group
    on the same axis (colour-coded), without a separate histogram per group
    to keep the chart readable.
    """
    GROUP_COLORS = [
        "crimson", "steelblue", "seagreen", "darkorange",
        "mediumpurple", "sienna", "teal", "deeppink",
    ]

    fig = go.Figure()

    if group_series is not None and group_labels:
        # Pooled histogram in the background
        s_all = series.dropna()
        fig.add_trace(go.Histogram(
            x=s_all, name="All (histogram)",
            histnorm="probability density",
            marker_color="lightgrey", opacity=0.4,
            nbinsx=min(60, max(10, len(s_all) // 20)),
            showlegend=True,
        ))
        # One KDE per group
        for i, label in enumerate(group_labels):
            mask = group_series == label
            s = series[mask].dropna()
            if len(s) < 2:
                continue
            color = GROUP_COLORS[i % len(GROUP_COLORS)]
            kde = scipy_stats.gaussian_kde(s)
            x_range = np.linspace(s_all.min(), s_all.max(), 300)
            fig.add_trace(go.Scatter(
                x=x_range, y=kde(x_range),
                mode="lines", name=str(label),
                line=dict(color=color, width=2),
            ))
    else:
        s = series.dropna()
        fig.add_trace(go.Histogram(
            x=s, name="Histogram",
            histnorm="probability density",
            marker_color="steelblue", opacity=0.6,
            nbinsx=min(60, max(10, len(s) // 20)),
        ))
        kde = scipy_stats.gaussian_kde(s)
        x_range = np.linspace(s.min(), s.max(), 300)
        fig.add_trace(go.Scatter(
            x=x_range, y=kde(x_range),
            mode="lines", name="KDE",
            line=dict(color="crimson", width=2),
        ))

    fig.update_layout(
        title=f"Distribution — {col_name}",
        xaxis_title=col_name,
        yaxis_title="Density",
        legend=dict(orientation="h"),
        template="plotly_white",
        barmode="overlay",
    )
    return fig


def qq_plot(series: pd.Series, col_name: str) -> go.Figure:
    s = series.dropna().sort_values().values
    n = len(s)
    theoretical = scipy_stats.norm.ppf(np.linspace(0.01, 0.99, n))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theoretical, y=s,
        mode="markers",
        marker=dict(color="steelblue", size=4, opacity=0.6),
        name="Quantiles",
    ))
    # Reference line
    lo = min(theoretical.min(), s.min())
    hi = max(theoretical.max(), s.max())
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines", line=dict(color="red", dash="dash"),
        name="Normal reference",
    ))
    fig.update_layout(
        title=f"Q-Q Plot — {col_name}",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        template="plotly_white",
    )
    return fig


def categorical_bar(series: pd.Series, col_name: str) -> go.Figure:
    counts = series.value_counts().reset_index()
    counts.columns = [col_name, "Count"]
    fig = px.bar(
        counts, x="Count", y=col_name, orientation="h",
        title=f"Value Counts — {col_name}",
        template="plotly_white",
        color_discrete_sequence=["steelblue"],
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def correlation_heatmap(
    corr: pd.DataFrame,
    pvals: pd.DataFrame,
) -> go.Figure:
    """
    Single annotated heatmap. Each cell shows the correlation value with
    significance stars appended (e.g. 0.83***).
    Diagonal cells show '—'.
    """
    cols = corr.columns.tolist()
    z = corr.values.copy()

    # Build annotation text
    text = []
    for i in range(len(cols)):
        row_text = []
        for j in range(len(cols)):
            if i == j:
                row_text.append("—")
            else:
                val = corr.iloc[i, j]
                p = pvals.iloc[i, j]
                if np.isnan(p):
                    stars = ""
                elif p < 0.001:
                    stars = "***"
                elif p < 0.01:
                    stars = "**"
                elif p < 0.05:
                    stars = "*"
                else:
                    stars = ""
                row_text.append(f"{val:.2f}{stars}")
        text.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=z, x=cols, y=cols,
        text=text,
        texttemplate="%{text}",
        colorscale="RdBu_r",
        zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title="r"),
    ))
    fig.update_layout(
        title="Correlation Matrix (with significance stars: * p<0.05  ** p<0.01  *** p<0.001)",
        template="plotly_white",
        xaxis=dict(tickangle=-45),
        height=max(400, 60 * len(cols)),
        width=max(500, 70 * len(cols)),
    )
    return fig


def scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    size_col: str | None = None,
) -> go.Figure:
    fig = px.scatter(
        df, x=x_col, y=y_col,
        color=color_col,
        size=size_col,
        opacity=0.6,
        template="plotly_white",
        title=f"{y_col} vs. {x_col}",
    )
    # Trendline via numpy
    sub = df[[x_col, y_col]].dropna()
    if len(sub) > 1:
        m, b = np.polyfit(sub[x_col], sub[y_col], 1)
        x_range = np.linspace(sub[x_col].min(), sub[x_col].max(), 200)
        fig.add_trace(go.Scatter(
            x=x_range, y=m * x_range + b,
            mode="lines", line=dict(color="red", dash="dash", width=1.5),
            name="OLS trend",
        ))
    return fig


def pairplot_image(
    df: pd.DataFrame,
    cols: list[str],
    hue_col: str | None = None,
) -> bytes:
    """
    Render a Seaborn pairplot and return PNG bytes for st.image().
    If hue_col is provided, colour-codes points by that column.
    """
    keep_cols = list(cols) + ([hue_col] if hue_col and hue_col not in cols else [])
    sub = df[keep_cols].dropna(subset=cols)
    g = sns.pairplot(
        sub,
        hue=hue_col,
        diag_kind="kde",
        plot_kws={"alpha": 0.4, "s": 10},
    )
    g.figure.suptitle("Pairplot", y=1.02)
    buf = io.BytesIO()
    g.figure.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(g.figure)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Regression diagnostic plots
# ---------------------------------------------------------------------------

def residual_diagnostic_plots(model_result) -> go.Figure:
    """
    2×2 grid of residual diagnostic plots using Plotly:
    1. Residuals vs. Fitted
    2. Q-Q plot of residuals
    3. Scale-Location
    4. Residuals vs. Leverage (with Cook's distance contours)
    """
    fitted = model_result.fittedvalues
    residuals = model_result.resid
    std_resid = residuals / residuals.std()
    sqrt_abs_resid = np.sqrt(np.abs(std_resid))

    influence = model_result.get_influence()
    leverage = influence.hat_matrix_diag
    cooks_d = influence.cooks_distance[0]

    # Q-Q data
    sorted_resid = np.sort(residuals.values)
    n = len(sorted_resid)
    theoretical_q = scipy_stats.norm.ppf(np.linspace(0.01, 0.99, n))

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Residuals vs. Fitted",
            "Q-Q Plot of Residuals",
            "Scale-Location",
            "Residuals vs. Leverage",
        ],
    )

    # 1 — Residuals vs. Fitted
    fig.add_trace(go.Scatter(
        x=fitted, y=residuals, mode="markers",
        marker=dict(color="steelblue", size=4, opacity=0.5),
        name="Residuals",
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # 2 — Q-Q
    fig.add_trace(go.Scatter(
        x=theoretical_q, y=sorted_resid, mode="markers",
        marker=dict(color="steelblue", size=4, opacity=0.5),
        name="Q-Q",
    ), row=1, col=2)
    lo = min(theoretical_q.min(), sorted_resid.min())
    hi = max(theoretical_q.max(), sorted_resid.max())
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="red", dash="dash"), name="Normal ref",
    ), row=1, col=2)

    # 3 — Scale-Location
    fig.add_trace(go.Scatter(
        x=fitted, y=sqrt_abs_resid, mode="markers",
        marker=dict(color="steelblue", size=4, opacity=0.5),
        name="Scale-Loc",
    ), row=2, col=1)

    # 4 — Residuals vs. Leverage
    fig.add_trace(go.Scatter(
        x=leverage, y=std_resid, mode="markers",
        marker=dict(
            color=cooks_d,
            colorscale="Reds",
            size=5,
            opacity=0.7,
            colorbar=dict(title="Cook's D", x=1.02),
        ),
        name="Leverage",
        text=[f"Cook's D: {c:.4f}" for c in cooks_d],
    ), row=2, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="grey", row=2, col=2)

    fig.update_layout(
        height=700,
        template="plotly_white",
        showlegend=False,
        title_text="Residual Diagnostic Plots",
    )
    fig.update_xaxes(title_text="Fitted values", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
    fig.update_xaxes(title_text="Fitted values", row=2, col=1)
    fig.update_yaxes(title_text="√|Std. Residuals|", row=2, col=1)
    fig.update_xaxes(title_text="Leverage", row=2, col=2)
    fig.update_yaxes(title_text="Std. Residuals", row=2, col=2)

    return fig


def coefficient_plot(model_result) -> go.Figure:
    """Coefficient point estimates with 95% CI, excluding the intercept visually."""
    params = model_result.params
    conf = model_result.conf_int()
    names = params.index.tolist()

    fig = go.Figure()
    for name in names:
        lo = conf.loc[name, 0]
        hi = conf.loc[name, 1]
        est = params[name]
        fig.add_trace(go.Scatter(
            x=[est],
            y=[name],
            mode="markers",
            marker=dict(size=10, color="steelblue"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[hi - est],
                arrayminus=[est - lo],
                color="steelblue",
                thickness=2,
            ),
            name=name,
        ))
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_layout(
        title="Coefficient Plot (95% CI)",
        xaxis_title="Estimate",
        yaxis_title="Predictor",
        template="plotly_white",
        showlegend=False,
        height=max(300, 40 * len(names) + 100),
    )
    return fig
