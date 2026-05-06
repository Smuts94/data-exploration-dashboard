"""
core/group_utils.py — Per-visual group selection helpers.

Key concepts
------------
• universal_filter(...)   — Page-level "Variable of interest" + "Default group split"
                            rendered once at the top of each analysis page.
                            Returns (default_var, default_grp_col).
                            When either value changes it auto-resets all local
                            selector keys so every sub-visual inherits the new default.

• local_group_selector(...) — Per-visual override selectbox.
                            Starts at the page-level default; user may change it
                            independently for any specific visual.

• render_group_layout(...)  — Chooses layout based on group count:
                              1 group  → single block
                              2 groups → st.columns(2), left / right
                              3+ groups→ st.tabs()
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

_NO_SPLIT = "— No split —"


# ---------------------------------------------------------------------------
# Universal (page-level) filter
# ---------------------------------------------------------------------------

def universal_filter(
    df: pd.DataFrame,
    col_types: dict[str, str],
    page_prefix: str,
    local_var_keys: list[str],
    local_grp_keys: list[str],
) -> tuple[str | None, str | None]:
    """
    Render a page-level "Variable of interest" and "Default group split" block.

    Parameters
    ----------
    df              : The analysis DataFrame for this page.
    col_types       : Column-type dict from session_state.
    page_prefix     : Short unique string for this page, e.g. "p1".
    local_var_keys  : session_state keys of all per-visual *column* selectors.
                      These are reset when the default variable changes.
    local_grp_keys  : session_state keys of all per-visual *group* selectors.
                      These are reset when the default group changes.

    Returns
    -------
    default_var     : The selected default numeric column (str | None).
    default_grp_col : The selected default group column (str | None).
    """
    numeric_cols = [c for c, t in col_types.items() if t == "Numeric" and c in df.columns]
    cat_cols     = [c for c, t in col_types.items() if t == "Categorical" and c in df.columns]

    var_key      = f"{page_prefix}_default_var"
    grp_key      = f"{page_prefix}_default_grp"
    prev_var_key = f"{page_prefix}_prev_var"
    prev_grp_key = f"{page_prefix}_prev_grp"

    with st.container(border=True):
        st.markdown("##### Page defaults — all visuals below inherit these unless overridden")
        c1, c2 = st.columns(2)

        default_var = c1.selectbox(
            "Variable of interest",
            options=numeric_cols,
            key=var_key,
            help="Sets the default variable shown in every distribution, normality, and outlier visual on this page.",
        ) if numeric_cols else None

        grp_options = [_NO_SPLIT] + cat_cols
        grp_val = c2.selectbox(
            "Default group split",
            options=grp_options,
            key=grp_key,
            help="Sets the default group split for every visual on this page. Individual visuals can override this.",
        )
        default_grp_col = None if grp_val == _NO_SPLIT else grp_val

        # ── Cascade: reset local selectors when page defaults change ─────────
        if st.session_state.get(prev_var_key) != default_var:
            for k in local_var_keys:
                st.session_state.pop(k, None)
            st.session_state[prev_var_key] = default_var

        if st.session_state.get(prev_grp_key) != default_grp_col:
            for k in local_grp_keys:
                st.session_state.pop(k, None)
            st.session_state[prev_grp_key] = default_grp_col

    return default_var, default_grp_col


# ---------------------------------------------------------------------------
# Per-visual local group selector
# ---------------------------------------------------------------------------

def local_group_selector(
    df: pd.DataFrame,
    col_types: dict[str, str],
    key: str,
    default_col: str | None = None,
) -> tuple[str | None, list[str], list[pd.DataFrame]]:
    """
    Renders a compact "Override group split" selectbox directly above a visual.

    The selectbox defaults to `default_col` (the page-level default) if provided.
    The user may change it freely; their choice is persisted in session_state[key].

    Returns
    -------
    group_col  : chosen column name, or None if no split
    group_vals : sorted list of unique group value strings (empty if no split)
    subsets    : list of DataFrame subsets, one per group value
    """
    cat_cols = [c for c, t in col_types.items() if t == "Categorical" and c in df.columns]
    if not cat_cols:
        return None, [], []

    options = [_NO_SPLIT] + cat_cols

    # Compute default index: page default if set, otherwise "No split"
    if default_col and default_col in cat_cols:
        default_idx = options.index(default_col)
    else:
        default_idx = 0  # "— No split —"

    # Only set index if key is not already in session_state (respects manual overrides)
    if key not in st.session_state:
        st.session_state[key] = options[default_idx]

    choice = st.selectbox(
        "Override group split",
        options=options,
        key=key,
        help=(
            "Split this visual by a different group than the page default, "
            "or choose '— No split —' to show all data combined."
        ),
    )

    if choice == _NO_SPLIT:
        st.caption("Group split: all data combined")
        return None, [], []

    group_vals = sorted(df[choice].dropna().unique().tolist(), key=str)
    subsets    = [df[df[choice] == v].copy() for v in group_vals]
    st.caption(f"Group split: **{choice}** ({len(group_vals)} groups)")
    return choice, [str(v) for v in group_vals], subsets


# ---------------------------------------------------------------------------
# Layout renderer
# ---------------------------------------------------------------------------

def render_group_layout(
    group_vals: list[str],
    subsets: list[pd.DataFrame],
    render_fn,
    *args,
    **kwargs,
) -> None:
    """
    Renders render_fn for each group using a layout chosen by group count:

        1 group  → single block
        2 groups → st.columns(2), group 1 left / group 2 right
        3+ groups→ st.tabs(), one tab per group

    render_fn signature:
        render_fn(sub_df: pd.DataFrame, group_label: str, *args, **kwargs)
    """
    n = len(group_vals)
    if n == 0:
        return

    if n == 1:
        render_fn(subsets[0], group_vals[0], *args, **kwargs)

    elif n == 2:
        col_left, col_right = st.columns(2)
        with col_left:
            st.caption(f"**{group_vals[0]}**")
            render_fn(subsets[0], group_vals[0], *args, **kwargs)
        with col_right:
            st.caption(f"**{group_vals[1]}**")
            render_fn(subsets[1], group_vals[1], *args, **kwargs)

    else:
        tabs = st.tabs(group_vals)
        for tab, label, sub_df in zip(tabs, group_vals, subsets):
            with tab:
                render_fn(sub_df, label, *args, **kwargs)


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------

def download_csv(
    df: pd.DataFrame,
    label: str = "Download results as CSV",
    filename: str = "results.csv",
    key: str | None = None,
) -> None:
    """
    Renders a Streamlit download button that exports `df` as a UTF-8 CSV.

    Parameters
    ----------
    df       : DataFrame to export.
    label    : Button label shown to the user.
    filename : Default filename for the saved file.
    key      : Streamlit widget key (auto-generated from filename if not given).
    """
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        key=key or f"dl_{filename}",
    )
