"""
core/sidebar.py — Global study / group / variable filter sidebar component.

Usage on every analysis page:
    from core.sidebar import render_sidebar
    analysis_df, selected_groups = render_sidebar(filtered_df, col_types)

Returns:
    analysis_df     — filtered_df further subset by the active study + group selections
    selected_groups — list of currently selected group values (for plots that
                      branch on group count, e.g. split KDE)

Study filter is a single-select (selectbox): exactly one study is always active,
so all visuals on every analysis page show data for that one study only.
Group filter remains a multiselect — useful for within-study group comparisons.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

from core.state import (
    get_study_col, get_group_col,
    get_selected_study, get_selected_groups, get_selected_vars,
    set_selected_study, set_selected_groups, set_selected_vars,
    get_dataset_name, set_dataset_name,
)


def render_sidebar(
    filtered_df: pd.DataFrame,
    col_types: dict[str, str],
) -> tuple[pd.DataFrame, list]:
    """
    Render study / group / variable filters in the sidebar.
    Returns (analysis_df, selected_groups).
    analysis_df is a copy — never mutates filtered_df.
    """
    study_col = get_study_col()
    group_col = get_group_col()
    numeric_cols = [c for c, t in col_types.items() if t == "Numeric" and c in filtered_df.columns]

    analysis_df = filtered_df.copy()

    with st.sidebar:
        # ── Dataset name ─────────────────────────────────────────────────────
        current_name = get_dataset_name()
        if current_name:
            new_name = st.text_input(
                "Dataset",
                value=current_name,
                key="sb_dataset_name",
                help="Editable label — rename to something memorable.",
            )
            if new_name != current_name:
                set_dataset_name(new_name)

        st.markdown("---")
        st.subheader("Analysis Scope")

        # ── Study filter ────────────────────────────────────────────────────
        selected_study: str | None = None
        if study_col and study_col in filtered_df.columns:
            all_studies = sorted(filtered_df[study_col].dropna().unique().tolist(), key=str)
            saved = get_selected_study()
            # Default to first study if nothing saved or saved value no longer valid
            default_idx = all_studies.index(saved) if saved in all_studies else 0

            selected_study = st.selectbox(
                f"Study ({study_col})",
                options=all_studies,
                index=default_idx,
                key="sb_study",
            )
            set_selected_study(selected_study)
            analysis_df = analysis_df[analysis_df[study_col] == selected_study]
        else:
            st.caption("_Study column: not set_")

        # ── Group filter ─────────────────────────────────────────────────────
        selected_groups: list = []
        if group_col and group_col in filtered_df.columns:
            all_groups = sorted(analysis_df[group_col].dropna().unique().tolist(), key=str)
            saved_g = [g for g in get_selected_groups() if g in all_groups] or all_groups

            selected_groups = st.multiselect(
                f"Groups ({group_col})",
                options=all_groups,
                default=saved_g,
                key="sb_groups",
            )
            set_selected_groups(selected_groups)

            if not selected_groups:
                st.warning("No groups selected — showing all.")
                selected_groups = all_groups

            analysis_df = analysis_df[analysis_df[group_col].isin(selected_groups)]

            if len(selected_groups) == 2:
                st.info(f"Comparing: **{selected_groups[0]}** vs **{selected_groups[1]}**")
            elif len(selected_groups) > 2:
                st.caption(f"{len(selected_groups)} groups selected")
        else:
            st.caption("_Group column: not set_")

        # ── Variable filter ──────────────────────────────────────────────────
        if numeric_cols:
            saved_v = [v for v in get_selected_vars() if v in numeric_cols] or numeric_cols
            selected_vars = st.multiselect(
                "Variables",
                options=numeric_cols,
                default=saved_v,
                key="sb_vars",
            )
            if not selected_vars:
                st.warning("No variables selected — using all.")
                selected_vars = numeric_cols
            set_selected_vars(selected_vars)
        else:
            selected_vars = []
            st.caption("_No numeric columns_")

        # ── Active filter summary badge ──────────────────────────────────────
        st.markdown("---")
        parts = []
        if study_col and selected_study is not None:
            parts.append(f"**Study:** {selected_study}")
        if group_col and selected_groups:
            parts.append(f"**Groups:** {len(selected_groups)}")
        parts.append(f"**Vars:** {len(selected_vars)}")
        parts.append(f"**n =** {len(analysis_df):,}")
        st.caption(" · ".join(parts))

    return analysis_df, selected_groups
