"""
core/export_ui.py — Streamlit glue for reproducible code export.

Builds a `codegen.Provenance` from `session_state`, renders the per-result
"Reproduce this in R / Python" expander, and the whole-session export.

Export is a nice-to-have: every function here fails soft — a codegen error
must never break the page it sits on.
"""
from __future__ import annotations

import streamlit as st

from core import codegen
from core.state import (
    get_dataset_name, get_col_types, get_filters,
    get_study_col, get_selected_study, get_group_col, get_selected_groups,
    log_analysis, get_analysis_log,
)


def build_provenance() -> codegen.Provenance:
    """Capture the current filter/study/group context into a Provenance."""
    name = get_dataset_name() or "your_data.csv"
    return codegen.Provenance(
        data_file=name,
        filters=dict(get_filters() or {}),
        col_types=dict(get_col_types() or {}),
        study_col=get_study_col(),
        selected_study=get_selected_study(),
        group_col=get_group_col(),
        selected_groups=list(get_selected_groups() or []),
    )


def render_export(kind: str, params: dict, *, key: str, log: bool = True) -> None:
    """
    Render an expander with R + Python download buttons that reproduce one
    analysis. Also records the analysis in the session log (for whole-session
    export) unless `log=False`.
    """
    if kind not in codegen.REGISTRY:
        return
    if log:
        try:
            log_analysis(kind, params)
        except Exception:
            pass

    with st.expander("⬇ Reproduce this in R / Python", expanded=False):
        try:
            prov = build_provenance()
            spec = {"kind": kind, "params": params}
            has_filters = codegen.has_provenance(prov)
            bake = False
            if has_filters:
                bake = st.checkbox(
                    "Bake in dashboard filters (reproduce the exact subset)",
                    value=False, key=f"{key}_bake",
                    help="Off: the script loads your full data file into `df` and "
                         "runs the analysis (the filter lines are included but "
                         "commented out). On: the filter/study/group selections "
                         "are applied so the script reproduces the exact subset.",
                )
            py_code = codegen.python_script(prov, [spec], include_filters=bake)
            r_code = codegen.r_script(prov, [spec], include_filters=bake)
        except Exception as exc:  # never break the host page
            st.caption(f"Code export unavailable: {exc}")
            return

        st.caption(
            "Standalone scripts that load your original data file into `df` and "
            "run this analysis. Open the script and set `DATA_FILE` at the top to "
            "your data file, then run it."
        )
        c1, c2 = st.columns(2)
        c1.download_button(
            "Python script (.py)", py_code,
            file_name=f"{kind}.py", mime="text/x-python", key=f"{key}_py",
            use_container_width=True,
        )
        c2.download_button(
            "R script (.R)", r_code,
            file_name=f"{kind}.R", mime="text/plain", key=f"{key}_r",
            use_container_width=True,
        )


def render_session_export(*, key: str = "session_export") -> None:
    """Render download buttons for one script covering the whole session."""
    log = get_analysis_log()
    if not log:
        return
    try:
        prov = build_provenance()
        bake = False
        if codegen.has_provenance(prov):
            bake = st.checkbox(
                "Bake in dashboard filters (reproduce the exact subset)",
                value=False, key=f"{key}_bake",
                help="Off: scripts load your full data file into `df`. On: the "
                     "filter/study/group selections are applied to reproduce the "
                     "exact analysed subset.",
            )
        py_code = codegen.python_script(prov, log, include_filters=bake)
        r_code = codegen.r_script(prov, log, include_filters=bake)
    except Exception as exc:
        st.caption(f"Session export unavailable: {exc}")
        return

    n = len(log)
    st.download_button(
        f"Python — full session ({n})", py_code,
        file_name="session_analysis.py", mime="text/x-python",
        key=f"{key}_py", use_container_width=True,
    )
    st.download_button(
        f"R — full session ({n})", r_code,
        file_name="session_analysis.R", mime="text/plain",
        key=f"{key}_r", use_container_width=True,
    )
