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
    try:
        prov = build_provenance()
        spec = {"kind": kind, "params": params}
        py_code = codegen.python_script(prov, [spec])
        r_code = codegen.r_script(prov, [spec])
    except Exception as exc:  # never break the host page
        st.caption(f"Code export unavailable: {exc}")
        return

    with st.expander("⬇ Reproduce this in R / Python", expanded=False):
        st.caption(
            "Standalone scripts that recreate this exact result on your own "
            "machine. Open the script and set `DATA_FILE` at the top to your "
            "data file, then run it. Filters and study/group selections are "
            "baked in."
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
        py_code = codegen.python_script(prov, log)
        r_code = codegen.r_script(prov, log)
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
