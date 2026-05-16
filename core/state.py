"""
core/state.py — Streamlit session_state helpers.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------
KEY_RAW_DF = "raw_df"
KEY_FILTERED_DF = "filtered_df"
KEY_COL_TYPES = "col_types"
KEY_FILTERS = "filters"
KEY_DATASET_NAME = "dataset_name"
KEY_SELECTED_COLS = "selected_cols"   # columns kept after Variable Selection page

# Study / group / variable context
KEY_STUDY_COL = "study_col"
KEY_GROUP_COL = "group_col"
KEY_SELECTED_STUDY = "selected_study"
KEY_SELECTED_GROUPS = "selected_groups"
KEY_SELECTED_VARS = "selected_vars"

# Reproducible code export — ordered log of analyses run this session
KEY_ANALYSIS_LOG = "analysis_log"


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_state() -> None:
    """Ensure all keys exist with sensible defaults."""
    defaults = {
        KEY_RAW_DF: None,
        KEY_FILTERED_DF: None,
        KEY_COL_TYPES: {},
        KEY_FILTERS: {},
        KEY_DATASET_NAME: "",
        KEY_SELECTED_COLS: None,   # None = all columns kept
        KEY_STUDY_COL: None,
        KEY_GROUP_COL: None,
        KEY_SELECTED_STUDY: None,
        KEY_SELECTED_GROUPS: [],
        KEY_SELECTED_VARS: [],
        KEY_ANALYSIS_LOG: [],
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def get_dataset_name() -> str:
    return st.session_state.get(KEY_DATASET_NAME, "")


def set_dataset_name(name: str) -> None:
    st.session_state[KEY_DATASET_NAME] = name


def get_selected_cols() -> list[str] | None:
    """Returns the user-chosen column subset, or None if all columns are kept."""
    return st.session_state.get(KEY_SELECTED_COLS)


def set_selected_cols(cols: list[str] | None) -> None:
    st.session_state[KEY_SELECTED_COLS] = cols


def get_working_df() -> pd.DataFrame | None:
    """
    Returns raw_df restricted to the columns selected on the Variable Selection page.
    Falls back to raw_df if no column selection has been made.
    This is the base from which filtered_df is derived on the Data Filter page.
    """
    raw = get_raw_df()
    if raw is None:
        return None
    cols = get_selected_cols()
    if cols:
        keep = [c for c in cols if c in raw.columns]
        return raw[keep].copy()
    return raw.copy()


def get_raw_df() -> pd.DataFrame | None:
    return st.session_state.get(KEY_RAW_DF)


def get_filtered_df() -> pd.DataFrame | None:
    return st.session_state.get(KEY_FILTERED_DF)


def get_col_types() -> dict[str, str]:
    return st.session_state.get(KEY_COL_TYPES, {})


def get_filters() -> dict:
    return st.session_state.get(KEY_FILTERS, {})


def get_study_col() -> str | None:
    return st.session_state.get(KEY_STUDY_COL)


def get_group_col() -> str | None:
    return st.session_state.get(KEY_GROUP_COL)


def get_selected_studies() -> list:
    # Legacy shim — returns a single-item list for backwards compat with any
    # code that still iterates over selected studies.
    val = st.session_state.get(KEY_SELECTED_STUDY)
    return [val] if val is not None else []


def get_selected_study() -> str | None:
    return st.session_state.get(KEY_SELECTED_STUDY)


def get_selected_groups() -> list:
    return st.session_state.get(KEY_SELECTED_GROUPS, [])


def get_selected_vars() -> list:
    return st.session_state.get(KEY_SELECTED_VARS, [])


# ---------------------------------------------------------------------------
# Mutators
# ---------------------------------------------------------------------------

def set_upload(raw_df: pd.DataFrame, col_types: dict[str, str]) -> None:
    """Called once on upload. Sets raw_df and resets filtered_df to a copy."""
    st.session_state[KEY_RAW_DF] = raw_df
    st.session_state[KEY_FILTERED_DF] = raw_df.copy()
    st.session_state[KEY_COL_TYPES] = col_types
    st.session_state[KEY_FILTERS] = {}
    st.session_state[KEY_SELECTED_COLS] = None   # reset column selection on new upload
    # Reset study/group context on new upload
    st.session_state[KEY_STUDY_COL] = None
    st.session_state[KEY_GROUP_COL] = None
    st.session_state[KEY_SELECTED_STUDY] = None
    st.session_state[KEY_SELECTED_GROUPS] = []
    st.session_state[KEY_SELECTED_VARS] = []
    st.session_state[KEY_ANALYSIS_LOG] = []


def set_col_types(col_types: dict[str, str]) -> None:
    st.session_state[KEY_COL_TYPES] = col_types


def set_filtered_df(df: pd.DataFrame) -> None:
    st.session_state[KEY_FILTERED_DF] = df


def set_filters(filters: dict) -> None:
    st.session_state[KEY_FILTERS] = filters


def reset_filters() -> None:
    working = get_working_df()
    if working is not None:
        st.session_state[KEY_FILTERED_DF] = working
    st.session_state[KEY_FILTERS] = {}


def set_study_col(col: str | None) -> None:
    st.session_state[KEY_STUDY_COL] = col
    st.session_state[KEY_SELECTED_STUDY] = None


def set_group_col(col: str | None) -> None:
    st.session_state[KEY_GROUP_COL] = col
    st.session_state[KEY_SELECTED_GROUPS] = []


def set_selected_studies(vals: list) -> None:
    # Legacy shim — accepts a list but stores only the first element
    st.session_state[KEY_SELECTED_STUDY] = vals[0] if vals else None


def set_selected_study(val: str | None) -> None:
    st.session_state[KEY_SELECTED_STUDY] = val


def set_selected_groups(vals: list) -> None:
    st.session_state[KEY_SELECTED_GROUPS] = vals


def set_selected_vars(vals: list) -> None:
    st.session_state[KEY_SELECTED_VARS] = vals


# ---------------------------------------------------------------------------
# Analysis log — drives reproducible code export
# ---------------------------------------------------------------------------

def get_analysis_log() -> list[dict]:
    """Ordered list of {'kind': str, 'params': dict} for analyses run this session."""
    return st.session_state.get(KEY_ANALYSIS_LOG, [])


def log_analysis(kind: str, params: dict) -> None:
    """
    Record that an analysis was run, for the whole-session code export.
    Deduplicates exact (kind, params) repeats — Streamlit reruns the whole
    script on every interaction — and caps the log at 50 entries.
    """
    log = list(st.session_state.get(KEY_ANALYSIS_LOG, []))
    entry = {"kind": kind, "params": params}
    log = [e for e in log if e != entry]
    log.append(entry)
    st.session_state[KEY_ANALYSIS_LOG] = log[-50:]


def clear_analysis_log() -> None:
    st.session_state[KEY_ANALYSIS_LOG] = []


# ---------------------------------------------------------------------------
# Guard helpers (used at the top of each page)
# ---------------------------------------------------------------------------

def require_upload() -> bool:
    """
    Returns True if a dataset has been uploaded.
    If not, renders a warning and returns False — the page should st.stop() after.
    """
    if get_raw_df() is None:
        st.warning("No dataset loaded. Please upload a file on the **Upload** page first.")
        return False
    return True


def require_nonempty_filtered() -> bool:
    """
    Returns True if filtered_df is non-empty.
    Renders a warning and returns False otherwise.
    """
    df = get_filtered_df()
    if df is None or df.empty:
        st.error(
            "The filtered dataset is empty. "
            "Go to the **Filter** page and relax your filters, or reset them."
        )
        return False
    return True
