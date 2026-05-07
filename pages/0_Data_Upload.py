import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.loader import load_file, infer_col_types
from core.state import (
    init_state, set_upload, set_col_types, set_study_col, set_group_col,
    get_raw_df, get_col_types, get_study_col, get_group_col,
    get_dataset_name, set_dataset_name,
)
from core.theme import inject_theme, page_header

st.set_page_config(page_title="Data Upload", layout="wide")
init_state()
inject_theme()

page_header(
    eyebrow="Step 0",
    title="Data Upload",
    lede=(
        "Drop a CSV, TSV, or XLSX file. We'll preview shape, types, and "
        "missingness, then let you mark the study and group columns that "
        "drive every downstream analysis page."
    ),
)

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload your dataset",
    type=["csv", "tsv", "xlsx"],
    help="Accepts .csv, .tsv, and .xlsx files. Delimiter and encoding are auto-detected.",
)

if uploaded is not None:
    with st.spinner("Parsing file…"):
        try:
            df = load_file(uploaded)
        except ValueError as e:
            st.error(f"Could not load file: {e}")
            st.stop()

    col_types = infer_col_types(df)
    set_upload(df, col_types)
    set_dataset_name(uploaded.name)
    st.success(f"Loaded **{uploaded.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# Render preview (if data is in state)
# ---------------------------------------------------------------------------
raw_df = get_raw_df()

if raw_df is None:
    st.info("Upload a file above to get started.")
    st.stop()

df = raw_df
col_types = get_col_types()

# ── Shape
st.subheader("Dataset Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{df.shape[0]:,}")
c2.metric("Columns", df.shape[1])
c3.metric("Duplicate rows", f"{df.duplicated().sum():,}")

# ── First 10 rows
st.subheader("First 10 Rows")
st.dataframe(df.head(10), use_container_width=True)

# ── Column info
st.subheader("Column Info")
col_info = pd.DataFrame({
    "Column": df.columns,
    "Dtype": [str(df[c].dtype) for c in df.columns],
    "Non-null": [df[c].notna().sum() for c in df.columns],
    "Null count": [df[c].isna().sum() for c in df.columns],
    "Null %": [round(df[c].isna().mean() * 100, 2) for c in df.columns],
    "Unique values": [df[c].nunique() for c in df.columns],
})
st.dataframe(col_info, use_container_width=True)

# ── Missing value summary (sortable)
st.subheader("Missing Value Summary")
missing = col_info[col_info["Null count"] > 0][["Column", "Null count", "Null %"]].sort_values(
    "Null %", ascending=False
)
if missing.empty:
    st.success("No missing values found.")
else:
    st.dataframe(missing, use_container_width=True)

# ── Duplicate rows
dup_count = df.duplicated().sum()
if dup_count > 0:
    with st.expander(f"Preview duplicate rows ({dup_count:,})"):
        st.dataframe(df[df.duplicated(keep=False)], use_container_width=True)

# ---------------------------------------------------------------------------
# Column type overrides
# ---------------------------------------------------------------------------
st.subheader("Column Type Overrides")
st.caption(
    "Override the inferred type for any column. "
    "Changes are applied immediately and persist across all pages."
)

TYPE_OPTIONS = ["Numeric", "Categorical", "DateTime"]
updated_types = {}
n_cols = min(3, len(df.columns))
chunks = [list(df.columns)[i::n_cols] for i in range(n_cols)]
grid_cols = st.columns(n_cols)

for col_idx, chunk in enumerate(chunks):
    with grid_cols[col_idx]:
        for col in chunk:
            current = col_types.get(col, "Categorical")
            if current not in TYPE_OPTIONS:
                current = "Categorical"
            sel = st.selectbox(
                col,
                TYPE_OPTIONS,
                index=TYPE_OPTIONS.index(current),
                key=f"coltype_{col}",
            )
            updated_types[col] = sel

for col, ctype in updated_types.items():
    if ctype == "Categorical" and df[col].nunique() > 50:
        st.warning(
            f"**{col}** is typed as Categorical but has {df[col].nunique():,} unique values. "
            "Consider changing it to Numeric or DateTime."
        )

set_col_types(updated_types)

# ---------------------------------------------------------------------------
# Study / Group column designation
# ---------------------------------------------------------------------------
st.subheader("Study & Group Columns")
st.caption(
    "Designate which columns identify studies and comparison groups. "
    "These drive the global filters available on every analysis page."
)

cat_cols_available = [c for c, t in updated_types.items() if t == "Categorical"]
none_option = "— none —"
options_with_none = [none_option] + cat_cols_available

current_study = get_study_col()
current_group = get_group_col()

study_default = current_study if current_study in cat_cols_available else none_option
group_default = current_group if current_group in cat_cols_available else none_option

sc1, sc2 = st.columns(2)
with sc1:
    study_sel = st.selectbox(
        "Study column",
        options=options_with_none,
        index=options_with_none.index(study_default),
        help="Column whose values identify different studies in the dataset.",
        key="upload_study_col",
    )
with sc2:
    group_sel = st.selectbox(
        "Group / condition column",
        options=options_with_none,
        index=options_with_none.index(group_default),
        help="Column whose values identify groups or conditions to compare.",
        key="upload_group_col",
    )

set_study_col(study_sel if study_sel != none_option else None)
set_group_col(group_sel if group_sel != none_option else None)

if study_sel != none_option:
    unique_studies = sorted(df[study_sel].dropna().unique().tolist(), key=str)
    st.caption(f"Studies detected: {', '.join(str(s) for s in unique_studies)}")

if group_sel != none_option:
    unique_groups = sorted(df[group_sel].dropna().unique().tolist(), key=str)
    st.caption(f"Groups detected: {', '.join(str(g) for g in unique_groups)}")
