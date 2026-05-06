"""
Page 1 — Variable Selection
Pick which columns to carry into all downstream analysis.
Reduces the working dataset before any row filtering happens.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd

from core.state import (
    init_state, require_upload,
    get_raw_df, get_col_types, set_col_types,
    get_selected_cols, set_selected_cols, get_working_df,
    set_filtered_df, set_filters,
    get_dataset_name,
)

st.set_page_config(page_title="Variable Selection", layout="wide")
init_state()

st.title("1 · Variable Selection")
st.caption(
    "Choose which columns to carry forward into all downstream pages. "
    "Columns you deselect here are dropped from the working dataset immediately — "
    "this keeps memory low and analysis focused."
)

if not require_upload():
    st.stop()

raw_df   = get_raw_df()
col_types = get_col_types()
current_selection = get_selected_cols()   # None = all columns

all_cols    = list(raw_df.columns)
n_total     = len(all_cols)
numeric_cols = [c for c, t in col_types.items() if t == "Numeric"     and c in all_cols]
cat_cols     = [c for c, t in col_types.items() if t == "Categorical" and c in all_cols]
other_cols   = [c for c in all_cols if c not in numeric_cols + cat_cols]

# ---------------------------------------------------------------------------
# Quick-select helpers
# ---------------------------------------------------------------------------
st.subheader("Quick select")

col_a, col_b, col_c = st.columns(3)

if col_a.button("Select all columns", use_container_width=True):
    st.session_state["varsel_chosen"] = all_cols[:]

if col_b.button("Numeric only", use_container_width=True):
    st.session_state["varsel_chosen"] = numeric_cols[:]

top_n = col_c.number_input(
    "Keep top N numeric columns (by variance)",
    min_value=1, max_value=max(1, len(numeric_cols)),
    value=min(10, len(numeric_cols)),
    step=1,
    key="varsel_topn",
    help="Ranks numeric columns by their variance in the raw dataset and keeps the N highest.",
)
if col_c.button(f"Apply top {int(top_n)}", use_container_width=True):
    if numeric_cols:
        variances = raw_df[numeric_cols].var().sort_values(ascending=False)
        top_numeric = variances.head(int(top_n)).index.tolist()
        # Always keep categorical & other cols alongside the selected numeric ones
        st.session_state["varsel_chosen"] = top_numeric + cat_cols + other_cols
    else:
        st.warning("No numeric columns detected.")

# ---------------------------------------------------------------------------
# Column selector — grouped by type
# ---------------------------------------------------------------------------
st.subheader("Select columns to keep")
st.caption("Study / Group columns you designated on the Upload page are highlighted — deselecting them will remove those filters.")

# Determine the default selection
if "varsel_chosen" in st.session_state:
    _default = st.session_state["varsel_chosen"]
elif current_selection:
    _default = current_selection
else:
    _default = all_cols[:]

# Build display table for context
col_meta = pd.DataFrame({
    "Column":   all_cols,
    "Type":     [col_types.get(c, "—") for c in all_cols],
    "Non-null": [int(raw_df[c].notna().sum()) for c in all_cols],
    "Unique":   [int(raw_df[c].nunique()) for c in all_cols],
    "Variance": [
        f"{raw_df[c].var():.4g}" if col_types.get(c) == "Numeric" else "—"
        for c in all_cols
    ],
})

with st.expander("Column overview (click to expand)", expanded=False):
    st.dataframe(col_meta, use_container_width=True, hide_index=True)

# Group columns into sections for readability
sections = [
    ("Numeric columns",     numeric_cols),
    ("Categorical columns", cat_cols),
    ("Other columns",       other_cols),
]

chosen: list[str] = []
for section_label, section_cols in sections:
    if not section_cols:
        continue
    st.markdown(f"**{section_label}** ({len(section_cols)} total)")
    sec_default = [c for c in section_cols if c in _default]
    picked = st.multiselect(
        f"Select from {section_label.lower()}",
        options=section_cols,
        default=sec_default,
        key=f"varsel_{section_label}",
        label_visibility="collapsed",
    )
    chosen.extend(picked)

# ---------------------------------------------------------------------------
# Summary & Apply
# ---------------------------------------------------------------------------
st.divider()

n_kept   = len(chosen)
n_rows   = len(raw_df)
kept_num = sum(1 for c in chosen if col_types.get(c) == "Numeric")
kept_cat = sum(1 for c in chosen if col_types.get(c) == "Categorical")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Columns kept",    f"{n_kept} / {n_total}")
c2.metric("Numeric kept",    kept_num)
c3.metric("Categorical kept",kept_cat)
c4.metric("Rows (unchanged)", f"{n_rows:,}")

if n_kept == 0:
    st.error("You must keep at least one column.")
    st.stop()

if st.button("Apply variable selection", type="primary", key="varsel_apply"):
    # Preserve column-type map to only kept columns
    new_col_types = {c: col_types[c] for c in chosen if c in col_types}
    set_col_types(new_col_types)
    set_selected_cols(chosen)
    # Reset filtered_df to the new column-subset so Data Filter starts fresh
    set_filtered_df(raw_df[chosen].copy())
    set_filters({})
    st.success(
        f"Working dataset updated: {n_kept} columns × {n_rows:,} rows. "
        "Downstream pages now use only these columns."
    )
    st.info("Proceed to **Data Filter** to apply row filters, or go straight to analysis.")

# Show current active selection
if current_selection:
    st.caption(
        f"Currently active: **{len(current_selection)}** columns selected. "
        "Press *Apply* above to update with your new choices."
    )
else:
    st.caption("No variable selection applied yet — all columns are in use.")
