import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.state import (
    init_state, require_upload,
    get_working_df, get_filtered_df, get_col_types, get_filters,
    set_filtered_df, set_filters, reset_filters,
)
from core.sidebar import render_sidebar

st.set_page_config(page_title="Data Filter", layout="wide")
init_state()

st.title("2 · Data Filter")

if not require_upload():
    st.stop()

raw_df = get_working_df()
col_types = get_col_types()
current_filters = get_filters()

# Show sidebar (study/group context) — read-only context on this page
filtered_df_current = get_filtered_df()
render_sidebar(filtered_df_current, col_types)

# ---------------------------------------------------------------------------
# Filter panel — operates on raw_df
# ---------------------------------------------------------------------------
st.subheader("Filter Panel")
st.caption("Filters here define the base dataset used by all analysis pages.")

new_filters: dict = {}
mask = pd.Series([True] * len(raw_df), index=raw_df.index)

numeric_cols = [c for c, t in col_types.items() if t == "Numeric" and c in raw_df.columns]
cat_cols = [c for c, t in col_types.items() if t == "Categorical" and c in raw_df.columns]
dt_cols = [c for c, t in col_types.items() if t == "DateTime" and c in raw_df.columns]

if numeric_cols:
    st.markdown("**Numeric filters**")
    num_grid = st.columns(min(3, len(numeric_cols)))
    for idx, col in enumerate(numeric_cols):
        col_data = raw_df[col].dropna()
        if col_data.empty:
            continue
        col_min = float(col_data.min())
        col_max = float(col_data.max())
        if col_min == col_max:
            continue
        saved = current_filters.get(col, (col_min, col_max))
        saved = (max(col_min, saved[0]), min(col_max, saved[1]))
        with num_grid[idx % len(num_grid)]:
            lo, hi = st.slider(
                col,
                min_value=col_min,
                max_value=col_max,
                value=saved,
                key=f"filter_num_{col}",
            )
        new_filters[col] = (lo, hi)
        mask &= raw_df[col].between(lo, hi, inclusive="both") | raw_df[col].isna()

if cat_cols:
    st.markdown("**Categorical filters**")
    for col in cat_cols:
        all_vals = sorted(raw_df[col].dropna().unique().tolist(), key=str)
        saved_vals = current_filters.get(col, all_vals)
        saved_vals = [v for v in saved_vals if v in all_vals] or all_vals
        selected = st.multiselect(
            col,
            options=all_vals,
            default=saved_vals,
            key=f"filter_cat_{col}",
        )
        new_filters[col] = selected
        if selected:
            mask &= raw_df[col].isin(selected) | raw_df[col].isna()
        else:
            mask &= pd.Series([False] * len(raw_df), index=raw_df.index)

if dt_cols:
    st.markdown("**DateTime filters**")
    for col in dt_cols:
        col_data = raw_df[col].dropna()
        if col_data.empty:
            continue
        d_min = col_data.min().date()
        d_max = col_data.max().date()
        saved = current_filters.get(col, (d_min, d_max))
        c1, c2 = st.columns(2)
        lo_date = c1.date_input(f"{col} from", value=saved[0], min_value=d_min, max_value=d_max,
                                key=f"filter_dt_lo_{col}")
        hi_date = c2.date_input(f"{col} to", value=saved[1], min_value=d_min, max_value=d_max,
                                key=f"filter_dt_hi_{col}")
        new_filters[col] = (lo_date, hi_date)
        mask &= (raw_df[col].dt.date >= lo_date) & (raw_df[col].dt.date <= hi_date) | raw_df[col].isna()

filtered = raw_df[mask].copy()
set_filtered_df(filtered)
set_filters(new_filters)

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
n_total = len(raw_df)
n_filtered = len(filtered)
pct = round(n_filtered / n_total * 100, 1) if n_total > 0 else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Rows retained", f"{n_filtered:,}")
c2.metric("Original rows", f"{n_total:,}")
c3.metric("Retained %", f"{pct}%")

# ---------------------------------------------------------------------------
# Buttons
# ---------------------------------------------------------------------------
col_a, col_b = st.columns(2)
with col_a:
    if st.button("Reset All Filters", type="secondary"):
        reset_filters()
        st.rerun()

with col_b:
    if not filtered.empty:
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Filtered Dataset (CSV)",
            data=csv_bytes,
            file_name="filtered_dataset.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# Preview / empty-state guard
# ---------------------------------------------------------------------------
if filtered.empty:
    st.error(
        "The filtered dataset is empty. Relax your filters or click **Reset All Filters**. "
        "The Regression page will be blocked until at least one row is available."
    )
else:
    st.subheader("Preview (first 20 rows)")
    st.dataframe(filtered.head(20), use_container_width=True)
