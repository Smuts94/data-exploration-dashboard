"""
core/loader.py — File parsing, type inference, and validation.
"""
from __future__ import annotations

import io
import chardet
import pandas as pd


def detect_encoding(raw: bytes) -> str:
    result = chardet.detect(raw)
    return result.get("encoding") or "utf-8"


def load_file(uploaded_file) -> pd.DataFrame:
    """
    Parse an uploaded Streamlit file object (.csv, .tsv, .xlsx).
    Returns a DataFrame with basic type inference applied.
    Raises ValueError with a plain-English message on failure.
    """
    name: str = uploaded_file.name.lower()
    raw: bytes = uploaded_file.read()
    uploaded_file.seek(0)

    if name.endswith(".xlsx"):
        try:
            df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
        except Exception as exc:
            raise ValueError(f"Could not read Excel file: {exc}") from exc
        return _coerce_types(df)

    # CSV / TSV — detect encoding first
    encoding = detect_encoding(raw)
    text = io.StringIO(raw.decode(encoding, errors="replace"))

    if name.endswith(".tsv"):
        sep = "\t"
    else:
        # Sniff delimiter
        sample = raw[:4096].decode(encoding, errors="replace")
        sep = _sniff_delimiter(sample)

    try:
        df = pd.read_csv(text, sep=sep, engine="python")
    except Exception as exc:
        raise ValueError(f"Could not parse file: {exc}") from exc

    return _coerce_types(df)


def _sniff_delimiter(sample: str) -> str:
    import csv
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to downcast object columns to numeric or datetime."""
    for col in df.columns:
        if df[col].dtype == object:
            # Try numeric
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() / max(df[col].notna().sum(), 1) > 0.8:
                df[col] = converted
                continue
            # Try datetime
            try:
                converted_dt = pd.to_datetime(df[col], errors="coerce")
                if converted_dt.notna().sum() / max(df[col].notna().sum(), 1) > 0.8:
                    df[col] = converted_dt
            except Exception:
                pass
    return df


def infer_col_types(df: pd.DataFrame) -> dict[str, str]:
    """
    Return a dict mapping column name → 'Numeric' | 'Categorical' | 'DateTime'.
    Used as the initial value for session_state["col_types"].
    """
    mapping: dict[str, str] = {}
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_datetime64_any_dtype(dtype):
            mapping[col] = "DateTime"
        elif pd.api.types.is_numeric_dtype(dtype):
            mapping[col] = "Numeric"
        else:
            mapping[col] = "Categorical"
    return mapping
