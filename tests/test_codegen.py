"""
tests/test_codegen.py — guards for reproducible code export.

The key test (`test_generated_python_runs`) writes each generated Python script
to disk and executes it against a fixture dataset in a subprocess. This catches
template syntax/logic errors — the main drift risk — end to end.
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core import codegen  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture dataset — structured so every analysis kind has valid inputs
# ---------------------------------------------------------------------------

def _make_fixture(path) -> None:
    """Write a balanced 36-row CSV (12 subjects x 3 timepoints) with real
    cluster structure, so mixed models and RM-ANOVA converge."""
    rng = np.random.default_rng(0)
    n_subj, n_time = 12, 3
    rows = []
    for s in range(n_subj):
        subj_effect = rng.normal(0, 2)
        for t in range(n_time):
            idx = s * n_time + t
            xvar = rng.normal(50, 10)
            mediator = 0.5 * xvar + rng.normal(0, 5)
            score = subj_effect + 0.3 * mediator + 2 * t + rng.normal(0, 3)
            rows.append(dict(
                subject=f"S{s:02d}",
                timepoint=["t1", "t2", "t3"][t],
                arm="A" if s < n_subj // 2 else "B",
                site=["X", "Y", "Z"][idx % 3],
                cohort=["P", "Q"][idx % 2],
                age=rng.normal(40, 10), weight=rng.normal(70, 12),
                pre=rng.normal(100, 15), post=rng.normal(105, 15),
                xvar=xvar, mediator=mediator, score=score,
            ))
    pd.DataFrame(rows).to_csv(path, index=False)


# Valid params for every registered analysis kind, against the fixture columns.
PARAMS = {
    "descriptive_stats": {"columns": ["score", "age", "weight"]},
    "outliers": {"columns": ["score", "age", "weight"]},
    "normality": {"column": "score"},
    "correlation": {"columns": ["score", "age", "weight"], "method": "pearson"},
    "independent_ttest": {"value_col": "score", "group_col": "arm"},
    "paired_ttest": {"col1": "pre", "col2": "post"},
    "onesample_ttest": {"column": "score", "mu0": 0.0},
    "oneway_anova": {"dv": "score", "factor": "site"},
    "twoway_anova": {"dv": "score", "factor1": "site", "factor2": "cohort"},
    "rm_anova": {"dv": "score", "within": "timepoint", "subject": "subject"},
    "mediation": {"x": "xvar", "m": "mediator", "y": "score",
                  "covariates": [], "n_boot": 50, "seed": 42},
    "multilevel_mediation": {"outcome": "score", "x": "xvar",
                             "mediators": ["mediator"], "cluster": "subject",
                             "covariates": [], "n_boot": 50, "seed": 42},
    "ols_regression": {"y": "score", "x_cols": ["age", "weight"],
                       "include_const": True},
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_params_cover_registry():
    """Every registered kind must have test params (and vice versa)."""
    assert set(PARAMS) == set(codegen.REGISTRY)


@pytest.mark.parametrize("kind", sorted(codegen.REGISTRY))
def test_scripts_generate(kind):
    prov = codegen.Provenance(data_file="data.csv")
    spec = {"kind": kind, "params": PARAMS[kind]}
    py = codegen.python_script(prov, [spec])
    r = codegen.r_script(prov, [spec])
    label = codegen.REGISTRY[kind]["label"]
    assert "import pandas as pd" in py and label in py
    assert "read.csv" in r and label in r


def test_filter_chain_emitted():
    prov = codegen.Provenance(
        data_file="data.csv",
        col_types={"age": "Numeric", "arm": "Categorical"},
        filters={"age": (10.0, 50.0), "arm": ["A", "B"]},
        study_col="cohort", selected_study="P",
        group_col="arm", selected_groups=["A", "B"],
    )
    spec = {"kind": "normality", "params": {"column": "score"}}
    py = codegen.python_script(prov, [spec])
    assert ".between(10.0, 50.0)" in py
    assert ".isin(['A', 'B'])" in py
    assert "df[df['cohort'] == 'P']" in py
    r = codegen.r_script(prov, [spec])
    assert "%in%" in r and ">= 10.0" in r


def test_session_script_covers_multiple_analyses():
    prov = codegen.Provenance(data_file="data.csv")
    specs = [{"kind": k, "params": PARAMS[k]}
             for k in ("descriptive_stats", "correlation", "independent_ttest")]
    py = codegen.python_script(prov, specs)
    for k in ("descriptive_stats", "correlation", "independent_ttest"):
        assert codegen.REGISTRY[k]["label"] in py


def test_tier2_r_carries_note():
    """Tier 2 R blocks must explain divergence from the Python dashboard."""
    for kind in ("mediation", "multilevel_mediation", "rm_anova"):
        block = codegen.REGISTRY[kind]["r"](PARAMS[kind])
        assert block.note, f"{kind} R block is missing a divergence NOTE"


@pytest.mark.parametrize("kind", sorted(codegen.REGISTRY))
def test_generated_python_runs(kind, tmp_path):
    """End-to-end: the generated Python script runs cleanly on a fixture."""
    data = tmp_path / "fixture.csv"
    _make_fixture(data)
    prov = codegen.Provenance(data_file=str(data))
    spec = {"kind": kind, "params": PARAMS[kind]}
    script = tmp_path / f"{kind}.py"
    script.write_text(codegen.python_script(prov, [spec]), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, (
        f"Generated '{kind}' script failed.\n"
        f"--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}"
    )
