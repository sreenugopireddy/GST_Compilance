"""
Phase 1 — Schema profiling.

The real dataset is anonymized (Column0..Column21 with no known semantic
meaning), so instead of hand-coding distribution assumptions per column
(as you'd do if you knew "this is turnover, that's a filing delay"), we
profile the REAL training data directly: per-column cardinality (to
classify continuous vs. discrete/binary), summary statistics, missingness
rate, and correlation with the target. The synthetic generator (see
synthetic_generator.py) is fit directly on this profile, so synthetic
rows are grounded in the actual data rather than invented distributions.
"""

import numpy as np
import pandas as pd
from src import config


def profile_columns(df, numeric_cols=config.NUMERIC_COLS,
                     discrete_threshold=config.DISCRETE_CARDINALITY_THRESHOLD):
    """
    Classify each numeric column as 'continuous' or 'discrete' by
    cardinality, and compute the stats needed to bootstrap-resample it.

    Inputs:
        df: pd.DataFrame - real training data (features only, or with label).
        numeric_cols: list[str]
        discrete_threshold: int - nunique <= this => discrete/binary

    Returns:
        dict[col] -> {
            "kind": "continuous" | "discrete",
            "nunique": int,
            "missing_rate": float,
            "mean": float, "std": float, "min": float, "max": float,
            "values": np.ndarray of non-null observed values (for bootstrap),
        }
    """
    profile = {}
    for col in numeric_cols:
        series = df[col]
        non_null = series.dropna().values
        nunique = series.nunique(dropna=True)
        kind = "continuous" if nunique > discrete_threshold else "discrete"
        profile[col] = {
            "kind": kind,
            "nunique": int(nunique),
            "missing_rate": float(series.isnull().mean()),
            "mean": float(non_null.mean()) if len(non_null) else 0.0,
            "std": float(non_null.std()) if len(non_null) else 1.0,
            "min": float(non_null.min()) if len(non_null) else 0.0,
            "max": float(non_null.max()) if len(non_null) else 1.0,
            "values": non_null,
        }
    return profile


def compute_target_correlations(df, numeric_cols=config.NUMERIC_COLS, label_col=config.LABEL_COL):
    """
    Point-biserial correlation of each numeric column with the binary
    target (equivalent to Pearson correlation when one variable is 0/1).

    Inputs:
        df: pd.DataFrame - must contain label_col
        numeric_cols: list[str]
        label_col: str

    Returns:
        pd.Series indexed by column name, sorted by |correlation| descending.
    """
    corr = df[numeric_cols + [label_col]].corr(numeric_only=True)[label_col].drop(label_col)
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def print_schema_profile(profile, correlations):
    """Console summary of the auto-detected schema — useful for a project report."""
    print("=" * 70)
    print("PHASE 1 — SCHEMA PROFILE (auto-detected from real data)")
    print("=" * 70)
    n_continuous = sum(1 for p in profile.values() if p["kind"] == "continuous")
    n_discrete = len(profile) - n_continuous
    print(f"{n_continuous} continuous columns, {n_discrete} discrete/binary columns\n")

    rows = []
    for col, p in profile.items():
        rows.append({
            "column": col, "kind": p["kind"], "nunique": p["nunique"],
            "missing_rate": round(p["missing_rate"], 3),
            "corr_with_target": round(float(correlations.get(col, np.nan)), 3),
        })
    summary_df = pd.DataFrame(rows).sort_values("corr_with_target", key=abs, ascending=False)
    print(summary_df.to_string(index=False))
    print()
