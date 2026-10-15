"""
Shared utilities: schema validation and a lightweight run-report collector
used across phases so every script prints/saves a consistent audit trail.
"""

import json
import os


def validate_schema(df, required_cols, df_name="dataframe"):
    """
    Raise a clear, actionable error if required_cols are missing from df,
    rather than letting a KeyError surface deep inside preprocessing.

    Inputs:
        df: pd.DataFrame
        required_cols: list[str]
        df_name: str - used in the error message for context

    Returns:
        None (raises ValueError on failure)
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[schema validation] '{df_name}' is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def check_duplicate_ids(df, id_col, df_name="dataframe"):
    """Warn (not raise) if an ID column has duplicate values."""
    dup_count = df[id_col].duplicated().sum()
    if dup_count > 0:
        print(f"[warning] '{df_name}' has {dup_count} duplicate '{id_col}' values.")
    return dup_count


class RunReport:
    """
    Accumulates key facts/metrics during a phase run and writes them to a
    single JSON file — useful both as an audit trail and as something to
    screenshot/attach in a project report.
    """

    def __init__(self):
        self._entries = {}

    def log(self, key, value):
        self._entries[key] = value

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._entries, f, indent=2, default=str)
        print(f"[report] Saved run report to {path}")
