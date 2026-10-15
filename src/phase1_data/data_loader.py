"""
Phase 1 — Data loading, validation, subsampling, and exploratory data
analysis, against the REAL GST-Predictive-Dataset-2024 schema (anonymized
Column0..Column21, ID, target).

The real files are large (train ~785K rows, test ~261K rows) — far more
than a k-NN graph needs for a final-year project to build and demo
quickly. This module loads the full CSVs (so EDA and the synthetic
generator's class-conditional profile are fit on the true distribution),
then stratified-subsamples down to config.MAX_TRAIN_ROWS /
config.MAX_TEST_ROWS for everything downstream. Set those to None in
config.py to use the full dataset (slower — the k-NN graph build is the
bottleneck).

Note: unlike a typical Kaggle test set, Y_Test_Data_Target.csv contains
REAL labels here (a labeled holdout, not a blind competition test set).
load_test_labels() exposes this for genuine held-out evaluation in
Phase 3 — but the modeling pipeline still treats test nodes as unlabeled
during graph construction and GNN training; the label is only used to
score predictions afterward, never to fit anything.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src import config
from src.utils.io import validate_schema, check_duplicate_ids, RunReport


def _stratified_subsample(df, max_rows, label_col, random_state):
    """Stratified subsample of df down to max_rows, preserving class balance."""
    if max_rows is None or len(df) <= max_rows:
        return df
    from sklearn.model_selection import train_test_split
    sampled, _ = train_test_split(
        df, train_size=max_rows, stratify=df[label_col], random_state=random_state
    )
    return sampled.reset_index(drop=True)


def load_gst_data(train_features_path=config.TRAIN_FEATURES_PATH,
                   train_labels_path=config.TRAIN_LABELS_PATH,
                   test_features_path=config.TEST_FEATURES_PATH,
                   max_train_rows=config.MAX_TRAIN_ROWS,
                   max_test_rows=config.MAX_TEST_ROWS,
                   random_state=config.RANDOM_STATE,
                   run_eda=True, report: RunReport = None):
    """
    Load, validate, subsample, and (optionally) profile the real GST dataset.

    Inputs:
        train_features_path, train_labels_path, test_features_path: str
        max_train_rows, max_test_rows: int or None - stratified subsample
            caps; None uses the full dataset.
        random_state: int
        run_eda: bool - if True, print + plot a full EDA report
        report: RunReport or None - if given, key facts are logged into it

    Returns:
        train_df: pd.DataFrame - merged, subsampled train features + target
        test_df: pd.DataFrame - subsampled test features (target dropped —
                 use load_test_labels() separately for true-holdout eval)

    Raises:
        FileNotFoundError - if a CSV path doesn't exist
        ValueError - if a required column is missing from any CSV
    """
    for path, name in [(train_features_path, "train features"),
                        (train_labels_path, "train labels"),
                        (test_features_path, "test features")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[Phase 1] Could not find {name} CSV at '{path}'. "
                f"Place the real dataset files there, or update the path in src/config.py."
            )

    print(f"[Phase 1] Loading real dataset (this may take a moment — files are large)...")
    train_features = pd.read_csv(train_features_path)
    train_labels = pd.read_csv(train_labels_path)
    test_features = pd.read_csv(test_features_path)

    validate_schema(train_features, [config.ID_COL] + config.NUMERIC_COLS, "train_features")
    validate_schema(train_labels, [config.ID_COL, config.LABEL_COL], "train_labels")
    validate_schema(test_features, [config.ID_COL] + config.NUMERIC_COLS, "test_features")
    check_duplicate_ids(train_features, config.ID_COL, "train_features")
    check_duplicate_ids(test_features, config.ID_COL, "test_features")

    train_df_full = train_features.merge(
        train_labels[[config.ID_COL, config.LABEL_COL]], on=config.ID_COL, how="inner"
    )
    dropped = len(train_features) - len(train_df_full)
    if dropped > 0:
        print(f"[Phase 1] Warning: {dropped} train rows had no matching label and were dropped.")

    print(f"[Phase 1] Full sizes — train: {len(train_df_full)}, test: {len(test_features)}")

    train_df = _stratified_subsample(train_df_full, max_train_rows, config.LABEL_COL, random_state)

    if max_test_rows and len(test_features) > max_test_rows:
        test_df = test_features.sample(n=max_test_rows, random_state=random_state).reset_index(drop=True)
    else:
        test_df = test_features.reset_index(drop=True)

    if max_train_rows and len(train_df_full) > max_train_rows:
        print(f"[Phase 1] Subsampled train to {len(train_df)} rows (stratified on {config.LABEL_COL}) "
              f"for tractable graph construction. Set MAX_TRAIN_ROWS=None in config.py to use all "
              f"{len(train_df_full)} rows.")
    if max_test_rows and len(test_features) > max_test_rows:
        print(f"[Phase 1] Subsampled test to {len(test_df)} rows.")

    if run_eda:
        run_eda_report(train_df, test_df, report=report)

    return train_df, test_df


def load_test_labels(test_labels_path=config.TEST_LABELS_PATH):
    """
    Load the real test-set labels (Y_Test_Data_Target.csv) for genuine
    held-out evaluation in Phase 3. Not used anywhere in training/graph
    construction — call this only when you're ready to score predictions.

    Inputs:
        test_labels_path: str

    Returns:
        pd.DataFrame with columns [ID_COL, LABEL_COL]
    """
    if not os.path.exists(test_labels_path):
        print(f"[Phase 1] No real test labels found at {test_labels_path} — "
              f"true-holdout evaluation will be skipped.")
        return None
    labels = pd.read_csv(test_labels_path)
    validate_schema(labels, [config.ID_COL, config.LABEL_COL], "test_labels")
    return labels


def run_eda_report(train_df, test_df, report: RunReport = None, plots_dir=config.PLOTS_DIR):
    """
    Print a full text summary and save EDA plots: class balance,
    missing-value heatmap, numeric feature distributions, and a
    correlation heatmap. (No categorical-frequency plot — the real
    schema has no text-categorical columns.)

    Inputs:
        train_df, test_df: pd.DataFrame
        report: RunReport or None - key facts logged here if provided
        plots_dir: str - output directory for saved figures
    """
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 70)
    print("PHASE 1 — DATASET OVERVIEW (real GST-Predictive-Dataset-2024)")
    print("=" * 70)
    print(f"Train shape (merged, subsampled): {train_df.shape}")
    print(f"Test shape (subsampled)         : {test_df.shape}\n")

    missing = train_df.isnull().sum()
    missing_pct = (missing / len(train_df) * 100).round(2)
    print("Missing values (train), columns with any missing:")
    if missing.sum() > 0:
        print(pd.DataFrame({"missing_count": missing[missing > 0],
                             "missing_pct": missing_pct[missing > 0]})
              .sort_values("missing_pct", ascending=False))
    else:
        print("  None.")
    print()

    class_balance = train_df[config.LABEL_COL].value_counts(normalize=True)
    print("Class balance (target):")
    print(class_balance.rename("proportion"), "\n")

    numeric_summary = train_df[config.NUMERIC_COLS].describe().T
    print("Numeric feature summary:")
    print(numeric_summary, "\n")

    if report is not None:
        report.log("train_shape", train_df.shape)
        report.log("test_shape", test_df.shape)
        report.log("missing_value_pct", missing_pct[missing_pct > 0].to_dict())
        report.log("class_balance", class_balance.to_dict())

    # --- Plot 1: class balance ---
    plt.figure(figsize=(5, 4))
    sns.countplot(x=train_df[config.LABEL_COL])
    plt.title("Class Balance — target")
    plt.xlabel("target (0 = compliant, 1 = risky)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase1_class_balance.png"), dpi=120)
    plt.close()

    # --- Plot 2: missing values heatmap ---
    if train_df.isnull().values.any():
        plt.figure(figsize=(10, 5))
        sns.heatmap(train_df[config.NUMERIC_COLS].isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Value Map (train)")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "phase1_missing_values.png"), dpi=120)
        plt.close()

    # --- Plot 3: numeric distributions, split by target (x-axis clipped
    # to 1st-99th percentile — several columns have rare extreme outliers) ---
    n_cols = 4
    n_rows = -(-len(config.NUMERIC_COLS) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3.5 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(config.NUMERIC_COLS):
        lo, hi = train_df[col].quantile([0.01, 0.99])
        if lo == hi:
            lo, hi = train_df[col].min(), train_df[col].max()
        sns.histplot(data=train_df, x=col, hue=config.LABEL_COL, bins=30,
                     stat="density", common_norm=False, ax=axes[i], element="step")
        axes[i].set_xlim(lo, hi)
        axes[i].set_title(col, fontsize=9)
    for j in range(len(config.NUMERIC_COLS), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase1_numeric_distributions.png"), dpi=120)
    plt.close()

    # --- Plot 4: correlation heatmap ---
    plt.figure(figsize=(10, 8))
    corr = train_df[config.NUMERIC_COLS + [config.LABEL_COL]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 6})
    plt.title("Numeric Feature Correlation (incl. target)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "phase1_correlation_heatmap.png"), dpi=120)
    plt.close()

    print(f"[Phase 1] EDA plots saved to {plots_dir}/ "
          f"(class_balance, missing_values, numeric_distributions, correlation_heatmap)\n")
