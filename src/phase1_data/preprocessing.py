"""
Phase 1 — Preprocessing and real+synthetic combination.

Cleans, clips outliers, encodes, and scales GST taxpayer feature tables,
and produces the final combined (real + synthetic + test) feature matrix
that Phase 2 builds the graph on. The encoder/scaler/clip-bounds are always
fit ONLY on the training split and passed explicitly to every downstream
transform call — no hidden/mutable state — to avoid leakage into
validation/holdout/test/new-upload nodes.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from src import config
from src.utils.io import RunReport


def preprocess_gst_data(df, encoder=None, scaler=None, clip_bounds=None, fit=True,
                         categorical_cols=config.CATEGORICAL_COLS,
                         numeric_cols=config.NUMERIC_COLS,
                         clip_outliers=True, clip_quantiles=(0.01, 0.99)):
    """
    Clean, clip, encode, and scale a GST taxpayer feature table.

    Inputs:
        df: pd.DataFrame - raw features.
        encoder: fitted sklearn OneHotEncoder, or None (created if fit=True).
        scaler: fitted sklearn StandardScaler, or None (created if fit=True).
        clip_bounds: dict[str, (low, high)] or None - per-column winsorization
            bounds. Computed from this df when fit=True; required (and reused
            as-is, never recomputed) when fit=False and clip_outliers=True.
        fit: bool - fit new transformers/bounds on this df if True; else
             transform only, reusing what's passed in. Always use fit=False
             for validation/holdout/test/new-upload data so nothing new ever
             leaks into fitting.
        categorical_cols, numeric_cols: list[str]
        clip_outliers: bool - winsorize numeric columns before scaling, so a
             handful of extreme rows (e.g. a data-entry error with
             turnover=1e12) can't dominate StandardScaler.
        clip_quantiles: tuple(low, high) - quantile bounds used when fit=True.

    Returns:
        X: np.ndarray - processed feature matrix (numeric + one-hot).
        encoder, scaler: the fitted transformers, to be reused downstream.
        clip_bounds: dict[str, (low, high)] - to be reused downstream
            (None if clip_outliers=False).
    """
    df = df.copy()

    # --- Missing value handling: median for numeric, mode for categorical ---
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        if col in df.columns:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    # --- Outlier clipping (winsorization), bounds always computed on TRAIN ---
    if clip_outliers:
        if fit:
            clip_bounds = {
                col: (df[col].quantile(clip_quantiles[0]), df[col].quantile(clip_quantiles[1]))
                for col in numeric_cols if col in df.columns
            }
        elif clip_bounds is None:
            raise ValueError("clip_bounds must be provided when fit=False and clip_outliers=True")
        for col, (lo, hi) in clip_bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lo, upper=hi)

    # --- Categorical encoding (one-hot, unknown categories safely ignored) ---
    if fit:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_encoded = encoder.fit_transform(df[categorical_cols])
    else:
        cat_encoded = encoder.transform(df[categorical_cols])

    # --- Numeric scaling ---
    if fit:
        scaler = StandardScaler()
        num_scaled = scaler.fit_transform(df[numeric_cols])
    else:
        num_scaled = scaler.transform(df[numeric_cols])

    X = np.hstack([num_scaled, cat_encoded])
    return X, encoder, scaler, clip_bounds


def get_feature_names(encoder, categorical_cols=config.CATEGORICAL_COLS,
                       numeric_cols=config.NUMERIC_COLS):
    """Return column names aligned with preprocess_gst_data()'s output columns."""
    return list(numeric_cols) + list(encoder.get_feature_names_out(categorical_cols))


def combine_and_preprocess_real_and_synthetic(train_df, test_df, synth_df,
                                               random_state=config.RANDOM_STATE,
                                               report: RunReport = None):
    """
    Concatenate real (train) and synthetic taxpayers, fit one preprocessing
    pipeline on the training split only, and transform val/holdout/test
    with the same fitted encoder/scaler/clip-bounds.

    Inputs:
        train_df: pd.DataFrame - real labeled training data.
        test_df: pd.DataFrame - real unlabeled test data.
        synth_df: pd.DataFrame - synthetic labeled data.
        random_state: int
        report: RunReport or None - key facts logged here if provided

    Returns:
        dict with:
            X_all, y_all, gstin_all, is_synthetic_all: np.ndarray
            train_idx, val_idx, holdout_idx, test_idx: np.ndarray (node indices)
            encoder, scaler, clip_bounds: fitted transformers/bounds, reusable
                on new data (e.g. by the Streamlit app) via preprocess_gst_data
                with fit=False.
            raw_df_all: pd.DataFrame - raw (unencoded) rows aligned with X_all
            feature_names: list[str]
    """
    real_train = train_df.copy()
    real_train["is_synthetic"] = 0
    test_local = test_df.copy()
    test_local["is_synthetic"] = 0
    if config.LABEL_COL not in test_local.columns:
        test_local[config.LABEL_COL] = np.nan

    combined_labeled = pd.concat([real_train, synth_df], axis=0, ignore_index=True)

    # Node-level stratified split within the labeled pool: 70% train /
    # 15% val (used for early stopping) / 15% holdout (used for final,
    # untouched-during-training evaluation in Phase 3).
    labeled_idx = np.arange(len(combined_labeled))
    train_idx, temp_idx = train_test_split(
        labeled_idx, test_size=0.3, random_state=random_state,
        stratify=combined_labeled[config.LABEL_COL])
    val_idx, holdout_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=random_state,
        stratify=combined_labeled.loc[temp_idx, config.LABEL_COL])

    # Fit encoder/scaler/clip-bounds ONLY on train_idx — this is the single
    # most important leakage-prevention step in the whole pipeline.
    _, encoder, scaler, clip_bounds = preprocess_gst_data(combined_labeled.iloc[train_idx], fit=True)

    X_labeled, _, _, _ = preprocess_gst_data(
        combined_labeled, encoder=encoder, scaler=scaler, clip_bounds=clip_bounds, fit=False)
    X_test, _, _, _ = preprocess_gst_data(
        test_local, encoder=encoder, scaler=scaler, clip_bounds=clip_bounds, fit=False)

    X_all = np.vstack([X_labeled, X_test])
    y_all = np.concatenate([combined_labeled[config.LABEL_COL].values,
                             np.full(len(test_local), np.nan)])
    gstin_all = np.concatenate([combined_labeled[config.ID_COL].values, test_local[config.ID_COL].values])
    is_synthetic_all = np.concatenate([combined_labeled["is_synthetic"].values,
                                        test_local["is_synthetic"].values])
    raw_df_all = pd.concat([combined_labeled, test_local], axis=0, ignore_index=True)

    n_labeled = len(combined_labeled)
    test_node_idx = np.arange(n_labeled, n_labeled + len(test_local))

    assert not np.isnan(X_all).any(), "NaNs leaked into X_all after preprocessing"
    assert X_all.shape[0] == len(gstin_all) == len(y_all) == len(is_synthetic_all)

    print(f"[Phase 1] Combined dataset: {n_labeled} labeled nodes "
          f"(real={len(real_train)}, synthetic={len(synth_df)}) "
          f"+ {len(test_local)} unlabeled test nodes. X_all shape: {X_all.shape}")
    print(f"[Phase 1] Split sizes — train: {len(train_idx)}, val: {len(val_idx)}, "
          f"holdout: {len(holdout_idx)}, test: {len(test_node_idx)}")

    if report is not None:
        report.log("n_labeled_nodes", int(n_labeled))
        report.log("n_test_nodes", int(len(test_local)))
        report.log("X_all_shape", list(X_all.shape))
        report.log("split_sizes", {"train": len(train_idx), "val": len(val_idx),
                                    "holdout": len(holdout_idx), "test": len(test_node_idx)})

    return {
        "X_all": X_all,
        "y_all": y_all,
        "gstin_all": gstin_all,
        "is_synthetic_all": is_synthetic_all,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "holdout_idx": holdout_idx,
        "test_idx": test_node_idx,
        "encoder": encoder,
        "scaler": scaler,
        "clip_bounds": clip_bounds,
        "raw_df_all": raw_df_all,
        "feature_names": get_feature_names(encoder),
    }
