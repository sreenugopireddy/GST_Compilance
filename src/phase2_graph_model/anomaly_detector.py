"""
Phase 2 — Isolation Forest anomaly detection on GNN embeddings.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from src import config


def run_isolation_forest_on_embeddings(embeddings, y_all=None,
                                        n_estimators=config.ISO_FOREST_N_ESTIMATORS,
                                        contamination=config.ISO_FOREST_CONTAMINATION,
                                        random_state=config.RANDOM_STATE):
    """
    Fit an Isolation Forest on GNN node embeddings and compute normalized
    anomaly scores.

    Inputs:
        embeddings: np.ndarray (N, embed_dim)
        y_all: np.ndarray or None - risk_flag per node (NaN where unknown);
               used only to report alignment, never for fitting.
        n_estimators, contamination, random_state: IsolationForest hyperparameters.

    Returns:
        iso_forest: fitted IsolationForest
        anomaly_scores: np.ndarray (N,) in [0, 1], higher = more anomalous
        raw_score_range: tuple(min, max) of the raw (pre-normalization)
            scores — pass this to score_new_embeddings() at inference time
            so new points are normalized on the same scale.
    """
    iso_forest = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                                  random_state=random_state, n_jobs=-1)
    iso_forest.fit(embeddings)
    raw_scores = -iso_forest.score_samples(embeddings)
    raw_lo, raw_hi = raw_scores.min(), raw_scores.max()
    anomaly_scores = (raw_scores - raw_lo) / (raw_hi - raw_lo + 1e-9)

    if y_all is not None:
        labeled_mask = ~np.isnan(y_all)
        if labeled_mask.sum() > 0:
            auc = roc_auc_score(y_all[labeled_mask], anomaly_scores[labeled_mask])
            print(f"[Phase 2] Isolation Forest anomaly-score vs risk_flag AUC: {auc:.4f}")

    return iso_forest, anomaly_scores, (float(raw_lo), float(raw_hi))


def score_new_embeddings(iso_forest, new_embeddings, reference_scores_range):
    """
    Score new (inference-time) embeddings with an already-fitted Isolation
    Forest, normalizing against the SAME min/max used at training time so
    scores stay comparable across runs.

    Inputs:
        iso_forest: fitted IsolationForest
        new_embeddings: np.ndarray (M, embed_dim)
        reference_scores_range: tuple(min_raw_score, max_raw_score) captured
            at training time.

    Returns:
        anomaly_scores: np.ndarray (M,) in [0, 1] (may exceed slightly if
        new points are more extreme than anything seen in training).
    """
    raw_scores = -iso_forest.score_samples(new_embeddings)
    lo, hi = reference_scores_range
    return np.clip((raw_scores - lo) / (hi - lo + 1e-9), 0.0, 1.0)
