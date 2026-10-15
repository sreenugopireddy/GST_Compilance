"""
Phase 3 — Risk score fusion.
"""

from src import config


def compute_final_risk_score(gnn_prob, iforest_score,
                              weight_gnn=config.RISK_WEIGHT_GNN,
                              weight_iforest=config.RISK_WEIGHT_IFOREST):
    """
    Fuse the GNN risk probability and Isolation Forest anomaly score into a
    single interpretable risk score in [0, 1].

    Inputs:
        gnn_prob: float or np.ndarray - GNN predicted P(risk_flag==1), in [0,1].
        iforest_score: float or np.ndarray - normalized anomaly score, in [0,1].
        weight_gnn, weight_iforest: float - fusion weights (must sum to 1).

    Returns:
        final_score: float or np.ndarray in [0, 1]
    """
    assert abs((weight_gnn + weight_iforest) - 1.0) < 1e-6, "weights must sum to 1"
    return weight_gnn * gnn_prob + weight_iforest * iforest_score


def risk_label_from_score(score, threshold=config.RISK_LABEL_THRESHOLD):
    """Binary risk label from a fused score."""
    return int(score >= threshold)


def risk_tier_from_score(score):
    """Human-readable risk tier for UI display."""
    if score >= 0.75:
        return "High Risk"
    elif score >= 0.5:
        return "Elevated Risk"
    elif score >= 0.25:
        return "Low Risk"
    return "Compliant"
