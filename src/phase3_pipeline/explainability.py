"""
Phase 3 — Explainability.

Feature-level explanations via SHAP, and a lightweight, dependency-light
GNNExplainer-style node explanation (learns a soft mask over a node's
incident edges and input features) — written custom because the official
torch_geometric.explain API has changed across PyG versions and isn't
stable to depend on for a submitted project.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src import config


def explain_features_shap(model, X_all, feature_names, background_idx, explain_idx):
    """
    Feature-level explainability using SHAP's KernelExplainer. Explains how
    a node's OWN features drive its GNN risk probability, holding graph
    structure fixed (a standard, fast practical simplification for GNNs —
    it does not capture neighbor-message contributions; use
    explain_node_gnnexplainer for that).

    Inputs:
        model: trained GST_GraphSAGE
        X_all: np.ndarray - full node feature matrix
        feature_names: list[str]
        background_idx, explain_idx: np.ndarray - node indices (keep small,
            SHAP's KernelExplainer is slow)

    Returns:
        shap_values: np.ndarray, shape (len(explain_idx), n_features)
    """
    import shap

    empty_edges = torch.empty((2, 0), dtype=torch.long, device=config.DEVICE)

    def predict_fn(X_subset):
        x_tensor = torch.tensor(X_subset, dtype=torch.float32, device=config.DEVICE)
        with torch.no_grad():
            h = F.relu(model.conv1(x_tensor, empty_edges))
            emb = F.relu(model.conv2(h, empty_edges))
            logits = model.classifier(emb)
        return F.softmax(logits, dim=1)[:, 1].cpu().numpy()

    background = X_all[background_idx]
    to_explain = X_all[explain_idx]
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(to_explain, nsamples=100)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features = sorted(zip(feature_names, mean_abs_shap), key=lambda t: -t[1])[:10]
    print("Top contributing features (mean |SHAP value|):")
    for name, val in top_features:
        print(f"  {name:30s} {val:.4f}")

    return shap_values


def explain_node_gnnexplainer(model, data, node_idx, feature_names=None,
                               epochs=100, lr=0.05, top_k_neighbors=5, top_k_features=5):
    """
    Learn a soft mask over a node's incident edges and input features so
    that the masked subgraph/features still reproduce (close to) the
    original prediction — the surviving mask weights indicate which
    neighbors and features mattered most.

    Inputs:
        model: trained GST_GraphSAGE (frozen during explanation)
        data: torch_geometric.data.Data
        node_idx: int - node to explain
        feature_names: list[str] or None - for readable output
        epochs, lr: mask optimization settings
        top_k_neighbors, top_k_features: how many to report

    Returns:
        dict with "important_neighbors": list[(neighbor_idx, weight)],
                  "top_features": list[(name_or_idx, weight)]
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    edge_index = data.edge_index
    incident_mask = (edge_index[0] == node_idx) | (edge_index[1] == node_idx)
    incident_edges = edge_index[:, incident_mask]
    n_incident = incident_edges.shape[1]

    if n_incident == 0:
        return {"important_neighbors": [], "top_features": []}

    edge_mask = nn.Parameter(torch.randn(n_incident, device=config.DEVICE) * 0.1)
    feat_mask = nn.Parameter(torch.randn(data.x.shape[1], device=config.DEVICE) * 0.1)
    optimizer = torch.optim.Adam([edge_mask, feat_mask], lr=lr)

    with torch.no_grad():
        full_logits, _ = model(data.x, data.edge_index)
        target_prob = F.softmax(full_logits[node_idx], dim=0)

    for _ in range(epochs):
        optimizer.zero_grad()
        sig_edge_mask = torch.sigmoid(edge_mask)
        sig_feat_mask = torch.sigmoid(feat_mask)

        keep = sig_edge_mask > 0.5
        masked_incident = incident_edges[:, keep] if keep.any() else incident_edges[:, :1]
        other_edges = edge_index[:, ~incident_mask]
        masked_edge_index = torch.cat([other_edges, masked_incident], dim=1)

        masked_x = data.x * sig_feat_mask.unsqueeze(0)
        logits, _ = model(masked_x, masked_edge_index)
        pred_prob = F.softmax(logits[node_idx], dim=0)

        fidelity_loss = F.kl_div(pred_prob.log(), target_prob, reduction="sum")
        sparsity_loss = sig_edge_mask.mean() + sig_feat_mask.mean()
        loss = fidelity_loss + 0.01 * sparsity_loss
        loss.backward()
        optimizer.step()

    final_edge_weights = torch.sigmoid(edge_mask).detach().cpu().numpy()
    neighbor_ids = []
    for i in range(n_incident):
        a, b = incident_edges[0, i].item(), incident_edges[1, i].item()
        neighbor_ids.append(b if a == node_idx else a)

    ranked_neighbors = sorted(zip(neighbor_ids, final_edge_weights), key=lambda t: -t[1])[:top_k_neighbors]

    feature_weights = torch.sigmoid(feat_mask).detach().cpu().numpy()
    top_feat_idx = np.argsort(-feature_weights)[:top_k_features]
    if feature_names is not None:
        top_features = [(feature_names[i], float(feature_weights[i])) for i in top_feat_idx]
    else:
        top_features = [(int(i), float(feature_weights[i])) for i in top_feat_idx]

    return {"important_neighbors": ranked_neighbors, "top_features": top_features}
