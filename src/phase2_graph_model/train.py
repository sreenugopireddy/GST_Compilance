"""
Phase 2 — GNN training loop.

Trains the GraphSAGE classifier with class-weighted cross-entropy and
early stopping on validation ROC-AUC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from src import config


def train_gnn(model, data, train_idx, val_idx, epochs=config.GNN_EPOCHS,
              lr=config.GNN_LR, weight_decay=config.GNN_WEIGHT_DECAY,
              patience=config.EARLY_STOP_PATIENCE, verbose=True):
    """
    Train the GNN with Adam + early stopping on validation ROC-AUC.

    Inputs:
        model: GST_GraphSAGE
        data: torch_geometric.data.Data (with .x, .edge_index, .y)
        train_idx, val_idx: np.ndarray - labeled node indices
        epochs, lr, weight_decay, patience: hyperparameters

    Returns:
        model: trained model with best-val-AUC weights restored
        history: dict of per-epoch metrics
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_counts = torch.bincount(data.y[train_idx].long())
    class_weights = (class_counts.sum() / (2.0 * class_counts.float())).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=config.DEVICE)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=config.DEVICE)

    best_val_auc = -1
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(data.x, data.edge_index)
        loss = criterion(logits[train_idx_t], data.y[train_idx_t].long())
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, _ = model(data.x, data.edge_index)
            val_logits = logits[val_idx_t]
            val_loss = criterion(val_logits, data.y[val_idx_t].long())
            val_probs = F.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
            val_labels = data.y[val_idx_t].cpu().numpy()
            val_acc = accuracy_score(val_labels, (val_probs >= 0.5).astype(int))
            val_auc = roc_auc_score(val_labels, val_probs)

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"[Phase 2] Epoch {epoch:3d} | train_loss {loss.item():.4f} | "
                  f"val_loss {val_loss.item():.4f} | val_acc {val_acc:.4f} | val_auc {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"[Phase 2] Early stopping at epoch {epoch} (best val_auc={best_val_auc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def get_node_embeddings(model, data):
    """Run a forward pass and return (logits, embeddings) as numpy arrays."""
    model.eval()
    with torch.no_grad():
        logits, embeddings = model(data.x, data.edge_index)
    return logits.cpu().numpy(), embeddings.cpu().numpy()
