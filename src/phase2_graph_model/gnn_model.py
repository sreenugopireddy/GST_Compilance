"""
Phase 2 — GNN model definition.

A 2-layer GraphSAGE encoder + linear classification head. GraphSAGE is
chosen (over a transductive GCN) specifically because it's inductive: the
same trained weights can embed and classify brand-new nodes (e.g. GSTINs
uploaded later through the Streamlit app) without retraining, as long as
those new nodes are attached to the graph via edges (see graph_builder.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from src import config


class GST_GraphSAGE(nn.Module):
    """
    forward() returns (logits, embeddings): embeddings are the penultimate
    node representations, reused for Isolation Forest and explainability.
    """

    def __init__(self, in_dim, hidden_dim=config.GNN_HIDDEN_DIM,
                 embed_dim=config.GNN_EMBED_DIM, dropout=config.GNN_DROPOUT,
                 n_classes=2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.n_classes = n_classes

        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        embeddings = self.conv2(h, edge_index)
        embeddings = F.relu(embeddings)
        logits = self.classifier(embeddings)
        return logits, embeddings

    def get_config(self):
        """Config dict needed to reconstruct this model architecture later."""
        return {
            "in_dim": self.in_dim,
            "hidden_dim": self.hidden_dim,
            "embed_dim": self.embed_dim,
            "dropout": self.dropout,
            "n_classes": self.n_classes,
        }


def create_gnn_model(in_dim, **kwargs):
    """Instantiate a GST_GraphSAGE model on the configured device."""
    return GST_GraphSAGE(in_dim=in_dim, **kwargs).to(config.DEVICE)
