"""
Phase 2 — Graph construction.

Builds an undirected k-NN similarity graph over GSTINs in feature space,
optionally densified with shared-attribute (same state / same sector) edges.
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from src import config


def build_knn_graph(X_all, k=config.KNN_K, extra_attr_cols=None, attr_df=None,
                     random_state=config.RANDOM_STATE):
    """
    Build an undirected k-NN graph over all nodes.

    Inputs:
        X_all: np.ndarray, shape (N, F) - node feature matrix.
        k: int - neighbors per node.
        extra_attr_cols: list[str] or None - categorical columns to also
            connect nodes on (e.g. ["state", "sector"]).
        attr_df: pd.DataFrame or None - raw dataframe aligned with X_all rows,
            required if extra_attr_cols is given.
        random_state: int

    Returns:
        edge_index: torch.LongTensor, shape (2, E) - PyG edge format.
    """
    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
    nn_model.fit(X_all)
    _, neighbor_idx = nn_model.kneighbors(X_all)

    src, dst = [], []
    for node_i in range(X_all.shape[0]):
        for node_j in neighbor_idx[node_i][1:]:  # skip self
            src.append(node_i)
            dst.append(node_j)

    if extra_attr_cols and attr_df is not None:
        rng = np.random.default_rng(random_state)
        for col in extra_attr_cols:
            groups = attr_df.groupby(col).indices
            for _, idxs in groups.items():
                idxs = np.array(idxs)
                if len(idxs) < 2:
                    continue
                for node_i in idxs:
                    peers = rng.choice(idxs[idxs != node_i],
                                        size=min(3, len(idxs) - 1), replace=False)
                    for p in peers:
                        src.append(node_i)
                        dst.append(p)

    edges = np.array(list(set(zip(src, dst)) | set(zip(dst, src))))
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    print(f"[Phase 2] Built graph: {X_all.shape[0]} nodes, {edge_index.shape[1]} directed edges (k={k}).")
    return edge_index


def attach_new_nodes_to_reference_graph(new_X, reference_X, k=config.KNN_K):
    """
    Inductive attachment: connect new (previously unseen) node feature
    vectors to their k nearest neighbors within a fixed reference pool,
    and to each other. Used at inference time in the Streamlit app so new
    uploaded GSTINs can be scored by the trained (inductive) GraphSAGE
    model without retraining.

    Inputs:
        new_X: np.ndarray (M, F) - features of new nodes, to be placed at
            indices [n_ref, n_ref + M) in the combined graph.
        reference_X: np.ndarray (n_ref, F) - features of the fixed reference
            node pool, at indices [0, n_ref).
        k: int - neighbors per new node.

    Returns:
        combined_X: np.ndarray (n_ref + M, F)
        edge_index: torch.LongTensor (2, E) over the combined node set
        new_node_offset: int - index where new nodes start in combined_X
    """
    n_ref = reference_X.shape[0]
    combined_X = np.vstack([reference_X, new_X])

    nn_model = NearestNeighbors(n_neighbors=min(k, n_ref), algorithm="auto", metric="euclidean")
    nn_model.fit(reference_X)
    _, neighbor_idx = nn_model.kneighbors(new_X)

    src, dst = [], []
    for i in range(new_X.shape[0]):
        new_node_id = n_ref + i
        for ref_node_id in neighbor_idx[i]:
            src.append(new_node_id)
            dst.append(int(ref_node_id))

    # Connect new nodes to each other too (small k, so uploaded batches
    # that resemble each other reinforce one another's signal).
    if new_X.shape[0] > 1:
        nn_new = NearestNeighbors(n_neighbors=min(k, new_X.shape[0]), algorithm="auto")
        nn_new.fit(new_X)
        _, new_neighbor_idx = nn_new.kneighbors(new_X)
        for i in range(new_X.shape[0]):
            for j in new_neighbor_idx[i][1:]:
                src.append(n_ref + i)
                dst.append(n_ref + int(j))

    edges = np.array(list(set(zip(src, dst)) | set(zip(dst, src))))
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    return combined_X, edge_index, n_ref
