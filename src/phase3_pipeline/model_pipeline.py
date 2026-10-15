"""
Phase 3 — End-to-end pipeline.

GSTComplianceRiskModel wraps the full Phase 1 -> Phase 2 -> Phase 3 flow:
load -> preprocess -> build graph -> train GNN -> train Isolation Forest ->
fuse risk scores. It also knows how to save/load every fitted artifact
(encoder, scaler, GNN weights, Isolation Forest, a reference node pool) so
that a completely separate process — the Streamlit app — can load a
trained model and score brand-new, previously-unseen GSTINs without any
retraining, by attaching them to the reference graph (see graph_builder.
attach_new_nodes_to_reference_graph).
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from src import config
from src.phase1_data.data_loader import load_gst_data, load_test_labels
from src.phase1_data.synthetic_generator import fit_synthetic_generator, generate_synthetic_gst_data
from src.phase1_data.preprocessing import preprocess_gst_data, combine_and_preprocess_real_and_synthetic
from src.phase2_graph_model.graph_builder import build_knn_graph, attach_new_nodes_to_reference_graph
from src.phase2_graph_model.gnn_model import GST_GraphSAGE, create_gnn_model
from src.phase2_graph_model.train import train_gnn, get_node_embeddings
from src.phase2_graph_model.anomaly_detector import run_isolation_forest_on_embeddings, score_new_embeddings
from src.phase3_pipeline.evaluation import evaluate_gnn_classifier, precision_recall_at_k
from src.phase3_pipeline.explainability import explain_node_gnnexplainer
from src.phase3_pipeline.risk_scoring import compute_final_risk_score, risk_tier_from_score


class GSTComplianceRiskModel:
    """
    Usage (training):
        model = GSTComplianceRiskModel()
        model.fit(train_features_path, train_labels_path, test_features_path)
        model.save()

    Usage (inference elsewhere, e.g. Streamlit, no retraining needed):
        model = GSTComplianceRiskModel.load()
        results_df = model.predict_for_new_data(uploaded_raw_df)
    """

    def __init__(self):
        self.encoder = None
        self.scaler = None
        self.clip_bounds = None
        self.gnn = None
        self.iso_forest = None
        self.raw_score_range = None
        self.feature_names = None
        self.reference_X = None       # feature matrix for a saved reference pool
        self.reference_gstin = None
        self.data = None              # full training-time graph (kept only in-process)
        self.gstin_all = None
        self.y_all = None
        self.embeddings = None
        self.gnn_probs = None
        self.anomaly_scores = None
        self._bundle = None

    # ---------------------------------------------------------------- fit
    def fit(self, train_features_path=config.TRAIN_FEATURES_PATH,
            train_labels_path=config.TRAIN_LABELS_PATH,
            test_features_path=config.TEST_FEATURES_PATH,
            max_train_rows=config.MAX_TRAIN_ROWS, max_test_rows=config.MAX_TEST_ROWS,
            evaluate_on_real_test=True):
        """
        Run the full Phase 1 -> Phase 2 -> Phase 3 pipeline in-process.

        Inputs:
            train_features_path, train_labels_path, test_features_path: str
            max_train_rows, max_test_rows: int or None - stratified subsample
                caps applied when loading the real dataset (see data_loader).
            evaluate_on_real_test: bool - if True and Y_Test_Data_Target.csv
                is present, also reports metrics on the REAL held-out test
                labels (never used for training/graph construction, only
                for scoring predictions afterward).
        """
        train_df, test_df = load_gst_data(
            train_features_path, train_labels_path, test_features_path,
            max_train_rows=max_train_rows, max_test_rows=max_test_rows)

        generator_stats = fit_synthetic_generator(train_df)
        synth_df = generate_synthetic_gst_data(generator_stats)
        bundle = combine_and_preprocess_real_and_synthetic(train_df, test_df, synth_df)

        self.encoder, self.scaler = bundle["encoder"], bundle["scaler"]
        self.clip_bounds = bundle["clip_bounds"]
        self.gstin_all, self.y_all = bundle["gstin_all"], bundle["y_all"]
        self.feature_names = bundle["feature_names"]

        edge_index = build_knn_graph(bundle["X_all"], k=config.KNN_K)
        x_tensor = torch.tensor(bundle["X_all"], dtype=torch.float32, device=config.DEVICE)
        y_tensor = torch.tensor(np.nan_to_num(bundle["y_all"], nan=-1), dtype=torch.float32, device=config.DEVICE)
        self.data = Data(x=x_tensor, edge_index=edge_index.to(config.DEVICE), y=y_tensor)

        self.gnn = create_gnn_model(in_dim=bundle["X_all"].shape[1])
        self.gnn, history = train_gnn(self.gnn, self.data, bundle["train_idx"], bundle["val_idx"])

        logits, self.embeddings = get_node_embeddings(self.gnn, self.data)
        self.gnn_probs = F.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]

        self.iso_forest, self.anomaly_scores, self.raw_score_range = run_isolation_forest_on_embeddings(
            self.embeddings, y_all=bundle["y_all"])

        holdout_idx = bundle["holdout_idx"]
        evaluate_gnn_classifier(bundle["y_all"][holdout_idx], self.gnn_probs[holdout_idx],
                                 title_suffix="(internal labeled holdout — real + synthetic mix)")
        precision_recall_at_k(np.nan_to_num(bundle["y_all"][holdout_idx]),
                               self.anomaly_scores[holdout_idx], k=50)

        # --- Genuine held-out evaluation on the REAL test labels, if available.
        # These labels were never used for training or graph construction —
        # test nodes were treated as unlabeled throughout. This is the
        # number that actually matters for judging real-world performance,
        # since the internal holdout above is diluted with synthetic rows.
        if evaluate_on_real_test:
            test_labels_df = load_test_labels()
            if test_labels_df is not None:
                test_idx = bundle["test_idx"]
                test_gstin = bundle["gstin_all"][test_idx]
                true_map = test_labels_df.set_index(config.ID_COL)[config.LABEL_COL]
                y_true_test = true_map.reindex(test_gstin)
                valid = y_true_test.notna().values
                if valid.sum() > 0:
                    y_true_valid = y_true_test.values[valid].astype(int)
                    gnn_probs_valid = self.gnn_probs[test_idx][valid]
                    anomaly_valid = self.anomaly_scores[test_idx][valid]
                    evaluate_gnn_classifier(y_true_valid, gnn_probs_valid,
                                             title_suffix="(REAL held-out test set — never seen in training)",
                                             plot_name="gnn_evaluation_real_test.png")
                    precision_recall_at_k(y_true_valid, anomaly_valid, k=50)
                else:
                    print("[Phase 3] No matching real test labels found for sampled test nodes — skipping.")

        # Keep a bounded reference node pool (prefer real, labeled nodes)
        # for inductive scoring of new GSTINs later.
        labeled_real_idx = np.where(
            (~np.isnan(bundle["y_all"])) & (bundle["is_synthetic_all"] == 0))[0]
        pool_idx = labeled_real_idx if len(labeled_real_idx) > 0 else bundle["train_idx"]
        if len(pool_idx) > config.REFERENCE_POOL_SIZE:
            rng = np.random.default_rng(config.RANDOM_STATE)
            pool_idx = rng.choice(pool_idx, size=config.REFERENCE_POOL_SIZE, replace=False)
        self.reference_X = bundle["X_all"][pool_idx]
        self.reference_gstin = bundle["gstin_all"][pool_idx]

        self._bundle = bundle
        print("\n[Phase 3] GSTComplianceRiskModel: fit complete.")
        return self

    # -------------------------------------------------- predict (known nodes)
    def predict_for_gstin_list(self, gstin_ids, explain=True):
        """
        Score GSTINs already present in the fitted training-time graph.

        Inputs:
            gstin_ids: list[str]
            explain: bool - run node-level GNNExplainer-style explanation

        Returns:
            pd.DataFrame: gstin, risk_label, risk_tier, final_risk_score,
                          gnn_prob, anomaly_score, top_features
        """
        rows = []
        for gid in gstin_ids:
            matches = np.where(self.gstin_all == gid)[0]
            if len(matches) == 0:
                print(f"Warning: GSTIN {gid} not in the fitted graph — skipping.")
                continue
            idx = int(matches[0])
            rows.append(self._row_from_scores(
                gid, float(self.gnn_probs[idx]), float(self.anomaly_scores[idx]),
                node_idx=idx if explain else None))
        return pd.DataFrame(rows)

    # ------------------------------------------------- predict (new/uploaded)
    def predict_for_new_data(self, raw_df, explain=False):
        """
        Score brand-new rows (e.g. an uploaded CSV) that were never part of
        training. New nodes are attached inductively to the saved reference
        graph via k-NN edges, embedded with the frozen trained GraphSAGE
        weights, then scored with the frozen Isolation Forest.

        Inputs:
            raw_df: pd.DataFrame with the real schema's numeric columns
                    (Column0..Column21 — see src/config.py NUMERIC_COLS).
                    An ID column is optional; target is not required/used.
            explain: bool - run node-level explanation for each row (slower)

        Returns:
            pd.DataFrame: ID, risk_label, risk_tier, final_risk_score,
                          gnn_prob, anomaly_score, top_features
        """
        missing_cols = [c for c in config.NUMERIC_COLS if c not in raw_df.columns]
        if missing_cols:
            raise ValueError(f"Uploaded data is missing required columns: {missing_cols}")

        new_X, _, _, _ = preprocess_gst_data(
            raw_df, encoder=self.encoder, scaler=self.scaler,
            clip_bounds=self.clip_bounds, fit=False)
        combined_X, edge_index, n_ref = attach_new_nodes_to_reference_graph(
            new_X, self.reference_X, k=config.KNN_K)

        x_tensor = torch.tensor(combined_X, dtype=torch.float32, device=config.DEVICE)
        inference_data = Data(x=x_tensor, edge_index=edge_index.to(config.DEVICE))

        self.gnn.eval()
        with torch.no_grad():
            logits, embeddings = self.gnn(inference_data.x, inference_data.edge_index)
        gnn_probs_all = F.softmax(logits, dim=1).cpu().numpy()[:, 1]
        embeddings_np = embeddings.cpu().numpy()

        new_embeddings = embeddings_np[n_ref:]
        new_gnn_probs = gnn_probs_all[n_ref:]
        new_anomaly_scores = score_new_embeddings(self.iso_forest, new_embeddings, self.raw_score_range)

        gstin_col = raw_df[config.ID_COL].values if config.ID_COL in raw_df.columns \
            else [f"UPLOADED_{i}" for i in range(len(raw_df))]

        rows = []
        for i, gid in enumerate(gstin_col):
            node_idx = n_ref + i if explain else None
            explain_data = inference_data if explain else None
            rows.append(self._row_from_scores(
                gid, float(new_gnn_probs[i]), float(new_anomaly_scores[i]),
                node_idx=node_idx, explain_data=explain_data))
        return pd.DataFrame(rows)

    def _row_from_scores(self, id_value, gnn_prob, anomaly_score, node_idx=None, explain_data=None):
        final_score = compute_final_risk_score(gnn_prob, anomaly_score)
        row = {
            config.ID_COL: id_value,
            "final_risk_score": round(final_score, 4),
            "risk_label": int(final_score >= config.RISK_LABEL_THRESHOLD),
            "risk_tier": risk_tier_from_score(final_score),
            "gnn_prob": round(gnn_prob, 4),
            "anomaly_score": round(anomaly_score, 4),
            "top_features": [],
        }
        if node_idx is not None:
            data_for_explain = explain_data if explain_data is not None else self.data
            explanation = explain_node_gnnexplainer(
                self.gnn, data_for_explain, node_idx,
                feature_names=self.feature_names, epochs=50)
            row["top_features"] = [name for name, _ in explanation["top_features"]]
        return row

    # -------------------------------------------------------------- save/load
    def save(self, models_dir=config.MODELS_DIR):
        """Persist every fitted artifact needed for later inference."""
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(self.encoder, config.ENCODER_PATH)
        joblib.dump(self.scaler, config.SCALER_PATH)
        joblib.dump(self.iso_forest, config.ISO_FOREST_PATH)

        torch.save(self.gnn.state_dict(), config.GNN_WEIGHTS_PATH)
        with open(config.GNN_CONFIG_PATH, "w") as f:
            json.dump(self.gnn.get_config(), f)
        with open(config.FEATURE_NAMES_PATH, "w") as f:
            json.dump({
                "feature_names": self.feature_names,
                "raw_score_range": list(self.raw_score_range),
                "clip_bounds": {col: [float(lo), float(hi)]
                                for col, (lo, hi) in self.clip_bounds.items()},
            }, f)

        np.savez_compressed(
            config.REFERENCE_GRAPH_PATH,
            reference_X=self.reference_X,
            reference_gstin=self.reference_gstin,
        )
        print(f"[Phase 3] Model artifacts saved to {models_dir}")

    @classmethod
    def load(cls, models_dir=config.MODELS_DIR):
        """Reconstruct a fitted GSTComplianceRiskModel from saved artifacts."""
        model = cls()
        model.encoder = joblib.load(config.ENCODER_PATH)
        model.scaler = joblib.load(config.SCALER_PATH)
        model.iso_forest = joblib.load(config.ISO_FOREST_PATH)

        with open(config.GNN_CONFIG_PATH) as f:
            gnn_config = json.load(f)
        model.gnn = GST_GraphSAGE(**gnn_config).to(config.DEVICE)
        model.gnn.load_state_dict(torch.load(config.GNN_WEIGHTS_PATH, map_location=config.DEVICE))
        model.gnn.eval()

        with open(config.FEATURE_NAMES_PATH) as f:
            meta = json.load(f)
        model.feature_names = meta["feature_names"]
        model.raw_score_range = tuple(meta["raw_score_range"])
        model.clip_bounds = {col: tuple(bounds) for col, bounds in meta["clip_bounds"].items()}

        ref = np.load(config.REFERENCE_GRAPH_PATH, allow_pickle=True)
        model.reference_X = ref["reference_X"]
        model.reference_gstin = ref["reference_gstin"]

        print(f"[Phase 3] Model artifacts loaded from {models_dir}")
        return model
