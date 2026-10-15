"""
run_pipeline.py — Full end-to-end pipeline: Phase 1 -> Phase 2 -> Phase 3.

Run this once to train and save everything the Streamlit app needs:
    python run_pipeline.py

This trains on the real GST-Predictive-Dataset-2024 (stratified-subsampled
per config.MAX_TRAIN_ROWS/MAX_TEST_ROWS — see src/config.py to change),
augmented with class-conditional synthetic data, builds the k-NN graph,
trains the GraphSAGE + Isolation Forest, evaluates on both an internal
holdout and (if present) the REAL test labels, then saves every fitted
artifact to models/ so app/streamlit_app.py can load and score new data
without retraining.
"""

from src.phase3_pipeline.model_pipeline import GSTComplianceRiskModel


def run_pipeline():
    model = GSTComplianceRiskModel()
    model.fit()
    model.save()

    # Quick smoke-test of both prediction paths before handing off to the app.
    sample_ids = list(model.gstin_all[model._bundle["test_idx"][:3]])
    print("\n[run_pipeline] Sample predictions on known test-graph nodes:")
    print(model.predict_for_gstin_list(sample_ids, explain=True).to_string(index=False))

    return model


if __name__ == "__main__":
    run_pipeline()
