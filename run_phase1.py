"""
run_phase1.py — Standalone entry point for Phase 1 only.

Run this to execute (and demo) Phase 1 in isolation:
    python run_phase1.py

What it does, in order:
    1. Loads the real GST-Predictive-Dataset-2024 CSVs from data/raw/
       (X_Train_Data_Input.csv, Y_Train_Data_Target.csv,
       X_Test_Data_Input.csv, Y_Test_Data_Target.csv), stratified-
       subsampled per config.MAX_TRAIN_ROWS / MAX_TEST_ROWS for a
       tractable graph in Phase 2.
    2. Validates schema, runs a full EDA report (prints + saves 4 plots
       to outputs/plots/).
    3. Fits the class-conditional synthetic generator on the (subsampled)
       real training data, generates synthetic rows, validates them
       numerically against the real data, and plots comparisons.
    4. Combines real + synthetic, applies the leak-safe preprocessing
       pipeline (median imputation, outlier clipping, scaling — all fit
       only on the train split).
    5. Saves the processed matrices to data/processed/ and a JSON run
       report to outputs/phase1_report.json.
"""

import os
import numpy as np

from src import config
from src.utils.io import RunReport
from src.phase1_data.data_loader import load_gst_data
from src.phase1_data.synthetic_generator import (
    fit_synthetic_generator, generate_synthetic_gst_data,
    validate_against_real, plot_real_vs_synthetic, plot_target_conditional_check,
)
from src.phase1_data.preprocessing import combine_and_preprocess_real_and_synthetic


def run_phase1():
    """Execute the full Phase 1 flow and return the preprocessed bundle dict."""
    report = RunReport()

    # --- Step 1 + 2: load, validate, subsample, EDA ---
    train_df, test_df = load_gst_data(run_eda=True, report=report)

    # --- Step 3: class-conditional synthetic data generation + validation ---
    generator_stats = fit_synthetic_generator(train_df)
    synth_df = generate_synthetic_gst_data(generator_stats)
    comparison = validate_against_real(synth_df, generator_stats)
    comparison.to_csv(os.path.join(config.OUTPUTS_DIR, "phase1_synthetic_validation.csv"), index=False)
    plot_real_vs_synthetic(train_df, synth_df)
    # The strongest real driver column gets an explicit conditional check —
    # see src/phase1_data/synthetic_generator.py docstring for why this
    # matters more than the marginal-distribution overlay for strong drivers.
    top_driver = generator_stats["correlations"].abs().idxmax()
    plot_target_conditional_check(train_df, synth_df, top_driver)

    report.log("n_synthetic_generated", len(synth_df))
    report.log("synthetic_risk_rate", float(synth_df[config.LABEL_COL].mean()))
    report.log("top_driver_column", top_driver)
    report.log("top_driver_real_corr", float(generator_stats["correlations"][top_driver]))

    # --- Step 4: combine + leak-safe preprocessing ---
    bundle = combine_and_preprocess_real_and_synthetic(train_df, test_df, synth_df, report=report)

    # --- Step 5: persist processed matrices for Phase 2 ---
    processed_path = os.path.join(config.DATA_PROCESSED_DIR, "phase1_bundle.npz")
    np.savez_compressed(
        processed_path,
        X_all=bundle["X_all"], y_all=bundle["y_all"], gstin_all=bundle["gstin_all"],
        is_synthetic_all=bundle["is_synthetic_all"], train_idx=bundle["train_idx"],
        val_idx=bundle["val_idx"], holdout_idx=bundle["holdout_idx"], test_idx=bundle["test_idx"],
        feature_names=np.array(bundle["feature_names"], dtype=object),
    )
    bundle["raw_df_all"].to_csv(os.path.join(config.DATA_PROCESSED_DIR, "raw_df_all.csv"), index=False)
    print(f"[Phase 1] Processed matrices saved to {processed_path}")

    report_path = os.path.join(config.OUTPUTS_DIR, "phase1_report.json")
    report.save(report_path)

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print(f"  Feature matrix   : {bundle['X_all'].shape}")
    print(f"  Feature columns  : {len(bundle['feature_names'])}")
    print(f"  Labeled nodes    : {len(bundle['train_idx']) + len(bundle['val_idx']) + len(bundle['holdout_idx'])}")
    print(f"  Unlabeled (test) : {len(bundle['test_idx'])}")
    print(f"  Top driver column: {top_driver} (real corr={generator_stats['correlations'][top_driver]:.3f})")
    print(f"  Plots            : {config.PLOTS_DIR}/phase1_*.png")
    print(f"  Report           : {report_path}")

    return bundle


if __name__ == "__main__":
    run_phase1()
