"""
Phase 1 — Synthetic GST data generation (class-conditional bootstrap).

Design history / why this approach: an earlier version generated features
unconditionally, then derived a synthetic label from a z-score-weighted
combination of the top-K real target-correlated columns. That approach
was tested and found to systematically weaken strong real relationships
— e.g. Column18 is a NEAR-PERFECT proxy for target in the real data
(96% of compliant rows have Column18=0; 92% of risky rows have
Column18=1), giving it a real correlation of 0.74, but the linear-
combination approach only reproduced ~0.55, because it treated Column18
as "one weighted input among several" instead of reproducing its
near-deterministic conditional relationship.

This version fixes that structurally: the synthetic target is drawn
FIRST (at the real, or a deliberately adjusted, risk rate), and every
column is then bootstrap-resampled from its REAL empirical distribution
CONDITIONAL on that drawn class. This is a non-parametric, class-
conditional generative model — simple, but it structurally preserves
every column's real relationship with the target, not just the top-K
ones, with no hand-tuned weights or label-flip-noise needed.
"""

import os
import numpy as np
import pandas as pd
from src import config
from src.phase1_data.schema_profile import profile_columns, compute_target_correlations


def fit_synthetic_generator(train_df, numeric_cols=config.NUMERIC_COLS,
                             label_col=config.LABEL_COL):
    """
    Fit the generator's statistics on real training data: an overall
    column profile (used only to decide continuous-vs-discrete and
    jitter/clip bounds), a per-class column profile (the actual source
    of synthetic samples), and the real column-target correlations
    (kept for reporting/validation, not used in generation).

    Inputs:
        train_df: pd.DataFrame - real training data (features + label).
        numeric_cols: list[str]
        label_col: str

    Returns:
        generator_stats: dict with
            "profile_overall": dict[col] -> stats (see schema_profile.profile_columns)
            "profile_by_class": {0: dict[col]->stats, 1: dict[col]->stats}
            "correlations": pd.Series - real column-target correlations
            "real_target_rate": float - P(target==1) in real training data
    """
    profile_overall = profile_columns(train_df, numeric_cols)
    profile_by_class = {
        cls: profile_columns(train_df[train_df[label_col] == cls], numeric_cols)
        for cls in (0, 1)
    }
    correlations = compute_target_correlations(train_df, numeric_cols, label_col)
    real_target_rate = float(train_df[label_col].mean())

    return {
        "profile_overall": profile_overall,
        "profile_by_class": profile_by_class,
        "correlations": correlations,
        "real_target_rate": real_target_rate,
    }


def generate_synthetic_gst_data(generator_stats, n_samples=config.N_SYNTHETIC_SAMPLES,
                                 random_state=config.RANDOM_STATE,
                                 save_path=config.SYNTHETIC_DATA_PATH,
                                 jitter_std_fraction=config.JITTER_STD_FRACTION,
                                 target_rate_multiplier=1.0,
                                 numeric_cols=config.NUMERIC_COLS):
    """
    Generate synthetic GSTIN rows via class-conditional bootstrap.

    Inputs:
        generator_stats: dict - output of fit_synthetic_generator()
        n_samples: int - number of synthetic rows to create
        random_state: int
        save_path: str or None - if given, writes the table to CSV
        jitter_std_fraction: float - jitter added to continuous columns
            (relative to that column's OVERALL std, so jitter magnitude
            stays consistent between the two classes), as a fraction
        target_rate_multiplier: float - synthetic risk rate relative to
            the real rate (>1 oversamples risky rows — useful for giving
            the GNN more risky examples without touching the real,
            untouched evaluation distribution)
        numeric_cols: list[str]

    Returns:
        synth_df: pd.DataFrame - ID, Column0..Column21, target,
                  is_synthetic=1, row order shuffled.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    rng = np.random.default_rng(random_state)
    profile_overall = generator_stats["profile_overall"]
    profile_by_class = generator_stats["profile_by_class"]

    target_rate = float(np.clip(generator_stats["real_target_rate"] * target_rate_multiplier, 0.01, 0.99))
    n_risky = int(round(n_samples * target_rate))
    n_compliant = n_samples - n_risky
    class_counts = {0: n_compliant, 1: n_risky}

    data = {col: np.empty(n_samples, dtype=float) for col in numeric_cols}
    cursor = {0: (0, n_compliant), 1: (n_compliant, n_samples)}  # (start, end) slice per class

    for col in numeric_cols:
        overall_kind = profile_overall[col]["kind"]
        overall_std = profile_overall[col]["std"]
        overall_min, overall_max = profile_overall[col]["min"], profile_overall[col]["max"]

        for cls in (0, 1):
            n_cls = class_counts[cls]
            cls_stats = profile_by_class[cls][col]
            values = cls_stats["values"]
            if len(values) == 0:
                # Extremely rare edge case: a column with zero non-null
                # observations in this class — fall back to the overall pool.
                values = profile_overall[col]["values"]

            replace = n_cls > len(values)
            sampled = rng.choice(values, size=n_cls, replace=replace)

            if overall_kind == "continuous" and overall_std > 0:
                jitter = rng.normal(0, overall_std * jitter_std_fraction, size=n_cls)
                sampled = np.clip(sampled + jitter, overall_min, overall_max)

            # Reinject THIS CLASS's real missingness rate for this column —
            # missingness patterns can themselves differ by class and are
            # part of what a class-conditional generator should reproduce.
            if cls_stats["missing_rate"] > 0:
                miss_mask = rng.random(n_cls) < cls_stats["missing_rate"]
                sampled[miss_mask] = np.nan

            start, end = cursor[cls]
            data[col][start:end] = sampled

    synth_target = np.concatenate([np.zeros(n_compliant, dtype=int), np.ones(n_risky, dtype=int)])

    synth_df = pd.DataFrame(data)
    synth_df[config.LABEL_COL] = synth_target

    # Shuffle row order (class-ordered construction above is otherwise
    # visible as a block structure, which is cosmetically odd and would
    # bias any code that accidentally relies on row order).
    shuffle_idx = rng.permutation(n_samples)
    synth_df = synth_df.iloc[shuffle_idx].reset_index(drop=True)

    synth_df[config.ID_COL] = [f"{config.SYNTHETIC_ID_PREFIX}{i:08d}" for i in range(n_samples)]
    synth_df["is_synthetic"] = 1
    synth_df = synth_df[[config.ID_COL] + numeric_cols + [config.LABEL_COL, "is_synthetic"]]

    _validate_synthetic_output(synth_df, n_samples)

    print(f"[Phase 1] Generated {n_samples} synthetic GSTINs via class-conditional bootstrap "
          f"({synth_df[config.LABEL_COL].mean():.1%} flagged risky; "
          f"real rate was {generator_stats['real_target_rate']:.1%}).")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        synth_df.to_csv(save_path, index=False)
        print(f"[Phase 1] Synthetic data saved to {save_path}")

    return synth_df


def _validate_synthetic_output(synth_df, n_samples):
    """Sanity checks before trusting the synthetic table."""
    assert len(synth_df) == n_samples, "row count mismatch"
    assert synth_df[config.ID_COL].nunique() == n_samples, "duplicate synthetic IDs generated"
    assert synth_df[config.LABEL_COL].nunique() == 2, (
        "synthetic target collapsed to a single class — check target_rate_multiplier"
    )


def validate_against_real(synth_df, generator_stats, numeric_cols=config.NUMERIC_COLS,
                           label_col=config.LABEL_COL):
    """
    Compare synthetic vs. real column medians/stds (overall) and
    column-target correlations, as a numeric sanity check to run before
    trusting the synthetic data (complements the visual check in
    plot_real_vs_synthetic).

    Inputs:
        synth_df: pd.DataFrame - output of generate_synthetic_gst_data()
        generator_stats: dict - output of fit_synthetic_generator()
        numeric_cols, label_col: as elsewhere

    Returns:
        pd.DataFrame comparing real vs. synthetic median/std/correlation per column.
    """
    profile = generator_stats["profile_overall"]
    real_corr = generator_stats["correlations"]
    synth_corr = synth_df[numeric_cols + [label_col]].corr(numeric_only=True)[label_col].drop(label_col)

    rows = []
    for col in numeric_cols:
        real_vals = profile[col]["values"]
        synth_vals = synth_df[col].dropna().values
        rows.append({
            "column": col,
            "kind": profile[col]["kind"],
            "real_median": round(float(np.median(real_vals)), 4),
            "synth_median": round(float(np.median(synth_vals)), 4),
            "real_std": round(profile[col]["std"], 4),
            "synth_std": round(float(np.std(synth_vals)), 4),
            "real_corr_target": round(float(real_corr.get(col, np.nan)), 3),
            "synth_corr_target": round(float(synth_corr.get(col, np.nan)), 3),
        })
    comparison = pd.DataFrame(rows)
    print("Real vs. synthetic column comparison "
          "(median is more reliable than std for fat-tailed columns):")
    print(comparison.to_string(index=False))
    return comparison


def plot_real_vs_synthetic(real_df, synth_df, numeric_cols=config.NUMERIC_COLS,
                            plots_dir=config.PLOTS_DIR):
    """
    Overlay real vs. synthetic distributions for each numeric column
    (x-axis clipped to the real column's 1st-99th percentile, since
    several columns have rare extreme outliers that otherwise compress
    the entire plot into an unreadable spike).

    Inputs:
        real_df: pd.DataFrame - real training data
        synth_df: pd.DataFrame - synthetic data (same numeric columns)
        numeric_cols: list[str]
        plots_dir: str - output directory
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(plots_dir, exist_ok=True)
    n_cols = 4
    n_rows = -(-len(numeric_cols) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3.5 * n_rows))
    axes = axes.flatten()

    combined = pd.concat([
        real_df[numeric_cols].assign(source="real"),
        synth_df[numeric_cols].assign(source="synthetic"),
    ], ignore_index=True)

    for i, col in enumerate(numeric_cols):
        lo, hi = real_df[col].quantile([0.01, 0.99])
        if lo == hi:
            lo, hi = real_df[col].min(), real_df[col].max()
        try:
            sns.kdeplot(data=combined, x=col, hue="source", common_norm=False, ax=axes[i])
        except Exception:
            sns.histplot(data=combined, x=col, hue="source", stat="density",
                         common_norm=False, ax=axes[i], bins=20)
        axes[i].set_xlim(lo, hi)
        axes[i].set_title(col, fontsize=9)
    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    out_path = os.path.join(plots_dir, "phase1_real_vs_synthetic.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[Phase 1] Real-vs-synthetic comparison plot saved to {out_path}")


def plot_target_conditional_check(real_df, synth_df, col, plots_dir=config.PLOTS_DIR,
                                   label_col=config.LABEL_COL):
    """
    For a single column, plot its value distribution split by target class,
    real vs. synthetic side by side — the direct visual check for whether
    the class-conditional relationship (not just the marginal distribution)
    was reproduced. Most useful for strong drivers like Column18.

    Inputs:
        real_df, synth_df: pd.DataFrame
        col: str - column to inspect
        plots_dir: str
        label_col: str
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, df, title in zip(axes, [real_df, synth_df], ["Real", "Synthetic"]):
        sns.histplot(data=df, x=col, hue=label_col, stat="proportion",
                     common_norm=False, bins=15, ax=ax, multiple="dodge")
        ax.set_title(f"{title}: {col} by {label_col}")
    plt.tight_layout()
    out_path = os.path.join(plots_dir, f"phase1_conditional_check_{col}.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[Phase 1] Class-conditional check for {col} saved to {out_path}")
