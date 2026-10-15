"""
Central configuration for the GST Compliance Risk project.

Schema note: the real dataset supplied (X_Train_Data_Input.csv /
Y_Train_Data_Target.csv / X_Test_Data_Input.csv / Y_Test_Data_Target.csv)
is anonymized — 22 numeric columns named Column0..Column21, ID as the
taxpayer identifier, target as the binary risk label. There are no
human-readable categorical columns (no state/business_type/sector text),
so this pipeline treats every feature as numeric and auto-classifies each
one as "continuous" (jitter+bootstrap for synthetic generation) or
"discrete/binary" (pure bootstrap) by cardinality, rather than assuming
column semantics we don't actually have.

Notably: Y_Test_Data_Target.csv contains real labels (this is a labeled
holdout, not a blind Kaggle test set) — TEST_LABELS_PATH exposes this so
Phase 3 can report genuine held-out test metrics, while the modeling
pipeline still treats test nodes as unlabeled during training/graph
construction (labels are only used for evaluation, never for fitting).
"""

import os
import torch

# ---- Paths ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")

TRAIN_FEATURES_PATH = os.path.join(DATA_RAW_DIR, "X_Train_Data_Input.csv")
TRAIN_LABELS_PATH = os.path.join(DATA_RAW_DIR, "Y_Train_Data_Target.csv")
TEST_FEATURES_PATH = os.path.join(DATA_RAW_DIR, "X_Test_Data_Input.csv")
TEST_LABELS_PATH = os.path.join(DATA_RAW_DIR, "Y_Test_Data_Target.csv")  # real labels — eval only
SYNTHETIC_DATA_PATH = os.path.join(DATA_SYNTHETIC_DIR, "gst_synthetic.csv")

# Saved artifact filenames (used by both run_pipeline.py and the Streamlit app)
ENCODER_PATH = os.path.join(MODELS_DIR, "encoder.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
GNN_WEIGHTS_PATH = os.path.join(MODELS_DIR, "gnn_model.pt")
GNN_CONFIG_PATH = os.path.join(MODELS_DIR, "gnn_config.json")
ISO_FOREST_PATH = os.path.join(MODELS_DIR, "iso_forest.joblib")
REFERENCE_GRAPH_PATH = os.path.join(MODELS_DIR, "reference_graph.npz")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")
SYNTHETIC_GENERATOR_STATS_PATH = os.path.join(MODELS_DIR, "synthetic_generator_stats.joblib")

for d in [DATA_RAW_DIR, DATA_SYNTHETIC_DIR, DATA_PROCESSED_DIR, MODELS_DIR,
          PLOTS_DIR, PREDICTIONS_DIR]:
    os.makedirs(d, exist_ok=True)

# ---- Reproducibility / device ----
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Schema (real GST-Predictive-Dataset-2024 columns) ----
ID_COL = "ID"
LABEL_COL = "target"
NUMERIC_COLS = [f"Column{i}" for i in range(22)]   # Column0 .. Column21
CATEGORICAL_COLS = []   # no text-categorical columns in the real schema

# Cardinality threshold used to auto-classify each numeric column as
# "continuous" (nunique > this, in a sample of the training data) vs.
# "discrete/binary" (nunique <= this) — see phase1_data/schema_profile.py.
# Discrete columns are bootstrap-resampled without jitter when generating
# synthetic data, so binary flags stay exactly {0, 1} rather than drifting.
DISCRETE_CARDINALITY_THRESHOLD = 50

# ---- Phase 1: synthetic data ----
N_SYNTHETIC_SAMPLES = 40000
SYNTHETIC_ID_PREFIX = "SYN"
JITTER_STD_FRACTION = 0.05   # jitter added to continuous cols, as a fraction of that col's std
SYNTHETIC_TOP_K_CORRELATED = 8   # how many real target-correlated columns drive synthetic labels

# ---- Phase 2: graph + GNN ----
KNN_K = 10
GNN_HIDDEN_DIM = 64
GNN_EMBED_DIM = 32
GNN_DROPOUT = 0.3
GNN_EPOCHS = 150
GNN_LR = 0.01
GNN_WEIGHT_DECAY = 5e-4
EARLY_STOP_PATIENCE = 15

# ---- Phase 2: Isolation Forest ----
ISO_FOREST_N_ESTIMATORS = 200
ISO_FOREST_CONTAMINATION = 0.1

# ---- Phase 3: risk fusion ----
RISK_WEIGHT_GNN = 0.6
RISK_WEIGHT_IFOREST = 0.4
RISK_LABEL_THRESHOLD = 0.5

# ---- Sampling (the real dataset has 785K/262K rows — far more than a
# k-NN graph needs for a final-year project to train and demo quickly).
# These caps subsample the REAL data before graph construction; set to
# None to use the full dataset (slow: k-NN graph build is the bottleneck).
MAX_TRAIN_ROWS = 20000
MAX_TEST_ROWS = 5000

# Size of the reference node pool saved for inductive inference in the
# Streamlit app (keeps models/reference_graph.npz small and fast to load).
REFERENCE_POOL_SIZE = 4000
