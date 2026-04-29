"""
config.py – Central configuration for the Customer Churn project.
All paths, column names, and hyper-parameter defaults live here so
every other module imports from a single source of truth.
"""

from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[3]   # customer-churn-project/
# When running from within the project dir, parents[3] may overshoot; fix it:
if not (ROOT_DIR / "data").exists():
    ROOT_DIR = Path(__file__).resolve().parents[2]

# ── Data paths ─────────────────────────────────────────────────────────────────
RAW_DATA_PATH      = ROOT_DIR / "data" / "raw"  / "Customer_Churn_Data_Large.xlsx"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "churn_processed.csv"

# ── Model artefact paths ───────────────────────────────────────────────────────
MODEL_DIR     = ROOT_DIR / "models"
MODEL_PATH    = MODEL_DIR / "churn_model.pkl"
SCALER_PATH   = MODEL_DIR / "scaler.pkl"
ENCODER_PATH  = MODEL_DIR / "encoder.pkl"

# ── Excel sheet names ─────────────────────────────────────────────────────────
SHEETS = {
    "demographics" : "Customer_Demographics",
    "transactions" : "Transaction_History",
    "service"      : "Customer_Service",
    "online"       : "Online_Activity",
    "churn"        : "Churn_Status",
}

# ── Column definitions ────────────────────────────────────────────────────────
TARGET_COL = "ChurnStatus"
ID_COL     = "CustomerID"

CATEGORICAL_COLS = [
    "Gender",
    "MaritalStatus",
    "IncomeLevel",
    "TopProductCategory",
    "ServiceUsage",
]

NUMERICAL_COLS = [
    "Age",
    "TotalSpent",
    "AvgTransactionValue",
    "NumTransactions",
    "NumServiceInteractions",
    "UnresolvedInteractions",
    "LoginFrequency",
    "DaysSinceLastLogin",
]

# ── Model hyper-parameters (defaults; tuning overrides these) ─────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20

RF_PARAMS = {
    "n_estimators"      : 300,
    "max_depth"         : 10,
    "min_samples_split" : 5,
    "min_samples_leaf"  : 2,
    "class_weight"      : "balanced",
    "random_state"      : RANDOM_STATE,
}

XGBOOST_PARAMS = {
    "n_estimators"   : 300,
    "max_depth"      : 6,
    "learning_rate"  : 0.05,
    "subsample"      : 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 4,          # ≈ (1-churn_rate)/churn_rate
    "use_label_encoder": False,
    "eval_metric"    : "logloss",
    "random_state"   : RANDOM_STATE,
}

# ── API settings ──────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
