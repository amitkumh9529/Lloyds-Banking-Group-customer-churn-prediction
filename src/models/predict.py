"""
predict.py – Inference helpers.

Loads the persisted model + artefacts and exposes:
  * predict_single(record: dict) → dict
  * predict_batch(df: pd.DataFrame) → pd.DataFrame
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.config import MODEL_PATH, SCALER_PATH, ENCODER_PATH, ID_COL, TARGET_COL
from src.features.build_features import engineer_features, encode_and_scale
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Load artefacts (lazy, once) ───────────────────────────────────────────────

_model   = None
_scaler  = None
_encoder = None


def _load_artefacts():
    global _model, _scaler, _encoder
    if _model is None:
        log.info("Loading model artefacts …")
        with open(MODEL_PATH,   "rb") as f: _model   = pickle.load(f)
        with open(SCALER_PATH,  "rb") as f: _scaler  = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f: _encoder = pickle.load(f)
        log.info("Artefacts loaded.")


# ── Internal transform helper ─────────────────────────────────────────────────

def _transform(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Run feature engineering + encode/scale without re-fitting."""
    _load_artefacts()

    # Remove target if it snuck in
    df = df_raw.drop(columns=[TARGET_COL], errors="ignore").copy()

    # Engineer features
    df_feat = engineer_features(df)

    # Encode + scale using pre-fitted transformers
    df_enc, _, _ = encode_and_scale(df_feat, fit=False,
                                     scaler=_scaler, encoder=_encoder)

    # Drop ID col before prediction
    feature_cols = [c for c in df_enc.columns if c != ID_COL]
    return df_enc[feature_cols]


# ── Public API ────────────────────────────────────────────────────────────────

def predict_single(record: dict) -> dict:
    """
    Predict churn for a single customer.

    Parameters
    ----------
    record : dict with the same keys as the raw merged DataFrame
             (one row per customer, no CustomerID required)

    Returns
    -------
    dict with keys:
        churn_probability : float  (0 – 1)
        churn_prediction  : int    (0 or 1)
        risk_level        : str    (Low / Medium / High)
    """
    df = pd.DataFrame([record])
    if ID_COL not in df.columns:
        df[ID_COL] = 0

    X = _transform(df)
    proba = _model.predict_proba(X)[0, 1]
    pred  = int(proba >= 0.5)

    risk = "Low" if proba < 0.35 else ("Medium" if proba < 0.65 else "High")

    return {
        "churn_probability": round(float(proba), 4),
        "churn_prediction" : pred,
        "risk_level"       : risk,
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict churn for a DataFrame of customers.

    Returns the input DataFrame with two extra columns:
        ChurnProbability, RiskLevel
    """
    X = _transform(df)
    probas = _model.predict_proba(X)[:, 1]

    out = df.copy()
    out["ChurnProbability"] = probas.round(4)
    out["ChurnPrediction"]  = (probas >= 0.5).astype(int)
    out["RiskLevel"]        = pd.cut(
        probas,
        bins=[-np.inf, 0.35, 0.65, np.inf],
        labels=["Low", "Medium", "High"],
    )
    return out


if __name__ == "__main__":
    sample = {
        "Age"                   : 35,
        "Gender"                : "F",
        "MaritalStatus"         : "Single",
        "IncomeLevel"           : "Medium",
        "TotalSpent"            : 1200.0,
        "AvgTransactionValue"   : 150.0,
        "NumTransactions"       : 8,
        "TopProductCategory"    : "Electronics",
        "NumServiceInteractions": 2,
        "UnresolvedInteractions": 1,
        "LoginFrequency"        : 15,
        "DaysSinceLastLogin"    : 30,
        "ServiceUsage"          : "Mobile App",
    }
    result = predict_single(sample)
    print(result)
