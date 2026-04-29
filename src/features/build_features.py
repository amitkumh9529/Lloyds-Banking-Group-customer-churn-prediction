"""
build_features.py – Feature engineering and encoding pipeline.

Steps
-----
1. Engineer interaction / ratio features
2. One-hot encode categorical columns
3. Standard-scale numerical columns
4. Split into train / test sets
5. Persist scaler and encoder artefacts
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from src.utils.config import (
    CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL, ID_COL,
    SCALER_PATH, ENCODER_PATH, TEST_SIZE, RANDOM_STATE,
    MODEL_DIR,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features that improve predictive power.

    New columns
    -----------
    SpendPerLogin         : avg spend per login session
    ServicePressureRatio  : unresolved / total service interactions
    HighUnresolved        : binary flag – >2 unresolved interactions
    AgeGroup              : binned age (Young / Middle / Senior)
    """
    df = df.copy()

    # Spend efficiency
    df["SpendPerLogin"] = np.where(
        df["LoginFrequency"] > 0,
        df["TotalSpent"] / df["LoginFrequency"],
        0.0,
    )

    # Service frustration proxy
    total_svc = df["NumServiceInteractions"].replace(0, np.nan)
    df["ServicePressureRatio"] = (df["UnresolvedInteractions"] / total_svc).fillna(0.0)

    # High unresolved flag
    df["HighUnresolved"] = (df["UnresolvedInteractions"] > 2).astype(int)

    # Age groups (ordinal buckets)
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 30, 50, 100],
        labels=["Young", "Middle", "Senior"],
    ).astype(str)

    log.info("Engineered 4 new features: SpendPerLogin, ServicePressureRatio, "
             "HighUnresolved, AgeGroup")
    return df


# ── Encoding & scaling ────────────────────────────────────────────────────────

def encode_and_scale(
    df: pd.DataFrame,
    fit: bool = True,
    scaler: StandardScaler | None = None,
    encoder: OneHotEncoder | None = None,
) -> tuple[pd.DataFrame, StandardScaler, OneHotEncoder]:
    """
    Encode categoricals + scale numerics.

    Parameters
    ----------
    df      : pre-processed + feature-engineered DataFrame
    fit     : True → fit new transformers; False → use provided ones
    scaler  : pre-fitted StandardScaler (used when fit=False)
    encoder : pre-fitted OneHotEncoder  (used when fit=False)

    Returns
    -------
    (transformed_df, scaler, encoder)
    """
    # Columns that actually exist in df
    cat_cols = [c for c in CATEGORICAL_COLS + ["AgeGroup"] if c in df.columns]
    num_cols = [
        c for c in NUMERICAL_COLS + ["SpendPerLogin", "ServicePressureRatio"]
        if c in df.columns
    ]
    binary_cols = [c for c in ["HighUnresolved"] if c in df.columns]
    preserve_cols = [ID_COL, TARGET_COL] if TARGET_COL in df.columns else [ID_COL]

    # ── Numerics ──────────────────────────────────────────────────────────────
    if fit:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        log.info("StandardScaler fitted on %d numeric columns", len(num_cols))
    else:
        df[num_cols] = scaler.transform(df[num_cols])

    # ── Categoricals ──────────────────────────────────────────────────────────
    if fit:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        enc_arr = encoder.fit_transform(df[cat_cols])
        log.info("OneHotEncoder fitted on %d categorical columns → %d OHE features",
                 len(cat_cols), enc_arr.shape[1])
    else:
        enc_arr = encoder.transform(df[cat_cols])

    enc_cols = encoder.get_feature_names_out(cat_cols)
    enc_df   = pd.DataFrame(enc_arr, columns=enc_cols, index=df.index)

    # ── Assemble final frame ──────────────────────────────────────────────────
    result = pd.concat(
        [df[preserve_cols].reset_index(drop=True),
         df[num_cols].reset_index(drop=True),
         df[binary_cols].reset_index(drop=True),
         enc_df.reset_index(drop=True)],
        axis=1,
    )

    log.info("Final feature matrix: %d rows × %d cols", *result.shape)
    return result, scaler, encoder


# ── Persist artefacts ─────────────────────────────────────────────────────────

def save_artefacts(scaler: StandardScaler, encoder: OneHotEncoder) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)
    log.info("Scaler → %s", SCALER_PATH)
    log.info("Encoder → %s", ENCODER_PATH)


def load_artefacts() -> tuple[StandardScaler, OneHotEncoder]:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    return scaler, encoder


# ── Train / test split ────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return X_train, X_test, y_train, y_test (stratified split)."""
    feature_cols = [c for c in df.columns if c not in [ID_COL, TARGET_COL]]
    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    log.info("Train: %d | Test: %d | Churn rate train: %.1f%% | test: %.1f%%",
             len(y_train), len(y_test),
             100 * y_train.mean(), 100 * y_test.mean())
    return X_train, X_test, y_train, y_test


# ── Pipeline convenience function ─────────────────────────────────────────────

def build_features(df: pd.DataFrame):
    """
    Full feature-building pipeline for training.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, encoder
    """
    df_feat = engineer_features(df)
    df_enc, scaler, encoder = encode_and_scale(df_feat, fit=True)
    save_artefacts(scaler, encoder)
    X_train, X_test, y_train, y_test = split_data(df_enc)
    return X_train, X_test, y_train, y_test, scaler, encoder


if __name__ == "__main__":
    from src.data.load_data import load_raw_data
    from src.data.preprocess import preprocess

    raw   = load_raw_data()
    clean = preprocess(raw, save=False)
    X_train, X_test, y_train, y_test, *_ = build_features(clean)
    print("X_train shape:", X_train.shape)
    print("Positive rate in y_train:", y_train.mean())
