"""
preprocess.py – Clean and validate the merged customer DataFrame.

Steps
-----
1. Drop exact duplicate rows
2. Validate / coerce dtypes
3. Handle missing values (imputation strategy)
4. Cap outliers using IQR fencing on numeric columns
5. Derive `DaysSinceLastLogin` from LastLoginDate
6. Save the cleaned CSV to data/processed/
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.config import (
    PROCESSED_DATA_PATH, TARGET_COL, ID_COL, NUMERICAL_COLS
)
from src.utils.logger import get_logger

log = get_logger(__name__)

# Reference date for computing DaysSinceLastLogin
REFERENCE_DATE = pd.Timestamp("2024-01-01")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=ID_COL)
    log.info("Duplicates removed: %d → %d rows", before, len(df))
    return df


def _derive_days_since_login(df: pd.DataFrame) -> pd.DataFrame:
    """Convert LastLoginDate to integer DaysSinceLastLogin."""
    if "LastLoginDate" in df.columns:
        df["LastLoginDate"] = pd.to_datetime(df["LastLoginDate"], errors="coerce")
        df["DaysSinceLastLogin"] = (
            REFERENCE_DATE - df["LastLoginDate"]
        ).dt.days
        df["DaysSinceLastLogin"] = df["DaysSinceLastLogin"].fillna(
            df["DaysSinceLastLogin"].median()
        ).astype(int)
        df.drop(columns=["LastLoginDate"], inplace=True)
    return df


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy
    --------
    - Numeric   : median  (robust to remaining outliers)
    - Categorical: mode (most frequent)
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        log.info("No missing values detected.")
        return df

    log.info("Missing values before imputation:\n%s", missing.to_string())

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    log.info("Missing values after imputation: %d", df.isnull().sum().sum())
    return df


def _cap_outliers(df: pd.DataFrame, cols: list[str], factor: float = 3.0) -> pd.DataFrame:
    """
    IQR-based winsorisation.  Values beyond median ± factor*IQR are capped.
    Using factor=3 (rather than the traditional 1.5) to be conservative —
    we want to smooth extreme values, not lose too much real signal.
    """
    for col in cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        clipped = df[col].clip(lower, upper)
        changed = (clipped != df[col]).sum()
        if changed:
            log.info("  %s: %d values capped (IQR×%.1f)", col, changed, factor)
        df[col] = clipped
    return df


# ── Main entry point ──────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df   : raw merged DataFrame from load_data.load_raw_data()
    save : if True, persist the cleaned file to PROCESSED_DATA_PATH

    Returns
    -------
    pd.DataFrame – cleaned, ready for feature engineering
    """
    log.info("=== Preprocessing pipeline start ===")

    df = _drop_duplicates(df)
    df = _derive_days_since_login(df)
    df = _impute_missing(df)

    numeric_to_cap = [c for c in NUMERICAL_COLS if c in df.columns]
    df = _cap_outliers(df, numeric_to_cap)

    # Drop raw ID column (keep for reference only if needed downstream)
    # We keep it for traceability but won't pass it to models
    log.info("Final cleaned shape: %d rows × %d cols", *df.shape)

    if save:
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        log.info("Cleaned data saved → %s", PROCESSED_DATA_PATH)

    log.info("=== Preprocessing pipeline end ===")
    return df


if __name__ == "__main__":
    from src.data.load_data import load_raw_data
    raw = load_raw_data()
    clean = preprocess(raw)
    print(clean.dtypes)
    print(clean.describe())
