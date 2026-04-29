"""
load_data.py – Load and merge all five sheets from the raw Excel workbook.

Usage
-----
    from src.data.load_data import load_raw_data
    df = load_raw_data()          # returns merged DataFrame (1 000 rows)
"""

import pandas as pd
from pathlib import Path
from src.utils.config import RAW_DATA_PATH, SHEETS, ID_COL
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Individual sheet loaders ──────────────────────────────────────────────────

def _load_sheet(path: Path, sheet_key: str) -> pd.DataFrame:
    sheet_name = SHEETS[sheet_key]
    log.info("Loading sheet '%s' …", sheet_name)
    df = pd.read_excel(path, sheet_name=sheet_name)
    log.info("  → %d rows, %d cols", *df.shape)
    return df


def load_demographics(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return _load_sheet(path, "demographics")


def load_transactions(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return _load_sheet(path, "transactions")


def load_customer_service(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return _load_sheet(path, "service")


def load_online_activity(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return _load_sheet(path, "online")


def load_churn_status(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return _load_sheet(path, "churn")


# ── Aggregation helpers ───────────────────────────────────────────────────────

def _aggregate_transactions(txn: pd.DataFrame) -> pd.DataFrame:
    """Collapse multi-row transaction history to one row per customer."""
    agg = (
        txn.groupby(ID_COL)
        .agg(
            TotalSpent          = ("AmountSpent", "sum"),
            AvgTransactionValue = ("AmountSpent", "mean"),
            NumTransactions     = ("TransactionID", "count"),
            TopProductCategory  = ("ProductCategory", lambda s: s.value_counts().idxmax()),
        )
        .reset_index()
    )
    return agg


def _aggregate_service(svc: pd.DataFrame) -> pd.DataFrame:
    """Collapse customer-service interactions to one row per customer."""
    agg = (
        svc.groupby(ID_COL)
        .agg(
            NumServiceInteractions = ("InteractionID", "count"),
            UnresolvedInteractions = (
                "ResolutionStatus",
                lambda s: (s == "Unresolved").sum()
            ),
        )
        .reset_index()
    )
    return agg


# ── Main entry point ──────────────────────────────────────────────────────────

def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load, aggregate, and merge all five sheets.

    Returns
    -------
    pd.DataFrame
        One row per customer (1 000 rows).  Target column is 'ChurnStatus'.
    """
    demo   = load_demographics(path)
    txn    = load_transactions(path)
    svc    = load_customer_service(path)
    online = load_online_activity(path)
    churn  = load_churn_status(path)

    txn_agg = _aggregate_transactions(txn)
    svc_agg = _aggregate_service(svc)

    log.info("Merging all sources on CustomerID …")
    df = (
        demo
        .merge(txn_agg,  on=ID_COL, how="left")
        .merge(svc_agg,  on=ID_COL, how="left")
        .merge(online,   on=ID_COL, how="left")
        .merge(churn,    on=ID_COL, how="left")
    )

    # Customers who never appear in the transaction / service tables → fill 0
    df["NumTransactions"]        = df["NumTransactions"].fillna(0).astype(int)
    df["TotalSpent"]             = df["TotalSpent"].fillna(0.0)
    df["AvgTransactionValue"]    = df["AvgTransactionValue"].fillna(0.0)
    df["NumServiceInteractions"] = df["NumServiceInteractions"].fillna(0).astype(int)
    df["UnresolvedInteractions"] = df["UnresolvedInteractions"].fillna(0).astype(int)

    log.info("Merge complete → %d rows, %d cols", *df.shape)
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(df.info())
    print(df.head())
