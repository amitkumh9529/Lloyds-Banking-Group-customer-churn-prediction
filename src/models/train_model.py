"""
train_model.py – Train, tune, and persist the churn classification model.

Algorithm choice
----------------
Random Forest is used as the primary model because:
  * Handles class imbalance well via `class_weight='balanced'`
  * Naturally provides feature importances
  * Robust to outliers and doesn't require feature scaling
  * Interpretable at business level via feature importance charts

XGBoost is trained in parallel for comparison; the better model (by ROC-AUC)
is saved as `churn_model.pkl`.

Evaluation metrics reported
----------------------------
Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC, Confusion Matrix.
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

from src.utils.config import (
    RF_PARAMS, XGBOOST_PARAMS, MODEL_PATH, MODEL_DIR, RANDOM_STATE,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Cross-validation helper ───────────────────────────────────────────────────

def cross_validate_model(model, X_train, y_train, cv: int = 5) -> float:
    """Return mean ROC-AUC from stratified k-fold CV."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train,
                              cv=skf, scoring="roc_auc", n_jobs=-1)
    log.info("CV ROC-AUC: %.4f ± %.4f", scores.mean(), scores.std())
    return scores.mean()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, label: str = "") -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc  = average_precision_score(y_test, y_proba)

    log.info("=== %s Evaluation ===", label)
    log.info("ROC-AUC : %.4f", roc_auc)
    log.info("PR-AUC  : %.4f", pr_auc)
    log.info("\n%s", classification_report(y_test, y_pred,
                                           target_names=["Retained", "Churned"]))

    return {
        "model"      : label,
        "roc_auc"    : roc_auc,
        "pr_auc"     : pr_auc,
        "report"     : classification_report(y_test, y_pred, output_dict=True),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "y_proba"    : y_proba,
    }


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _save_confusion_matrix(cm, label: str, output_dir) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Retained", "Churned"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix – {label}")
    path = output_dir / f"confusion_matrix_{label.lower().replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Confusion matrix saved → %s", path)


def _save_feature_importance(model, feature_names: list, output_dir) -> None:
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:20]     # top 20

    fig, ax = plt.subplots(figsize=(10, 6))
    idx_rev = idx[::-1]
    ax.barh([feature_names[i] for i in idx_rev],
             importances[idx_rev], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title("Top-20 Feature Importances (Random Forest)")
    path = output_dir / "feature_importances.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Feature importances saved → %s", path)


# ── Main training routine ─────────────────────────────────────────────────────

def train(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple:
    """
    Train Random Forest (and XGBoost if available).
    Save the best model to MODEL_PATH.

    Returns
    -------
    (best_model, results_dict)
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # ── Random Forest ──────────────────────────────────────────────────────────
    log.info("Training Random Forest …")
    rf = RandomForestClassifier(**RF_PARAMS)
    cross_validate_model(rf, X_train, y_train)
    rf.fit(X_train, y_train)
    results["rf"] = evaluate(rf, X_test, y_test, label="Random Forest")
    _save_confusion_matrix(results["rf"]["conf_matrix"], "Random Forest", MODEL_DIR)
    _save_feature_importance(rf, list(X_train.columns), MODEL_DIR)

    best_model = rf
    best_score = results["rf"]["roc_auc"]

    # ── XGBoost ────────────────────────────────────────────────────────────────
    if _XGB_AVAILABLE:
        log.info("Training XGBoost …")
        xgb = XGBClassifier(**XGBOOST_PARAMS)
        cross_validate_model(xgb, X_train, y_train)
        xgb.fit(X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False)
        results["xgb"] = evaluate(xgb, X_test, y_test, label="XGBoost")
        _save_confusion_matrix(results["xgb"]["conf_matrix"], "XGBoost", MODEL_DIR)

        if results["xgb"]["roc_auc"] > best_score:
            best_model = xgb
            best_score = results["xgb"]["roc_auc"]
            log.info("XGBoost is the better model (ROC-AUC %.4f)", best_score)
    else:
        log.warning("xgboost not installed – skipping XGBoost training.")

    # ── Persist best model ────────────────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    log.info("Best model (%s) saved → %s", type(best_model).__name__, MODEL_PATH)

    return best_model, results


if __name__ == "__main__":
    from src.data.load_data import load_raw_data
    from src.data.preprocess import preprocess
    from src.features.build_features import build_features

    raw   = load_raw_data()
    clean = preprocess(raw)
    X_train, X_test, y_train, y_test, *_ = build_features(clean)
    model, results = train(X_train, X_test, y_train, y_test)
    print("Best model:", type(model).__name__)
