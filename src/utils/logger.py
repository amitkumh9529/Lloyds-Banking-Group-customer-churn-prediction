"""
logger.py – Standardised logging setup for the project.
Import `get_logger(__name__)` in every module.
"""

import logging
import sys
from pathlib import Path

LOG_DIR  = Path(__file__).resolve().parents[3] / "logs"
LOG_DIR.mkdir(exist_ok=True)

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger that writes to both stdout and a rotating log file.

    Parameters
    ----------
    name  : __name__ of the calling module
    level : logging level (default INFO)
    """
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    # ── Console handler ───────────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # ── File handler ──────────────────────────────────────────────────────────
    fh = logging.FileHandler(LOG_DIR / "churn_project.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
