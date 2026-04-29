"""
main.py – FastAPI inference service for the Customer Churn model.

Start the server
----------------
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Interactive docs
----------------
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import pandas as pd
from typing import Optional

from src.models.predict import predict_single, predict_batch
from src.utils.logger import get_logger

log = get_logger(__name__)

app = FastAPI(
    title       = "SmartBank Customer Churn Prediction API",
    description = "Predict churn probability for individual customers or batches.",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class CustomerRecord(BaseModel):
    Age                    : int   = Field(..., ge=18, le=100, example=35)
    Gender                 : str   = Field(..., example="F")
    MaritalStatus          : str   = Field(..., example="Single")
    IncomeLevel            : str   = Field(..., example="Medium")
    TotalSpent             : float = Field(..., ge=0, example=1200.0)
    AvgTransactionValue    : float = Field(..., ge=0, example=150.0)
    NumTransactions        : int   = Field(..., ge=0, example=8)
    TopProductCategory     : str   = Field(..., example="Electronics")
    NumServiceInteractions : int   = Field(..., ge=0, example=2)
    UnresolvedInteractions : int   = Field(..., ge=0, example=1)
    LoginFrequency         : int   = Field(..., ge=0, example=15)
    DaysSinceLastLogin     : int   = Field(..., ge=0, example=30)
    ServiceUsage           : str   = Field(..., example="Mobile App")

    @field_validator("Gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        if v not in {"M", "F"}:
            raise ValueError("Gender must be 'M' or 'F'")
        return v

    @field_validator("IncomeLevel")
    @classmethod
    def validate_income(cls, v: str) -> str:
        if v not in {"Low", "Medium", "High"}:
            raise ValueError("IncomeLevel must be Low / Medium / High")
        return v


class ChurnResponse(BaseModel):
    churn_probability : float
    churn_prediction  : int
    risk_level        : str


class BatchRequest(BaseModel):
    customers: list[CustomerRecord]


class BatchResponse(BaseModel):
    predictions: list[dict]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Churn Prediction API v1.0"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=ChurnResponse, tags=["Prediction"])
def predict_endpoint(customer: CustomerRecord):
    """
    Predict churn probability for a **single** customer.

    Returns
    -------
    * **churn_probability** – float between 0 and 1
    * **churn_prediction**  – 1 (churn) or 0 (retain)
    * **risk_level**        – Low / Medium / High
    """
    try:
        result = predict_single(customer.model_dump())
        log.info("Single prediction → %s", result)
        return result
    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch_endpoint(batch: BatchRequest):
    """
    Predict churn for a **batch** of customers (up to 1 000 per request).
    """
    if len(batch.customers) > 1000:
        raise HTTPException(status_code=400,
                            detail="Batch size must not exceed 1 000 records.")
    try:
        df = pd.DataFrame([c.model_dump() for c in batch.customers])
        result_df = predict_batch(df)
        predictions = result_df[
            ["ChurnProbability", "ChurnPrediction", "RiskLevel"]
        ].to_dict(orient="records")
        return {"predictions": predictions}
    except Exception as exc:
        log.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/model/info", tags=["Model"])
def model_info():
    """Return metadata about the deployed model."""
    from src.utils.config import MODEL_PATH, SCALER_PATH, ENCODER_PATH
    return {
        "model_path"  : str(MODEL_PATH),
        "scaler_path" : str(SCALER_PATH),
        "encoder_path": str(ENCODER_PATH),
        "model_exists": MODEL_PATH.exists(),
    }
