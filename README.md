# 🏦 SmartBank Customer Churn Prediction

> **Lloyds Banking Group – Data Science & Analytics Graduate Project**  
> End-to-end machine learning system for predicting and preventing customer churn.

---

## Project Overview

SmartBank (a Lloyds subsidiary) has observed increased churn among young professionals and small business owners. This project builds a **predictive churn model** — from raw data to a live API and interactive dashboard — enabling the retention team to intervene *before* customers leave.

**Churn rate in dataset: ~20.4 %** (moderately imbalanced)

---

## Folder Structure

```
customer-churn-project/
│
├── data/
│   ├── raw/                       # Original Excel workbook (5 sheets)
│   └── processed/                 # Cleaned, feature-engineered CSV
│
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb    # Model training & evaluation
│
├── src/
│   ├── data/
│   │   ├── load_data.py           # Load & merge all five sheets
│   │   └── preprocess.py          # Clean, impute, cap outliers
│   ├── features/
│   │   └── build_features.py      # Feature engineering, encoding, scaling
│   ├── models/
│   │   ├── train_model.py         # Train RF + XGBoost, save best
│   │   └── predict.py             # Inference helpers (single + batch)
│   └── utils/
│       ├── config.py              # Central configuration
│       └── logger.py              # Structured logging
│
├── models/                        # Persisted artefacts (auto-generated)
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   └── encoder.pkl
│
├── api/
│   └── main.py                    # FastAPI inference service
│
├── app/
│   └── streamlit_app.py           # Interactive dashboard
│
├── requirements.txt
└── README.md
```

---

## Data Sources (5 sheets)

| Sheet | Rows | Key Columns |
|---|---|---|
| Customer_Demographics | 1 000 | Age, Gender, MaritalStatus, IncomeLevel |
| Transaction_History | 5 054 | TransactionDate, AmountSpent, ProductCategory |
| Customer_Service | 1 002 | InteractionType, ResolutionStatus |
| Online_Activity | 1 000 | LastLoginDate, LoginFrequency, ServiceUsage |
| Churn_Status | 1 000 | ChurnStatus (0/1) |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full training pipeline

```bash
# From project root
python -m src.data.load_data          # verify data loads
python -m src.data.preprocess         # clean + save CSV
python -m src.features.build_features # engineer + encode + save artefacts
python -m src.models.train_model      # train + evaluate + save model
```

Or run all at once via the notebooks:
```bash
jupyter notebook notebooks/
```

### 3. Start the FastAPI service

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI → [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Launch the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Service status |
| POST | `/predict` | Single customer prediction |
| POST | `/predict/batch` | Batch prediction (up to 1 000) |
| GET | `/model/info` | Model artefact metadata |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35,
    "Gender": "F",
    "MaritalStatus": "Single",
    "IncomeLevel": "Medium",
    "TotalSpent": 1200.0,
    "AvgTransactionValue": 150.0,
    "NumTransactions": 8,
    "TopProductCategory": "Electronics",
    "NumServiceInteractions": 2,
    "UnresolvedInteractions": 1,
    "LoginFrequency": 15,
    "DaysSinceLastLogin": 30,
    "ServiceUsage": "Mobile App"
  }'
```

**Response:**
```json
{
  "churn_probability": 0.2341,
  "churn_prediction": 0,
  "risk_level": "Low"
}
```

---

## Model Summary

| Algorithm | Notes |
|---|---|
| **Random Forest** | Primary model — `class_weight='balanced'` handles imbalance; native feature importances |
| **XGBoost** | Comparison model — `scale_pos_weight=4` compensates for class imbalance |

**Evaluation metrics used:** ROC-AUC, PR-AUC, Precision, Recall, F1-Score (Churned class), Confusion Matrix.

The model with the higher test-set **ROC-AUC** is automatically saved as `churn_model.pkl`.

---

## Engineered Features

| Feature | Description |
|---|---|
| `SpendPerLogin` | Total spend / login frequency |
| `ServicePressureRatio` | Unresolved / total service interactions |
| `HighUnresolved` | Flag: >2 unresolved interactions |
| `AgeGroup` | Young / Middle / Senior |
| `DaysSinceLastLogin` | Days elapsed since last platform login |

---

## Business Recommendations

1. **Tiered intervention** — High risk → relationship manager call; Medium → targeted email; Low → upsell.
2. **Monthly re-scoring** — Score entire customer base every month.
3. **Quarterly retraining** — Prevents model drift as behaviour changes.
4. **SHAP explainability** — Add SHAP values in v2 for per-customer explanations.
5. **A/B testing** — Measure retention uplift vs control group to quantify ROI.

---

## Author

Graduate Data Scientist — Lloyds Banking Group, Data Science & Analytics Team
