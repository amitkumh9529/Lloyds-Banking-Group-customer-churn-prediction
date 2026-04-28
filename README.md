# 🏦 SmartBank · Customer Churn Intelligence Platform
### Lloyds Banking Group — Customer Retention Enhancement through Predictive Analytics

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the data file in the same directory as app.py
#    (or update the DATA_PATH in the load_data() function)
cp preprocessed_churn_data.csv ./

# 3. Run
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## App Pages

| Page | Description |
|------|-------------|
| 🏠 Dashboard | KPI cards, churn overview, segment charts, model summary |
| 📊 EDA & Customer Insights | Age/gender/income/marital distributions, heat maps, data explorer |
| 🤖 Model Training & Evaluation | ROC / PR curves, confusion matrix, feature importance, model comparison |
| 🔮 Churn Predictor | Single customer gauge + recommendations · Batch CSV upload |
| 📋 Business Recommendations | Risk tiers, action plans, KPIs, FCA compliance notes |

---

## Models Trained

- Logistic Regression (scaled features)
- Decision Tree (max_depth=5)
- **Random Forest** ← best by ROC-AUC (ensemble, class_weight="balanced")
- Gradient Boosting

All models use **Stratified 5-Fold CV** and are evaluated on a held-out 20% test set.

---

## Data Notes

The `preprocessed_churn_data.csv` contains encoded features from Phase 1 EDA.
A **synthetic churn label** is generated using domain-informed probabilities
(age, income, marital status, gender) — consistent with the Task 2 notebook.
In production, replace with actual churn labels from Lloyds CRM data.

---

## Stack

`Streamlit` · `scikit-learn` · `Plotly` · `pandas` · `numpy`
