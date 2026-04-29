"""
streamlit_app.py – Interactive dashboard for SmartBank Churn Prediction.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "SmartBank – Churn Predictor",
    page_icon   = "🏦",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/fluency/96/bank-building.png",
    width=80,
)
st.sidebar.title("SmartBank")
st.sidebar.markdown("**Customer Churn Intelligence Platform**")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔍 Single Prediction", "📊 Batch Prediction", "📈 Model Insights"],
)


# ── Load model artefacts (cached) ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_model():
    from src.models.predict import _load_artefacts, _model, _scaler, _encoder
    _load_artefacts()
    import src.models.predict as _p
    return _p._model, _p._scaler, _p._encoder


@st.cache_data(show_spinner="Loading processed data …")
def load_processed():
    from src.utils.config import PROCESSED_DATA_PATH
    if PROCESSED_DATA_PATH.exists():
        return pd.read_csv(PROCESSED_DATA_PATH)
    return None


# ── Helper ────────────────────────────────────────────────────────────────────

def risk_badge(risk: str) -> str:
    colours = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    return f"{colours.get(risk, '⚪')} **{risk}**"


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Home
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🏦 SmartBank – Customer Churn Prediction")
    st.markdown("""
    Welcome to the **Customer Retention Intelligence Platform**, built for the
    Data Science & Analytics team at Lloyds Banking Group.

    ### What this dashboard does
    | Feature | Description |
    |---|---|
    | 🔍 Single Prediction | Enter one customer's details and get an instant churn probability |
    | 📊 Batch Prediction | Upload a CSV of customers for bulk scoring |
    | 📈 Model Insights | Explore feature importances and model performance metrics |

    ### Dataset overview
    * **1 000 customers** across 5 data sources (demographics, transactions,
      service interactions, online activity, churn labels)
    * **~20.4 % churn rate** — moderately imbalanced; handled via `class_weight='balanced'`
    * **Random Forest + XGBoost** evaluated; best model saved automatically

    > Navigate using the sidebar on the left ←
    """)

    df = load_processed()
    if df is not None and "ChurnStatus" in df.columns:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", len(df))
        col2.metric("Churned",         int(df["ChurnStatus"].sum()))
        col3.metric("Retained",        int((df["ChurnStatus"] == 0).sum()))
        col4.metric("Churn Rate",      f"{df['ChurnStatus'].mean()*100:.1f} %")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Single Prediction
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Single Prediction":
    st.title("🔍 Single Customer Churn Prediction")
    st.markdown("Fill in the customer details below and click **Predict**.")

    with st.form("single_pred_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Demographics")
            age            = st.slider("Age", 18, 100, 40)
            gender         = st.selectbox("Gender", ["M", "F"])
            marital_status = st.selectbox("Marital Status",
                                           ["Single", "Married", "Divorced", "Widowed"])
            income_level   = st.selectbox("Income Level", ["Low", "Medium", "High"])

        with col2:
            st.subheader("Transaction History")
            total_spent          = st.number_input("Total Spent (£)", 0.0, 50000.0, 1200.0, step=50.0)
            avg_txn_value        = st.number_input("Avg Transaction (£)", 0.0, 5000.0, 150.0, step=10.0)
            num_transactions     = st.number_input("# Transactions", 0, 500, 8, step=1)
            top_product_category = st.selectbox("Top Product Category",
                                                 ["Electronics", "Books", "Clothing",
                                                  "Groceries", "Travel"])

        with col3:
            st.subheader("Service & Online Activity")
            num_svc_interactions = st.number_input("# Service Interactions", 0, 50, 2, step=1)
            unresolved           = st.number_input("Unresolved Interactions", 0, 50, 1, step=1)
            login_freq           = st.slider("Login Frequency (times/month)", 0, 50, 15)
            days_since_login     = st.slider("Days Since Last Login", 0, 365, 30)
            service_usage        = st.selectbox("Service Usage",
                                                 ["Mobile App", "Website", "Online Banking"])

        submitted = st.form_submit_button("🚀 Predict Churn", use_container_width=True)

    if submitted:
        try:
            from src.models.predict import predict_single
            record = dict(
                Age                    = age,
                Gender                 = gender,
                MaritalStatus          = marital_status,
                IncomeLevel            = income_level,
                TotalSpent             = total_spent,
                AvgTransactionValue    = avg_txn_value,
                NumTransactions        = num_transactions,
                TopProductCategory     = top_product_category,
                NumServiceInteractions = num_svc_interactions,
                UnresolvedInteractions = unresolved,
                LoginFrequency         = login_freq,
                DaysSinceLastLogin     = days_since_login,
                ServiceUsage           = service_usage,
            )
            result = predict_single(record)

            st.markdown("---")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Churn Probability", f"{result['churn_probability']*100:.1f} %")
            col_b.metric("Prediction",
                         "⚠️ Will Churn" if result["churn_prediction"] else "✅ Will Stay")
            col_c.markdown(f"### Risk Level\n{risk_badge(result['risk_level'])}")

            # Gauge
            prob = result["churn_probability"]
            fig, ax = plt.subplots(figsize=(5, 0.6))
            ax.barh(["Churn Risk"], [prob], color=("#e74c3c" if prob > 0.65
                                                    else "#f39c12" if prob > 0.35
                                                    else "#2ecc71"), height=0.5)
            ax.barh(["Churn Risk"], [1 - prob], left=[prob],
                    color="#ecf0f1", height=0.5)
            ax.set_xlim(0, 1)
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)

            # Recommendations
            st.markdown("### 💡 Retention Recommendations")
            if result["risk_level"] == "High":
                st.error(
                    "🔴 **Urgent action required.**\n"
                    "- Assign a dedicated relationship manager\n"
                    "- Offer a personalised retention package (fee waiver, upgrade)\n"
                    "- Proactive call within 48 hours"
                )
            elif result["risk_level"] == "Medium":
                st.warning(
                    "🟡 **Monitor closely.**\n"
                    "- Send targeted product recommendations\n"
                    "- Resolve any outstanding service issues promptly\n"
                    "- Enrol in loyalty rewards programme"
                )
            else:
                st.success(
                    "🟢 **Customer appears stable.**\n"
                    "- Continue regular engagement\n"
                    "- Consider upsell / cross-sell opportunities"
                )

        except FileNotFoundError:
            st.error("Model artefacts not found. Please run the training pipeline first:\n"
                     "`python -m src.models.train_model`")
        except Exception as e:
            st.exception(e)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Batch Prediction
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Batch Prediction":
    st.title("📊 Batch Customer Scoring")

    st.markdown("""
    Upload a CSV file with one row per customer.
    Required columns (same as single prediction form):

    `Age, Gender, MaritalStatus, IncomeLevel, TotalSpent, AvgTransactionValue,
    NumTransactions, TopProductCategory, NumServiceInteractions,
    UnresolvedInteractions, LoginFrequency, DaysSinceLastLogin, ServiceUsage`
    """)

    uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.info(f"Loaded {len(df_upload)} records")
            st.dataframe(df_upload.head())

            if st.button("🚀 Score All Customers", use_container_width=True):
                from src.models.predict import predict_batch
                with st.spinner("Scoring …"):
                    result_df = predict_batch(df_upload)

                st.success(f"✅ Scored {len(result_df)} customers")

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Predicted Churn",
                             int(result_df["ChurnPrediction"].sum()))
                col2.metric("High Risk",
                             int((result_df["RiskLevel"] == "High").sum()))
                col3.metric("Avg Churn Probability",
                             f"{result_df['ChurnProbability'].mean()*100:.1f} %")

                # Risk distribution chart
                fig, ax = plt.subplots(figsize=(5, 3))
                risk_counts = result_df["RiskLevel"].value_counts()
                colors = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
                risk_counts.plot(kind="bar", ax=ax,
                                  color=[colors.get(r, "grey") for r in risk_counts.index])
                ax.set_title("Risk Level Distribution")
                ax.set_ylabel("Customers")
                ax.set_xlabel("")
                st.pyplot(fig)

                # Download
                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Scored CSV",
                    data    = csv,
                    file_name = "churn_predictions.csv",
                    mime    = "text/csv",
                )

        except Exception as e:
            st.exception(e)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Model Insights
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Insights":
    st.title("📈 Model Performance & Insights")

    from src.utils.config import MODEL_DIR

    # Feature importance plot
    fi_path = MODEL_DIR / "feature_importances.png"
    if fi_path.exists():
        st.subheader("Top-20 Feature Importances (Random Forest)")
        st.image(str(fi_path), use_container_width=True)
    else:
        st.info("Train the model to see feature importances.")

    # Confusion matrices
    for label in ["Random Forest", "XGBoost"]:
        cm_path = MODEL_DIR / f"confusion_matrix_{label.lower().replace(' ', '_')}.png"
        if cm_path.exists():
            st.subheader(f"Confusion Matrix – {label}")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(str(cm_path), width=350)

    # Evaluation summary (static for display if no live run)
    st.subheader("Key Evaluation Metrics")
    st.markdown("""
    | Metric | Why it matters for Churn |
    |---|---|
    | **ROC-AUC** | Overall discriminative power across all thresholds |
    | **PR-AUC** | Performance on the minority (churn) class – critical for imbalanced data |
    | **Recall (Churn)** | % of churners correctly identified – missing a churner is costly |
    | **Precision (Churn)** | % of flagged customers who actually churn – avoids wasted outreach |
    | **F1-Score** | Harmonic mean of precision & recall |

    ### Recommended decision threshold
    The default probability threshold is **0.50**, but lowering it to **0.35–0.40**
    can increase recall (catch more churners) at the cost of more false positives.
    Choose the threshold based on the business cost ratio of:
    * **False negative** (missed churner) – lose the customer
    * **False positive** (unnecessary outreach) – marketing cost
    """)

    st.subheader("Business Recommendations")
    st.markdown("""
    1. **Tier-based intervention** – High-risk: relationship manager call; Medium-risk:
       personalised email campaign; Low-risk: standard newsletter.
    2. **Monthly re-scoring** – Retrain the model quarterly with fresh data to maintain
       predictive accuracy as customer behaviour evolves.
    3. **Feature enrichment** – Add net promoter score (NPS), mobile app session length,
       and balance trend to further improve predictive power.
    4. **Explainability** – Use SHAP values for per-customer explanations to help
       relationship managers understand **why** a customer is flagged.
    """)
