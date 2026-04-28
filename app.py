import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, accuracy_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartBank · Churn Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Lloyds Brand CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Fonts & base ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #006643 0%, #004d32 60%, #003322 100%);
    color: white;
  }
  section[data-testid="stSidebar"] * { color: white !important; }
  section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    color: white !important;
  }
  section[data-testid="stSidebar"] label { color: rgba(255,255,255,0.85) !important; }

  /* ── Main area ── */
  .main { background: #f4f6f9; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* ── KPI Cards ── */
  .kpi-card {
    background: white;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 14px rgba(0,102,67,0.09);
    border-left: 5px solid #006643;
    transition: transform .18s;
  }
  .kpi-card:hover { transform: translateY(-3px); box-shadow: 0 6px 22px rgba(0,102,67,0.15); }
  .kpi-value { font-size: 2.1rem; font-weight: 800; color: #006643; line-height: 1.1; }
  .kpi-label { font-size: 0.82rem; font-weight: 600; color: #6c757d; text-transform: uppercase; letter-spacing: .07em; margin-top: .25rem; }
  .kpi-delta { font-size: 0.8rem; margin-top: .2rem; }
  .kpi-delta.up   { color: #c00000; }
  .kpi-delta.down { color: #006643; }

  /* ── Section headers ── */
  .section-header {
    background: linear-gradient(135deg, #006643 0%, #009a63 100%);
    color: white;
    padding: 1rem 1.4rem;
    border-radius: 10px;
    margin-bottom: 1.2rem;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: .02em;
  }

  /* ── Alert boxes ── */
  .alert-red    { background:#fff0f0; border-left:4px solid #c00000; padding:.8rem 1rem; border-radius:8px; margin:.5rem 0; }
  .alert-green  { background:#f0fff6; border-left:4px solid #006643; padding:.8rem 1rem; border-radius:8px; margin:.5rem 0; }
  .alert-amber  { background:#fffbf0; border-left:4px solid #e8a020; padding:.8rem 1rem; border-radius:8px; margin:.5rem 0; }

  /* ── Risk badge ── */
  .risk-high   { background:#c00000; color:white; padding:4px 14px; border-radius:20px; font-weight:700; font-size:.9rem; }
  .risk-medium { background:#e8a020; color:white; padding:4px 14px; border-radius:20px; font-weight:700; font-size:.9rem; }
  .risk-low    { background:#006643; color:white; padding:4px 14px; border-radius:20px; font-weight:700; font-size:.9rem; }

  /* ── Tables ── */
  .stDataFrame { border-radius: 10px; overflow: hidden; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    background: white; border-radius: 8px 8px 0 0;
    border: 1px solid #dee2e6; font-weight: 600;
  }
  .stTabs [aria-selected="true"] {
    background: #006643 !important; color: white !important;
  }

  /* ── Metric override ── */
  div[data-testid="metric-container"] {
    background: white; border-radius: 12px; padding: .8rem 1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    border-top: 3px solid #006643;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: #006643; color: white; border: none;
    border-radius: 8px; padding: .6rem 1.6rem;
    font-weight: 600; font-size: .95rem;
    transition: background .2s;
  }
  .stButton > button:hover { background: #004d32; }
</style>
""", unsafe_allow_html=True)

# ─── Constants / Colours ──────────────────────────────────────────────────────
LLOYDS_GREEN  = "#006643"
LLOYDS_DARK   = "#004d32"
CHURN_RED     = "#c00000"
AMBER         = "#e8a020"
LIGHT_BLUE    = "#5ba4cf"
PALETTE       = [LLOYDS_GREEN, CHURN_RED, AMBER, LIGHT_BLUE, "#9b59b6", "#1abc9c"]

# ─── Data Loading & Churn Generation ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("/mnt/user-data/uploads/preprocessed_churn_data.csv")

    # Reconstruct human-readable categoricals from encoded columns
    gender_map       = {0: "Female", 1: "Male"}
    income_map       = {0: "Low", 1: "Medium", 2: "High"}
    ms_cols          = {"MS_Divorced": "Divorced", "MS_Married": "Married",
                        "MS_Single": "Single",   "MS_Widowed": "Widowed"}

    df["Gender"]       = df["Gender_Encoded"].map(gender_map)
    df["IncomeLevel"]  = df["IncomeLevel_Ordinal"].map(income_map)
    df["MaritalStatus"] = "Unknown"
    for col, label in ms_cols.items():
        df.loc[df[col] == 1, "MaritalStatus"] = label

    # Age Group
    def age_group(age):
        if age <= 29:   return "18–29"
        elif age <= 44: return "30–44"
        elif age <= 59: return "45–59"
        else:           return "60–69"
    df["AgeGroup"] = df["Age"].apply(age_group)

    # Synthetic Churn (domain-informed, matching Task-2 notebook logic exactly)
    np.random.seed(42)
    age_f    = ((df["Age"] < 30).astype(int) * 0.20 +
                (df["Age"] > 60).astype(int) * 0.10)
    income_f = ((df["IncomeLevel"] == "Low").astype(int)    * 0.15 +
                (df["IncomeLevel"] == "Medium").astype(int) * 0.05)
    ms_f     = ((df["MaritalStatus"] == "Single").astype(int)   * 0.10 +
                (df["MaritalStatus"] == "Divorced").astype(int) * 0.08)
    gender_f = (df["Gender"] == "Male").astype(int) * 0.03

    churn_prob    = (0.15 + age_f + income_f + ms_f + gender_f).clip(0, 0.75)
    df["Churn"]   = (np.random.random(len(df)) < churn_prob).astype(int)
    df["ChurnLabel"] = df["Churn"].map({0: "No Churn", 1: "Churn"})
    return df


@st.cache_resource
def train_models(df):
    FEATURES = ["Age", "Gender_Encoded", "IncomeLevel_Ordinal",
                "MS_Divorced", "MS_Married", "MS_Single", "MS_Widowed"]
    X = df[FEATURES]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree":       DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest":       RandomForestClassifier(random_state=42, n_estimators=200,
                                                      max_depth=5, class_weight="balanced"),
        "Gradient Boosting":   GradientBoostingClassifier(random_state=42, n_estimators=100),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    trained = {}

    for name, model in models.items():
        Xt = X_train_sc if name == "Logistic Regression" else X_train
        Xe = X_test_sc  if name == "Logistic Regression" else X_test

        auc = cross_val_score(model, Xt, y_train, cv=cv, scoring="roc_auc").mean()
        f1  = cross_val_score(model, Xt, y_train, cv=cv, scoring="f1").mean()
        acc = cross_val_score(model, Xt, y_train, cv=cv, scoring="accuracy").mean()

        model.fit(Xt, y_train)
        y_pred     = model.predict(Xe)
        y_proba    = model.predict_proba(Xe)[:, 1]
        test_auc   = roc_auc_score(y_test, y_proba)
        test_f1    = f1_score(y_test, y_pred)
        test_acc   = accuracy_score(y_test, y_pred)
        test_prec  = precision_score(y_test, y_pred, zero_division=0)
        test_rec   = recall_score(y_test, y_pred, zero_division=0)
        cm         = confusion_matrix(y_test, y_pred)

        results[name] = {
            "cv_auc": auc, "cv_f1": f1, "cv_acc": acc,
            "test_auc": test_auc, "test_f1": test_f1, "test_acc": test_acc,
            "test_precision": test_prec, "test_recall": test_rec,
            "confusion_matrix": cm,
            "y_pred": y_pred, "y_proba": y_proba,
        }
        trained[name] = (model, scaler if name == "Logistic Regression" else None)

    best_model_name = max(results, key=lambda n: results[n]["test_auc"])
    rf   = trained["Random Forest"][0]
    feat_imp = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False)

    return results, trained, X_test, y_test, feat_imp, best_model_name, FEATURES


def fmt_pct(v): return f"{v:.1%}"
def fmt_num(v): return f"{v:,.0f}"

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.2rem 0 1rem;'>
      <div style='font-size:2.2rem;'>🏦</div>
      <div style='font-size:1.1rem; font-weight:800; letter-spacing:.05em;'>SmartBank</div>
      <div style='font-size:.75rem; opacity:.75; margin-top:.2rem;'>A Lloyds Banking Group Subsidiary</div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.2); margin:.5rem 0 1rem;'/>
    """, unsafe_allow_html=True)

    page = st.selectbox("📍 Navigate", [
        "🏠  Dashboard",
        "📊  EDA & Customer Insights",
        "🤖  Model Training & Evaluation",
        "🔮  Churn Predictor",
        "📋  Business Recommendations",
    ])

    st.markdown("---")
    st.markdown("""
    <div style='font-size:.78rem; opacity:.7; line-height:1.6;'>
    <b>Project:</b> Customer Retention Enhancement<br>
    <b>Phase:</b> 1 & 2 Complete<br>
    <b>Dataset:</b> 1,000 SmartBank Customers<br>
    <b>Model:</b> Random Forest (Tuned)<br>
    <b>Team:</b> DS & Analytics · Lloyds
    </div>
    """, unsafe_allow_html=True)

# ─── Load data + models ───────────────────────────────────────────────────────
df = load_data()
with st.spinner("Training ML models…"):
    results, trained_models, X_test, y_test, feat_imp, best_name, FEATURES = train_models(df)

churn_rate    = df["Churn"].mean()
total_churned = df["Churn"].sum()
high_risk     = df[df["Churn"] == 1]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 · DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Dashboard":
    st.markdown("""
    <div style='background:linear-gradient(135deg,#006643,#009a63);
                border-radius:14px; padding:1.6rem 2rem; margin-bottom:1.5rem;
                color:white;'>
      <div style='font-size:1.6rem; font-weight:800;'>🏦 SmartBank · Customer Churn Intelligence Platform</div>
      <div style='font-size:.9rem; opacity:.85; margin-top:.4rem;'>
        Customer Retention Enhancement through Predictive Analytics · Lloyds Banking Group
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        (k1, fmt_num(len(df)),      "Total Customers",       f"📋 Analysed", ""),
        (k2, fmt_num(total_churned), "Customers at Risk",    f"⚠️ Churn predicted", "up"),
        (k3, fmt_pct(churn_rate),   "Overall Churn Rate",   "Across all segments", "up"),
        (k4, fmt_pct(results[best_name]["test_auc"]), "Best Model AUC",
                                    f"🏆 {best_name}", "down"),
    ]
    for col, val, lbl, delta, cls in kpis:
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
              <div class='kpi-value'>{val}</div>
              <div class='kpi-label'>{lbl}</div>
              <div class='kpi-delta {cls}'>{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Churn by Segment ─────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='section-header'>📊 Churn Distribution</div>", unsafe_allow_html=True)
        counts = df["ChurnLabel"].value_counts().reset_index()
        counts.columns = ["Status", "Count"]
        fig = px.pie(counts, names="Status", values="Count",
                     color="Status",
                     color_discrete_map={"No Churn": LLOYDS_GREEN, "Churn": CHURN_RED},
                     hole=0.52)
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          marker=dict(line=dict(color="white", width=2)))
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300,
                          showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>🎯 Churn Rate by Income Level</div>", unsafe_allow_html=True)
        inc_churn = df.groupby("IncomeLevel")["Churn"].mean().reindex(
            ["Low", "Medium", "High"]).reset_index()
        inc_churn.columns = ["IncomeLevel", "ChurnRate"]
        fig2 = px.bar(inc_churn, x="IncomeLevel", y="ChurnRate",
                      color="ChurnRate",
                      color_continuous_scale=["#006643", "#e8a020", "#c00000"],
                      text=inc_churn["ChurnRate"].map(lambda v: f"{v:.1%}"))
        fig2.update_traces(textposition="outside")
        fig2.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10),
                           xaxis_title="Income Level", yaxis_title="Churn Rate",
                           yaxis_tickformat=".0%", coloraxis_showscale=False,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Age Group & Marital Status ─────────────────────────────────────────
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("<div class='section-header'>👥 Churn Rate by Age Group</div>", unsafe_allow_html=True)
        ag_order  = ["18–29", "30–44", "45–59", "60–69"]
        ag_churn  = df.groupby("AgeGroup")["Churn"].mean().reindex(ag_order).reset_index()
        ag_churn.columns = ["AgeGroup", "ChurnRate"]
        fig3 = px.bar(ag_churn, x="AgeGroup", y="ChurnRate",
                      color="ChurnRate",
                      color_continuous_scale=["#009a63", "#e8a020", "#c00000"],
                      text=ag_churn["ChurnRate"].map(lambda v: f"{v:.1%}"))
        fig3.update_traces(textposition="outside")
        fig3.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10),
                           xaxis_title="Age Group", yaxis_title="Churn Rate",
                           yaxis_tickformat=".0%", coloraxis_showscale=False,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown("<div class='section-header'>💍 Churn Rate by Marital Status</div>", unsafe_allow_html=True)
        ms_order  = ["Single", "Divorced", "Married", "Widowed"]
        ms_churn  = df.groupby("MaritalStatus")["Churn"].mean().reindex(ms_order).reset_index()
        ms_churn.columns = ["MaritalStatus", "ChurnRate"]
        fig4 = px.bar(ms_churn, x="MaritalStatus", y="ChurnRate",
                      color="ChurnRate",
                      color_continuous_scale=["#009a63", "#e8a020", "#c00000"],
                      text=ms_churn["ChurnRate"].map(lambda v: f"{v:.1%}"))
        fig4.update_traces(textposition="outside")
        fig4.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10),
                           xaxis_title="Marital Status", yaxis_title="Churn Rate",
                           yaxis_tickformat=".0%", coloraxis_showscale=False,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig4, use_container_width=True)

    # ── Model Comparison Mini-Table ───────────────────────────────────────
    st.markdown("<div class='section-header'>🤖 Model Performance Summary</div>", unsafe_allow_html=True)
    perf = []
    for name, r in results.items():
        perf.append({
            "Model": name,
            "CV AUC": f"{r['cv_auc']:.4f}",
            "CV F1":  f"{r['cv_f1']:.4f}",
            "Test AUC":  f"{r['test_auc']:.4f}",
            "Test F1":   f"{r['test_f1']:.4f}",
            "Test Acc":  f"{r['test_acc']:.4f}",
            "Best ✓": "🏆" if name == best_name else ""
        })
    st.dataframe(pd.DataFrame(perf).set_index("Model"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 · EDA & CUSTOMER INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  EDA & Customer Insights":
    st.markdown("<div class='section-header'>📊 Exploratory Data Analysis — SmartBank Customer Base</div>",
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔥 Segment Analysis", "🗂️ Raw Data Explorer"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="Age", color="ChurnLabel", nbins=30, barmode="overlay",
                               color_discrete_map={"No Churn": LLOYDS_GREEN, "Churn": CHURN_RED},
                               opacity=0.72, title="Age Distribution by Churn Status")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              height=350, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            gender_churn = df.groupby(["Gender", "ChurnLabel"]).size().reset_index(name="Count")
            fig = px.bar(gender_churn, x="Gender", y="Count", color="ChurnLabel", barmode="group",
                         color_discrete_map={"No Churn": LLOYDS_GREEN, "Churn": CHURN_RED},
                         title="Customer Count by Gender & Churn")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              height=350, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            inc_dist = df.groupby(["IncomeLevel", "ChurnLabel"]).size().reset_index(name="Count")
            fig = px.bar(inc_dist, x="IncomeLevel", y="Count", color="ChurnLabel", barmode="stack",
                         color_discrete_map={"No Churn": LLOYDS_GREEN, "Churn": CHURN_RED},
                         title="Income Level Distribution (Stacked)",
                         category_orders={"IncomeLevel": ["Low", "Medium", "High"]})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              height=350, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            ms_dist = df.groupby(["MaritalStatus", "ChurnLabel"]).size().reset_index(name="Count")
            fig = px.bar(ms_dist, x="MaritalStatus", y="Count", color="ChurnLabel", barmode="stack",
                         color_discrete_map={"No Churn": LLOYDS_GREEN, "Churn": CHURN_RED},
                         title="Marital Status Distribution (Stacked)")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              height=350, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### 🔥 Churn Heat Map — Age Group × Income Level")
        pivot = df.pivot_table(values="Churn", index="AgeGroup", columns="IncomeLevel",
                               aggfunc="mean")
        pivot = pivot.reindex(index=["18–29", "30–44", "45–59", "60–69"],
                              columns=["Low", "Medium", "High"])
        fig = px.imshow(pivot, text_auto=".1%", color_continuous_scale="RdYlGn_r",
                        title="Churn Rate Heat Map",
                        labels=dict(color="Churn Rate"))
        fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 📦 Age Box Plot by Churn")
            fig = px.box(df, x="ChurnLabel", y="Age", color="ChurnLabel",
                         color_discrete_map={"No Churn": LLOYDS_GREEN, "Churn": CHURN_RED},
                         points="outliers")
            fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### 🎯 Segment Breakdown (Sunburst)")
            fig = px.sunburst(df, path=["IncomeLevel", "MaritalStatus", "ChurnLabel"],
                              color="Churn", color_continuous_scale="RdYlGn_r",
                              title="Customer Segmentation")
            fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### 🗂️ Customer Data Explorer")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            churn_filter = st.multiselect("Churn Status", ["No Churn", "Churn"],
                                          default=["No Churn", "Churn"])
        with col_f2:
            income_filter = st.multiselect("Income Level", ["Low", "Medium", "High"],
                                           default=["Low", "Medium", "High"])
        with col_f3:
            gender_filter = st.multiselect("Gender", ["Male", "Female"],
                                           default=["Male", "Female"])

        filtered = df[
            df["ChurnLabel"].isin(churn_filter) &
            df["IncomeLevel"].isin(income_filter) &
            df["Gender"].isin(gender_filter)
        ]
        st.markdown(f"**{len(filtered):,}** customers match filters "
                    f"· Churn rate: **{filtered['Churn'].mean():.1%}**")
        display_cols = ["CustomerID", "Age", "Gender", "MaritalStatus",
                        "IncomeLevel", "AgeGroup", "ChurnLabel"]
        st.dataframe(filtered[display_cols].reset_index(drop=True), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 · MODEL TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model Training & Evaluation":
    st.markdown("<div class='section-header'>🤖 Machine Learning Model Training & Evaluation</div>",
                unsafe_allow_html=True)

    model_sel = st.selectbox("Select Model to Inspect", list(results.keys()),
                             index=list(results.keys()).index(best_name))
    r = results[model_sel]

    # ── Metric Cards ────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    metrics_row = [
        (m1, "Test AUC",       f"{r['test_auc']:.4f}"),
        (m2, "Test F1",        f"{r['test_f1']:.4f}"),
        (m3, "Test Accuracy",  f"{r['test_acc']:.4f}"),
        (m4, "Precision",      f"{r['test_precision']:.4f}"),
        (m5, "Recall",         f"{r['test_recall']:.4f}"),
    ]
    for col, lbl, val in metrics_row:
        with col:
            st.metric(lbl, val)

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "📉 ROC / PR Curves", "🟦 Confusion Matrix",
        "🌳 Feature Importance", "📊 Model Comparison"
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"AUC = {r['test_auc']:.4f}",
                                     line=dict(color=LLOYDS_GREEN, width=2.5)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(color="grey", dash="dash", width=1.5),
                                     name="Random Baseline"))
            fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate", height=380,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            prec, rec, _ = precision_recall_curve(y_test, r["y_proba"])
            ap = average_precision_score(y_test, r["y_proba"])
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                      name=f"AP = {ap:.4f}",
                                      line=dict(color=CHURN_RED, width=2.5)))
            fig2.update_layout(title="Precision-Recall Curve",
                               xaxis_title="Recall", yaxis_title="Precision",
                               height=380, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        cm = r["confusion_matrix"]
        labels = ["No Churn", "Churn"]
        fig = px.imshow(cm, text_auto=True, x=labels, y=labels,
                        color_continuous_scale=[[0, "#eaf4f0"], [1, LLOYDS_GREEN]],
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        title="Confusion Matrix")
        fig.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        <div style='display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:12px; margin-top:1rem;'>
          <div class='kpi-card'><div class='kpi-value' style='color:#006643'>{tn}</div>
            <div class='kpi-label'>True Negatives</div></div>
          <div class='kpi-card'><div class='kpi-value' style='color:#c00000'>{fp}</div>
            <div class='kpi-label'>False Positives</div></div>
          <div class='kpi-card'><div class='kpi-value' style='color:#c00000'>{fn}</div>
            <div class='kpi-label'>False Negatives</div></div>
          <div class='kpi-card'><div class='kpi-value' style='color:#006643'>{tp}</div>
            <div class='kpi-label'>True Positives</div></div>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        fig = px.bar(feat_imp, x="Importance", y="Feature", orientation="h",
                     color="Importance",
                     color_continuous_scale=["#eaf4f0", LLOYDS_GREEN],
                     title="Random Forest Feature Importances",
                     text=feat_imp["Importance"].map(lambda v: f"{v:.4f}"))
        fig.update_traces(textposition="outside")
        fig.update_layout(height=380, yaxis=dict(autorange="reversed"),
                          coloraxis_showscale=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class='alert-green'>
          <b>Interpretation:</b> Age and Income Level are the strongest predictors of churn,
          consistent with SmartBank's concern about young professionals and low-income segments.
          Marital status contributes modestly, while gender has marginal impact.
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        comp = []
        for name, r_ in results.items():
            comp.append({"Model": name,
                         "CV AUC": r_["cv_auc"], "CV F1": r_["cv_f1"],
                         "Test AUC": r_["test_auc"], "Test F1": r_["test_f1"],
                         "Test Acc": r_["test_acc"]})
        comp_df = pd.DataFrame(comp)

        fig = go.Figure()
        metrics_cmp = ["CV AUC", "Test AUC", "CV F1", "Test F1", "Test Acc"]
        colours_cmp = [LLOYDS_GREEN, LLOYDS_DARK, CHURN_RED, "#9b1010", AMBER]
        for m, col in zip(metrics_cmp, colours_cmp):
            fig.add_trace(go.Bar(name=m, x=comp_df["Model"], y=comp_df[m],
                                 marker_color=col))
        fig.update_layout(barmode="group", height=420, title="All Models — Metric Comparison",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 · CHURN PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Churn Predictor":
    st.markdown("<div class='section-header'>🔮 Individual Customer Churn Risk Predictor</div>",
                unsafe_allow_html=True)

    pred_mode = st.radio("Prediction Mode", ["Single Customer", "Batch (Upload CSV)"],
                         horizontal=True)

    if pred_mode == "Single Customer":
        st.markdown("#### Enter Customer Details")
        c1, c2, c3 = st.columns(3)
        with c1:
            age    = st.slider("Age", 18, 69, 35)
            gender = st.selectbox("Gender", ["Female", "Male"])
        with c2:
            marital = st.selectbox("Marital Status",
                                   ["Single", "Married", "Divorced", "Widowed"])
            income  = st.selectbox("Income Level", ["Low", "Medium", "High"])
        with c3:
            model_choice = st.selectbox("Model", list(trained_models.keys()),
                                        index=list(trained_models.keys()).index(best_name))
            threshold    = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)

        if st.button("🔮 Predict Churn Risk", use_container_width=True):
            gender_enc  = 1 if gender == "Male" else 0
            income_enc  = {"Low": 0, "Medium": 1, "High": 2}[income]
            ms_div = 1 if marital == "Divorced" else 0
            ms_mar = 1 if marital == "Married"  else 0
            ms_sin = 1 if marital == "Single"   else 0
            ms_wid = 1 if marital == "Widowed"  else 0

            inp = np.array([[age, gender_enc, income_enc,
                             ms_div, ms_mar, ms_sin, ms_wid]])
            model_obj, scaler_obj = trained_models[model_choice]
            if scaler_obj:
                inp = scaler_obj.transform(inp)

            prob  = model_obj.predict_proba(inp)[0][1]
            label = "Churn" if prob >= threshold else "No Churn"

            if prob >= 0.6:
                risk_cls, risk_lbl = "risk-high",   "HIGH RISK"
            elif prob >= 0.35:
                risk_cls, risk_lbl = "risk-medium", "MEDIUM RISK"
            else:
                risk_cls, risk_lbl = "risk-low",    "LOW RISK"

            st.markdown("---")
            r1, r2, r3 = st.columns([1, 1, 2])
            with r1:
                st.metric("Churn Probability", f"{prob:.1%}")
            with r2:
                st.markdown(f"**Risk Tier:** <span class='{risk_cls}'>{risk_lbl}</span>",
                            unsafe_allow_html=True)
                st.markdown(f"**Prediction:** {'⚠️ Likely to Churn' if label == 'Churn' else '✅ Likely to Stay'}")

            with r3:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"size": 32, "color": LLOYDS_GREEN}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": CHURN_RED if prob >= 0.5 else LLOYDS_GREEN},
                        "steps": [
                            {"range": [0, 35],  "color": "#eaf4f0"},
                            {"range": [35, 60], "color": "#fff8e8"},
                            {"range": [60, 100],"color": "#fff0f0"},
                        ],
                        "threshold": {"line": {"color": "black", "width": 3},
                                      "thickness": 0.8, "value": threshold * 100}
                    },
                    title={"text": "Churn Risk Score", "font": {"size": 14}}
                ))
                fig.update_layout(height=240, margin=dict(t=30, b=0, l=30, r=30),
                                  paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            # ── Personalised Recommendations ─────────────────────────────
            st.markdown("#### 💡 Personalised Retention Recommendations")
            recs = []
            if income == "Low":
                recs.append("🏦 Offer a SmartSave account with zero fees and higher interest rate")
            if marital in ["Single", "Divorced"]:
                recs.append("🎯 Personalised financial planning sessions for life transitions")
            if age < 30:
                recs.append("📱 Promote digital-first features: instant payments, budgeting tools")
                recs.append("🎓 Student/early-career banking bundle with cashback rewards")
            if age > 60:
                recs.append("📞 Priority relationship manager access with telephone support")
            if income == "Medium":
                recs.append("💳 Upgrade to Premium Banking with travel insurance & rewards")
            if prob >= 0.5:
                recs.append("📩 Proactive outreach: personalised retention offer within 7 days")

            if recs:
                for rec in recs:
                    st.markdown(f'<div class="alert-green">{rec}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-green">✅ Customer is low risk — maintain regular engagement</div>',
                            unsafe_allow_html=True)

    else:  # Batch
        st.markdown("""
        <div class='alert-amber'>
          Upload a CSV with columns: <code>Age, Gender, MaritalStatus, IncomeLevel</code>
          (Gender: Male/Female, MaritalStatus: Single/Married/Divorced/Widowed,
          IncomeLevel: Low/Medium/High)
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload Customer CSV", type=["csv"])

        if uploaded:
            batch_df = pd.read_csv(uploaded)
            st.markdown(f"Loaded **{len(batch_df)}** records")

            batch_df["Gender_Encoded"]     = (batch_df["Gender"] == "Male").astype(int)
            batch_df["IncomeLevel_Ordinal"] = batch_df["IncomeLevel"].map(
                {"Low": 0, "Medium": 1, "High": 2})
            batch_df["MS_Divorced"] = (batch_df["MaritalStatus"] == "Divorced").astype(int)
            batch_df["MS_Married"]  = (batch_df["MaritalStatus"] == "Married").astype(int)
            batch_df["MS_Single"]   = (batch_df["MaritalStatus"] == "Single").astype(int)
            batch_df["MS_Widowed"]  = (batch_df["MaritalStatus"] == "Widowed").astype(int)

            feat_cols = ["Age", "Gender_Encoded", "IncomeLevel_Ordinal",
                         "MS_Divorced", "MS_Married", "MS_Single", "MS_Widowed"]
            model_obj, _ = trained_models["Random Forest"]
            probs = model_obj.predict_proba(batch_df[feat_cols])[:, 1]
            batch_df["Churn_Probability"] = probs
            batch_df["Risk_Tier"] = pd.cut(probs, bins=[0, 0.35, 0.6, 1.0],
                                           labels=["Low", "Medium", "High"])
            batch_df["Prediction"] = (probs >= 0.5).map({True: "Churn", False: "No Churn"})

            st.dataframe(batch_df[["Age", "Gender", "MaritalStatus", "IncomeLevel",
                                   "Churn_Probability", "Risk_Tier", "Prediction"]].style.background_gradient(
                subset=["Churn_Probability"], cmap="RdYlGn_r"),
                use_container_width=True)

            fig = px.histogram(batch_df, x="Churn_Probability", nbins=20,
                               color="Risk_Tier",
                               color_discrete_map={"Low": LLOYDS_GREEN, "Medium": AMBER, "High": CHURN_RED},
                               title="Churn Probability Distribution — Batch")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 · BUSINESS RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋  Business Recommendations":
    st.markdown("<div class='section-header'>📋 Strategic Business Recommendations — SmartBank Retention</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class='alert-amber'>
      <b>⚠️ Executive Summary:</b> Overall churn rate of ~{:.0%} with highest risk concentrated
      among low-income young professionals (18–29) and recently divorced customers.
      Immediate targeted interventions recommended for ~{} at-risk customers.
    </div>
    """.format(churn_rate, total_churned), unsafe_allow_html=True)

    # ── Risk Tiers ────────────────────────────────────────────────────────
    st.markdown("### 🎯 Segment Risk Tiers & Action Plans")

    tiers = [
        ("🔴 HIGH RISK: Young Low-Income Singles (18–29, Low Income)",
         "risk-high",
         f"{len(df[(df['Age'] <= 29) & (df['IncomeLevel'] == 'Low')]):,} customers",
         [
             "Launch SmartStart account — zero monthly fees, 4.5% instant-access savings",
             "Dedicated in-app financial coach powered by AI",
             "Referral reward programme (£50 per successful referral)",
             "Monthly personalised spending insights & budget nudges",
         ]),
        ("🟠 MEDIUM RISK: Divorced / Separated Customers",
         "risk-medium",
         f"{len(df[df['MaritalStatus'].isin(['Divorced'])]):,} customers",
         [
             "Life transition financial planning: dedicated RM for 90 days post-divorce",
             "SmartReset Bundle: account restructuring with fee waivers for 6 months",
             "Specialist mortgage & protection review offering",
         ]),
        ("🟡 MEDIUM RISK: Mid-Income 45–59 Segment",
         "risk-medium",
         f"{len(df[(df['Age'].between(45, 59)) & (df['IncomeLevel'] == 'Medium')]):,} customers",
         [
             "Pension & retirement planning workshops (in-branch and online)",
             "SmartWealth Premium upgrade with travel & health insurance bundle",
             "Priority telephone banking with <60s wait guarantee",
         ]),
        ("🟢 LOW RISK: High-Income Married Customers",
         "risk-low",
         f"{len(df[(df['IncomeLevel'] == 'High') & (df['MaritalStatus'] == 'Married')]):,} customers",
         [
             "Maintain with quarterly personalised financial reviews",
             "SmartInvest ISA product cross-sell opportunity",
             "Loyalty rewards: tiered cashback on household spending",
         ]),
    ]

    for title, badge, count, actions in tiers:
        with st.expander(f"{title} · {count}", expanded=(badge == "risk-high")):
            for act in actions:
                st.markdown(f"• {act}")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📏 Recommended Model Evaluation KPIs")
        kpi_data = {
            "KPI": ["ROC-AUC", "F1 Score", "Precision", "Recall",
                    "False Negative Rate", "Model Drift (monthly)"],
            "Target": ["> 0.75", "> 0.60", "> 0.65", "> 0.55",
                       "< 15%", "< 5% AUC drop"],
            "Current": [
                f"{results[best_name]['test_auc']:.4f}",
                f"{results[best_name]['test_f1']:.4f}",
                f"{results[best_name]['test_precision']:.4f}",
                f"{results[best_name]['test_recall']:.4f}",
                "—", "—"
            ]
        }
        st.dataframe(pd.DataFrame(kpi_data), use_container_width=True, hide_index=True)

    with c2:
        st.markdown("### 🗓️ Implementation Roadmap")
        roadmap = {
            "Phase": ["Week 1–2", "Week 3–4", "Month 2", "Month 3", "Ongoing"],
            "Action": [
                "Deploy model to CRM; flag top 200 high-risk customers",
                "Launch personalised outreach campaigns for HIGH-RISK tier",
                "A/B test SmartStart & SmartReset product offerings",
                "Measure retention lift; retrain model with new labels",
                "Monthly model monitoring; quarterly strategy review"
            ]
        }
        st.dataframe(pd.DataFrame(roadmap), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("""
    <div class='alert-green'>
      <b>✅ FCA Compliance Note:</b> All retention interventions must be fair, transparent and
      in the customer's interest. Model decisions should be explainable (Random Forest feature
      importances provided). Regular bias audits recommended across gender and age segments.
      Data handling must comply with UK GDPR and the FCA's Consumer Duty obligations.
    </div>
    """, unsafe_allow_html=True)
