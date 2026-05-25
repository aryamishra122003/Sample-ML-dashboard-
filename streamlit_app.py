import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json

st.set_page_config(page_title="Godrej Capital ML Risk Classifier", layout="wide")

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("xgb_risk_model_v3.json")
    with open("model_meta_v3.json") as f:
        meta = json.load(f)
    return model, meta

@st.cache_data
def load_data():
    return pd.read_csv("prime_lap_senp_synthetic_10k.csv")

model, meta = load_model()
df = load_data()

LABEL_ORDER = meta["label_order"]
FEATURES = meta["feature_cols"]

FEATURE_LABELS = {
    "business_age_yrs": "Business Vintage",
    "cibil_score": "CIBIL Score",
    "num_overdue_accounts": "Overdue Accounts",
    "num_active_loans": "Active Loans",
    "loan_ask_lakhs": "Loan Ask",
    "revenue_lakhs": "Revenue",
    "gst_seasonality_cv": "GST Seasonality CV",
    "net_profit_margin_pct": "Net Profit Margin",
    "cash_profit_lakhs": "Cash Profit",
    "debt_to_equity": "Debt / Equity",
    "loan_to_net_worth": "Loan / Net Worth",
    "avg_bank_balance_lakhs": "Avg Bank Balance",
    "num_emi_bounces": "EMI Bounces",
    "foir": "FOIR",
    "cash_profit_to_loan": "Cash Profit / Loan",
}

RISK_COLORS = {"Low": "#34a853", "Medium": "#e8a317", "High": "#d93025"}

# Sidebar
with st.sidebar:
    st.markdown("### Model Stats")
    st.metric("Test Accuracy", f"{meta['test_accuracy']*100:.1f}%")
    st.metric("CV Accuracy", f"{meta['cv_accuracy_mean']*100:.1f}% +/- {meta['cv_accuracy_std']*100:.1f}%")
    st.metric("Training Cases", f"{meta['train_size']:,}")
    st.metric("Test Cases", f"{meta['test_size']:,}")
    st.metric("Total Dataset", f"{meta['total_cases']:,}")
    st.metric("Features", len(FEATURES))
    st.metric("Risk Classes", meta["n_classes"])
    st.markdown("---")
    st.markdown("**Class Distribution**")
    for cls in LABEL_ORDER:
        count = meta["class_distribution"].get(cls, 0)
        st.write(f"{cls}: {count:,} ({count/meta['total_cases']*100:.1f}%)")
    st.markdown("---")
    st.caption("XGBoost Classifier | Prototype v3 | SENP Business")

# Main
st.markdown("## Godrej Capital ML Risk Classifier")
st.markdown("Self-Employed Non-Professional cases, Light Underwriting stage")
st.markdown("---")

tab1, tab2 = st.tabs(["Classify a Case", "Training Data"])

with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Credit History **")
        business_age_yrs = st.number_input("Business Vintage (years)", 1, 40, 10)
        cibil_score = st.number_input("CIBIL Score", 300, 900, 720)
        num_overdue_accounts = st.number_input("Overdue Accounts", 0, 10, 0)
        num_active_loans = st.number_input("Active Loans", 0, 15, 2)

    with col2:
        st.markdown("**GST and other**")
        loan_ask_lakhs = st.number_input("Loan Ask (Rs Lakhs)", 10.0, 1500.0, 120.0, step=10.0)
        revenue_lakhs = st.number_input("Revenue (Rs Lakhs)", 0.0, 5000.0, 350.0, step=10.0)
        gst_seasonality_cv = st.number_input("GST Seasonality CV", 0.0, 1.0, 0.15, step=0.01,
                                              help="0 = stable quarterly sales, above 0.3 = volatile")
        net_profit_margin_pct = st.number_input("Net Profit Margin %", -20.0, 50.0, 15.0, step=0.5)
        cash_profit_lakhs = st.number_input("Cash Profit (Rs Lakhs)", 0.0, 500.0, 60.0, step=5.0)

    with col3:
        st.markdown("**Financials and Banking**")
        debt_to_equity = st.number_input("Debt / Equity", 0.0, 10.0, 0.6, step=0.1)
        loan_to_net_worth = st.number_input("Loan / Net Worth", 0.0, 10.0, 0.48, step=0.05)
        avg_bank_balance_lakhs = st.number_input("Avg Bank Balance (Rs Lakhs)", 0.0, 500.0, 15.0, step=1.0)
        num_emi_bounces = st.number_input("EMI Bounces (12M)", 0, 15, 0)

    # Derived
    estimated_emi = loan_ask_lakhs * 0.012
    foir = (estimated_emi + num_active_loans * 0.5) / (cash_profit_lakhs / 12 + 1e-6)
    cash_profit_to_loan = cash_profit_lakhs / (loan_ask_lakhs + 1e-6)

    st.markdown("---")

    if st.button("Classify Risk", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            'business_age_yrs': business_age_yrs,
            'cibil_score': cibil_score,
            'num_overdue_accounts': num_overdue_accounts,
            'num_active_loans': num_active_loans,
            'loan_ask_lakhs': loan_ask_lakhs,
            'revenue_lakhs': revenue_lakhs,
            'gst_seasonality_cv': gst_seasonality_cv,
            'net_profit_margin_pct': net_profit_margin_pct,
            'cash_profit_lakhs': cash_profit_lakhs,
            'debt_to_equity': debt_to_equity,
            'loan_to_net_worth': loan_to_net_worth,
            'avg_bank_balance_lakhs': avg_bank_balance_lakhs,
            'num_emi_bounces': num_emi_bounces,
            'foir': foir,
            'cash_profit_to_loan': cash_profit_to_loan,
        }])

        pred_idx = model.predict(input_data)[0]
        pred_proba = model.predict_proba(input_data)[0]
        pred_label = LABEL_ORDER[pred_idx]

        st.markdown("---")
        r1, r2 = st.columns([1, 2])

        with r1:
            color = RISK_COLORS[pred_label]
            confidence = float(pred_proba[pred_idx]) * 100
            st.markdown(f"""
            <div style="text-align:center; padding:20px; border:3px solid {color}; border-radius:12px; background:{color}11">
                <div style="font-size:13px; color:#666; margin-bottom:4px">RISK CLASSIFICATION</div>
                <div style="font-size:36px; font-weight:900; color:{color}">{pred_label.upper()}</div>
                <div style="font-size:12px; color:#888; margin-top:6px">Confidence: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Derived Ratios**")
            foir_delta = "High" if foir > 0.5 else "Normal"
            cp_delta = "Low" if cash_profit_to_loan < 0.3 else "Adequate"
            st.metric("FOIR", f"{foir:.3f}", delta=foir_delta, delta_color="inverse")
            st.metric("Cash Profit / Loan", f"{cash_profit_to_loan:.3f}", delta=cp_delta)

        with r2:
            st.markdown("**Class Probabilities**")
            prob_df = pd.DataFrame({
                "Class": LABEL_ORDER,
                "Probability": [float(pred_proba[i]) * 100 for i in range(len(LABEL_ORDER))]
            })
            for _, row in prob_df.iterrows():
                pct = row["Probability"]
                cls = row["Class"]
                bar_color = RISK_COLORS.get(cls, "#999")
                st.markdown(f"""
                <div style="margin-bottom:8px">
                    <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:3px">
                        <span style="font-weight:600">{cls}</span>
                        <span style="color:#666">{pct:.1f}%</span>
                    </div>
                    <div style="background:#f0f2f5; border-radius:4px; height:12px; overflow:hidden">
                        <div style="width:{pct}%; height:100%; background:{bar_color}; border-radius:4px; opacity:0.8"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # SHAP
            st.markdown("**SHAP Feature Contributions**")
            st.caption("How each feature pushed the prediction for this specific case")
            try:
                import shap
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_data)

                if isinstance(shap_values, list):
                    sv = shap_values[pred_idx][0]
                elif shap_values.ndim == 3:
                    sv = shap_values[0, :, pred_idx]
                else:
                    sv = shap_values[0]

                shap_df = pd.DataFrame({
                    'Feature': [FEATURE_LABELS.get(f, f) for f in FEATURES],
                    'SHAP Value': [float(v) for v in sv]
                }).sort_values('SHAP Value', key=abs, ascending=True)

                colors = ['#34a853' if v < 0 else '#d93025' for v in shap_df['SHAP Value']]
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6, 4.5))
                ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors, height=0.6)
                ax.axvline(0, color='#999', linewidth=0.8)
                ax.set_xlabel('SHAP Value (impact on prediction)', fontsize=9)
                ax.tick_params(labelsize=8.5)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"SHAP visualization unavailable: {e}")

with tab2:
    st.markdown("**Synthetic Training Dataset: 10,000 SENP Business Cases**")
    st.markdown(f"Columns: {len(df.columns)} | Rows: {len(df):,} | Trained on 12% misclassified or missing value cases")

    col_filter, _ = st.columns([1, 3])
    with col_filter:
        class_filter = st.selectbox("Filter by Risk Label", ["All"] + LABEL_ORDER)

    display_df = df if class_filter == "All" else df[df["risk_label_3class"] == class_filter]
    st.dataframe(display_df, use_container_width=True, height=500)

    st.download_button("Download Full Dataset (CSV)", df.to_csv(index=False).encode(),
                        "prime_lap_senp_synthetic_10k.csv", "text/csv")
