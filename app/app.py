
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Hiring Risk Predictor",
    page_icon="🧠",
    layout="wide"
)

# ── Load Artifacts ────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("models/tuned_model.pkl",  "rb") as f: model    = pickle.load(f)
    with open("models/scaler.pkl",       "rb") as f: scaler   = pickle.load(f)
    with open("models/edu_encoder.pkl",  "rb") as f: edu_enc  = pickle.load(f)
    with open("models/feature_list.pkl", "rb") as f: features = pickle.load(f)
    return model, scaler, edu_enc, features

model, scaler, edu_enc, FEATURES = load_artifacts()

SCALE_COLS = [
    "years_experience","num_past_companies","num_certifications",
    "employment_gap_months","skill_score","communication_score",
    "problem_solving_score","cultural_fit_score","num_interviews_given",
    "avg_skill_score","experience_per_company","cert_per_year"
]

# ── Header ────────────────────────────────────────────────────
st.title("🧠 Smart Hiring Risk Predictor")
st.markdown("##### Predict whether a candidate will clear all interview rounds — powered by XGBoost + SHAP")
st.markdown(" preconceived notion")

# ── Sidebar — Candidate Input ─────────────────────────────────
st.sidebar.header("📋 Candidate Profile")
st.sidebar.markdown("Fill in the candidate details below:")

years_exp      = st.sidebar.slider("Years of Experience",       0,  15,  3)
num_companies  = st.sidebar.slider("Number of Past Companies",  1,   8,  2)
education      = st.sidebar.selectbox("Education Level",
                    ["High School","Bachelor","Master","PhD"])
num_certs      = st.sidebar.slider("Number of Certifications",  0,   6,  1)
emp_gap        = st.sidebar.slider("Employment Gap (months)",   0,  24,  2)
skill          = st.sidebar.slider("Skill Score",              20, 100, 65)
communication  = st.sidebar.slider("Communication Score",      20, 100, 60)
problem_solving= st.sidebar.slider("Problem Solving Score",    20, 100, 62)
cultural_fit   = st.sidebar.slider("Cultural Fit Score",       20, 100, 58)
num_interviews = st.sidebar.slider("Interviews Given Before",   1,  10,  3)
role_match     = st.sidebar.radio("Role Matches Profile?",     ["Yes","No"]) == "Yes"
referral       = st.sidebar.radio("Has Referral?",             ["Yes","No"]) == "Yes"

predict_btn = st.sidebar.button("🔮 Predict Now", use_container_width=True)

# ── Feature Engineering ───────────────────────────────────────
def build_features(ye, nc, edu, certs, gap, sk, cm, ps, cf, ni, rm, ref):
    edu_encoded       = edu_enc.transform([[edu]])[0][0]
    avg_skill         = (sk + cm + ps + cf) / 4
    exp_per_company   = round(ye / max(nc, 1), 2)
    cert_per_year     = round(certs / (ye + 1), 3)
    has_gap           = int(gap > 6)
    high_performer    = int(sk > 75 and ps > 75)

    row = pd.DataFrame([{
        "years_experience":      ye,
        "num_past_companies":    nc,
        "num_certifications":    certs,
        "employment_gap_months": gap,
        "skill_score":           sk,
        "communication_score":   cm,
        "problem_solving_score": ps,
        "cultural_fit_score":    cf,
        "num_interviews_given":  ni,
        "applied_role_match":    int(rm),
        "referral":              int(ref),
        "education_encoded":     edu_encoded,
        "avg_skill_score":       avg_skill,
        "experience_per_company":exp_per_company,
        "cert_per_year":         cert_per_year,
        "has_gap":               has_gap,
        "high_performer":        high_performer,
    }])
    row[SCALE_COLS] = scaler.transform(row[SCALE_COLS])
    return row[FEATURES]

# ── Main Panel ────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

if predict_btn:
    X_input = build_features(
        years_exp, num_companies, education, num_certs,
        emp_gap, skill, communication, problem_solving,
        cultural_fit, num_interviews, role_match, referral
    )

    prediction  = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]
    risk_score  = round((1 - probability) * 100, 1)

    # ── Result Card ───────────────────────────────────────────
    st.markdown("## 🎯 Prediction Result")

    if prediction == 1:
        st.success(f"## ✅ LIKELY TO CLEAR ALL ROUNDS")
        st.markdown(f"**Confidence:** {probability*100:.1f}%  |  **Risk Score:** {risk_score}/100 (Low Risk)")
    else:
        st.error(f"## ❌ AT RISK — MAY NOT CLEAR ALL ROUNDS")
        st.markdown(f"**Confidence:** {(1-probability)*100:.1f}%  |  **Risk Score:** {risk_score}/100 (High Risk)")

    st.markdown(" preconceived notion")

    # ── Metrics Row ───────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clearing Probability", f"{probability*100:.1f}%")
    col2.metric("Risk Score",           f"{risk_score}/100")
    col3.metric("Education",            education)
    col4.metric("Avg Skill Score",      f"{(skill+communication+problem_solving+cultural_fit)/4:.1f}")

    st.markdown(" preconceived notion")

    # ── SHAP Explanation ──────────────────────────────────────
    st.markdown("### 🔍 Why this prediction? (SHAP Explanation)")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)
    shap_series = pd.Series(shap_values[0], index=FEATURES).sort_values(key=abs, ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(9, 4))
    colors  = ["#1D9E75" if v > 0 else "#E24B4A" for v in shap_series.values]
    ax.barh(shap_series.index[::-1], shap_series.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on prediction)")
    ax.set_title("Top 10 Features Driving This Prediction")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("> 🟢 **Green bars** push toward *Cleared* &nbsp;&nbsp; 🔴 **Red bars** push toward *Not Cleared*")

    st.markdown(" preconceived notion")

    # ── Candidate Summary Table ───────────────────────────────
    st.markdown("### 📋 Candidate Summary")
    summary = pd.DataFrame({
        "Field": ["Experience","Companies","Education","Certifications",
                  "Emp. Gap","Skill","Communication","Problem Solving",
                  "Cultural Fit","Role Match","Referral"],
        "Value": [f"{years_exp} yrs", num_companies, education, num_certs,
                  f"{emp_gap} months", f"{skill}/100", f"{communication}/100",
                  f"{problem_solving}/100", f"{cultural_fit}/100",
                  "Yes" if role_match else "No",
                  "Yes" if referral else "No"]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

else:
    # ── Landing State ─────────────────────────────────────────
    st.markdown("## 👈 Fill in the candidate profile and click **Predict Now**")
    st.markdown("")

    c1, c2, c3 = st.columns(3)
    c1.info("**🤖 Model**\n\nXGBoost tuned with RandomizedSearchCV over 40 parameter combinations")
    c2.info("**📊 Training**\n\n1,500 candidates · SMOTE balanced · 5-fold cross validation")
    c3.info("**🔍 Explainability**\n\nSHAP values explain every individual prediction in real time")

    st.markdown(" preconceived notion")
    st.markdown("### 🗓️ Built in 5 Days")
    days = {
        "Day 1 ✅": "Problem setup + EDA",
        "Day 2 ✅": "Feature engineering + preprocessing",
        "Day 3 ✅": "Model training + comparison (6 models)",
        "Day 4 ✅": "Hyperparameter tuning + SHAP",
        "Day 5 ✅": "Streamlit app + deployment",
    }
    for day, desc in days.items():
        st.markdown(f"- **{day}** — {desc}")
