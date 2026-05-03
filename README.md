# 🧠 Smart Hiring Risk Predictor

> Predict whether a candidate will clear all interview rounds — built from scratch in 5 days.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## 🎯 What It Does
Input a candidate's profile → get an instant prediction on whether they'll clear all interview rounds, with a full SHAP explanation of *why*.

## 🗓️ Built in 5 Days
| Day | Task |
|-----|------|
| Day 1 | Problem setup · Synthetic dataset · Full EDA |
| Day 2 | Feature engineering · Preprocessing · SMOTE |
| Day 3 | Trained 6 models · Compared metrics · Selected best |
| Day 4 | Hyperparameter tuning · SHAP explainability · Fairness check |
| Day 5 | Streamlit app · Deployment |

## 🤖 Tech Stack
- **Model:** XGBoost (tuned with RandomizedSearchCV)
- **Explainability:** SHAP (TreeExplainer)
- **Balancing:** SMOTE
- **App:** Streamlit
- **Notebook:** Google Colab

## 📁 Project Structure

## 🚀 Run Locally
```bash
git clone https://github.com/gnaneshwar2380/smart-hiring-risk-predictor.git
cd smart-hiring-risk-predictor
pip install -r requirements.txt
streamlit run app/app.py
```

## 👤 Author
**Gnaneshwar** - Built as a 5-day ML project from scratch
