# 💳 Credit Risk Prediction System

An end-to-end **Credit Risk Modeling** project that predicts the probability of a customer defaulting within the next 2 years.  
Built with a strong focus on **industry-grade evaluation**, **model explainability**, and **deployment readiness**.

---

## 🚀 Project Overview

This project covers the **full machine learning lifecycle**:
- Data cleaning & feature engineering
- Class imbalance handling
- Model selection & validation
- Business-driven threshold tuning
- Deployment using Streamlit

The final model is a **Logistic Regression** classifier validated using **ROC-AUC, KS statistic, and decile analysis** — metrics commonly used in real-world credit risk systems.

---

## 📊 Model Performance (Test Set)

| Metric | Value |
|------|------|
| ROC-AUC | ~0.85 |
| KS Statistic | ~0.54 |
| Default Rate (Top Decile) | ~35% |
| Default Rate (Bottom Decile) | < 1% |

These results indicate **strong separation** between defaulters and non-defaulters.

---

## 🧠 Why Logistic Regression?

- Highly interpretable (important for regulators & stakeholders)
- Stable under class imbalance
- Produces calibrated probabilities
- Widely used in banking and financial institutions

Class imbalance was handled using **class-weighted loss** instead of SMOTE, as SMOTE reduced ranking performance.

---

## 🧮 Model Explainability

Each feature coefficient represents its impact on default risk:

- **Positive coefficient** → Increases default probability
- **Negative coefficient** → Decreases default probability

Odds ratios were used to explain how a unit increase in a feature changes default risk.

---

## 🎯 Business Decision Threshold

Instead of using a default 0.5 threshold, a **0.4 probability cutoff** was selected to:
- Improve recall of high-risk customers
- Balance false positives and false negatives
- Reflect real-world credit approval trade-offs

---

## 🖥️ Live Demo (Streamlit)

The model is deployed as an interactive web application where users can:
- Input customer details
- View default probability
- Receive a credit risk decision (Low / High risk)

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## 📌 Key Learnings

- ROC-AUC is more reliable than accuracy for imbalanced data
- SMOTE is not always beneficial for credit risk problems
- Threshold tuning is a business decision, not just a technical one
- Simpler models can outperform complex ones when validated correctly

---

## 👤 Author

Built as an **industry-ready portfolio project** focused on credit risk modeling and deployment.
