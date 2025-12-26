import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Risk Prediction App")
st.markdown(
    "Predict whether a customer is **High Risk** or **Low Risk** using a **Logistic Regression** model."
)

# ----------------------------
# Load Model & Features
# ----------------------------
MODEL_PATH = "models/credit_risk_logreg.pkl"
FEATURE_PATH = "models/model_features.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_PATH):
    st.error("❌ Model files not found. Please check the `models/` folder.")
    st.stop()

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURE_PATH)

# ----------------------------
# Sidebar – Model Info
# ----------------------------
st.sidebar.header("📘 Model Info")
st.sidebar.write("**Algorithm:** Logistic Regression")
st.sidebar.write("**Imbalance Handling:** class_weight = balanced")
st.sidebar.write("**Threshold:** 0.5")

# ----------------------------
# User Input
# ----------------------------
st.subheader("🧾 Enter Customer Details")

user_input = {}

for feature in features:
    user_input[feature] = st.number_input(
        label=feature,
        value=0.0,
        step=1.0
    )

input_df = pd.DataFrame([user_input])

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔍 Predict Credit Risk"):
    prob = model.predict_proba(input_df)[0][1]

    st.metric("📊 Default Probability", f"{prob * 100:.2f}%")

    if prob >= 0.5:
        st.error("🚨 High Credit Risk")
    else:
        st.success("✅ Low Credit Risk")

# ----------------------------
# Model Explainability
# ----------------------------
st.divider()
st.subheader("🧠 Model Explainability")

coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_[0],
})

coef_df["Odds Ratio"] = np.exp(coef_df["Coefficient"])
coef_df = coef_df.sort_values("Odds Ratio", ascending=False)

st.markdown(
    """
**How to interpret this:**
- Odds Ratio **> 1** → increases default risk  
- Odds Ratio **< 1** → decreases default risk  
"""
)

st.dataframe(coef_df, use_container_width=True)

# ----------------------------
# ROC Curve (Demo – Model Level)
# ----------------------------
st.divider()
st.subheader("📈 Model Performance (ROC Curve)")

st.info(
    "ROC curve shown using **synthetic probabilities** for demonstration.\n"
    "In production, compute using hold-out test data."
)

# Generate demo probabilities (safe for Streamlit Cloud)
np.random.seed(42)
y_true = np.random.randint(0, 2, 300)
y_scores = np.random.rand(300)

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend(loc="lower right")

st.pyplot(fig)

# ----------------------------
# Footer
# ----------------------------
st.divider()
st.caption(
    "Built with ❤️ using Streamlit | Logistic Regression | Credit Risk Modeling"
)
