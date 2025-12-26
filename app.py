

import streamlit as st

import numpy as np
import pandas as pd
import joblib

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc





import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
st.divider()
st.subheader("📈 Model Performance (ROC Curve)")

# Load stored test predictions (temporary demo approach)
# In production, this would be precomputed and stored
y_true = joblib.load("models/y_test.pkl") if "y_test.pkl" in os.listdir("models") else None
y_scores = joblib.load("models/test_pred.pkl") if "test_pred.pkl" in os.listdir("models") else None

if y_true is not None and y_scores is not None:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    st.pyplot(fig)
else:
    st.info("ROC curve available in model evaluation notebook.")



import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Risk Prediction App")
st.write(
    "Predict whether a customer is **High Risk** or **Low Risk** "
    "using a Logistic Regression credit risk model."
)

# -------------------------------
# Load model safely
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_risk_logreg.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "model_features.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_PATH):
    st.error("❌ Model files not found. Please train and save the model first.")
    st.stop()

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURE_PATH)

# -------------------------------
# User Inputs
# -------------------------------
st.subheader("Enter Customer Details")

user_input = {}

for feature in features:
    user_input[feature] = st.number_input(
        label=feature,
        value=0.0,
        step=1.0
    )

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Credit Risk"):
    input_df = pd.DataFrame([user_input])
    input_df = input_df[features]  # ensure correct order

    prob = model.predict_proba(input_df)[0][1]

    st.metric("Default Probability", f"{prob:.2%}")

    if prob >= 0.5:
        st.error("🚨 High Credit Risk")
    else:
        st.success("✅ Low Credit Risk")
