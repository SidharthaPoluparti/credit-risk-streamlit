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
