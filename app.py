import joblib
import pandas as pd
from credit_risk_model.config import MODELS_DIR, DATA_PROCESSED
import streamlit as st
import numpy as np


model = joblib.load(f"{MODELS_DIR}/LightGBM.joblib")
feature_list = joblib.load(f"{DATA_PROCESSED}/feature_list.joblib")
# user input
user_input = {feat: 0.0 for feat in feature_list}
df = pd.DataFrame([user_input])

# enforce correct order
df = df[feature_list]

prob = model.predict_proba(df)[0, 1]

st.write(f"Probability of default: {prob}")

st.title("🏦 Credit Risk Scoring App")

st.sidebar.header("Applicant Information")

input_data = {}
for feat in feature_list:
    input_data[feat] = st.sidebar.number_input(feat, value=0.0)

df = pd.DataFrame([input_data])

if st.button("Predict Risk"):
    prob = model.predict_proba(df)[0, 1]

    if prob < 0.2:
        risk = "Low Risk"
    elif prob < 0.4:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    st.metric("Default Probability", f"{prob:.2%}")
    st.success(f"Risk Category: {risk}")
