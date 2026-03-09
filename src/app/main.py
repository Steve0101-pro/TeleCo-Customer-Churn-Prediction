

import streamlit as st
import pandas as pd
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.inference.inference import predict_churn

st.set_page_config(page_title="Telco Churn Predictor", layout="centered")
st.title("🔮 Telco Customer Churn Predictor")
st.write("Fill in customer details to predict churn risk.")

# -------------------------
# Define all fields
# -------------------------
form_fields = {
    "gender": ["Male", "Female"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "tenure": 0,
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    "MonthlyCharges": 0.0,
    "TotalCharges": 0.0
}

# -------------------------
# Streamlit form
# -------------------------
with st.form("churn_form"):
    user_input = {}
    for field, options in form_fields.items():
        if isinstance(options, list):
            user_input[field] = st.selectbox(field, options)
        else:
            user_input[field] = st.number_input(field, value=options)
    submitted = st.form_submit_button("Predict Churn")

# -------------------------
# On submission
# -------------------------
if submitted:
    # Make sure all values are scalar and no None
    safe_input = {}
    for k, v in user_input.items():
        if v is None:
            # Use default first option
            if isinstance(form_fields[k], list):
                safe_input[k] = form_fields[k][0]
            else:
                safe_input[k] = form_fields[k]
        elif isinstance(v, list):
            safe_input[k] = v[0]
        else:
            safe_input[k] = v

    # Build 2D DataFrame
    input_df = pd.DataFrame([safe_input])
    st.write("### Input Data")
    st.dataframe(input_df)
    

    print("DEBUG: type(input_df) =", type(input_df))
    print("DEBUG: input_df.shape =", input_df.shape)

    # Call predict_churn
    result = predict_churn(input_df)

    st.subheader("Prediction Result")
    st.write(f"**{result}**")