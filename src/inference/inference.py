# src/inference/inference.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# -------------------------
# Load model
# -------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "artifacts" / "churn_model1.pkl"
model = joblib.load(MODEL_PATH)

# -------------------------
# Feature names and defaults
# -------------------------
FEATURE_NAMES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

DEFAULTS = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 0,
    'PhoneService': 'No',
    'MultipleLines': 'No',
    'InternetService': 'No',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 0.0,
    'TotalCharges': 0.0
}

# -------------------------
# Predict function
# -------------------------
def predict_churn(data) -> str:
    """
    Predict churn from input data.

    Accepts dict, list, numpy array, or pandas DataFrame.
    Handles 1D, 2D, and accidental 3D inputs safely.
    """
    # Handle None input
    if data is None:
        return "Error: No input provided."

    # If it's already a DataFrame, make a copy
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        # Convert to NumPy array
        data = np.asarray(data)

        # Remove extra dimensions: 1D → (1, n_features), 3D → (n_samples, n_features)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim == 3:
            # Example: (1,1,19) → (1,19)
            data = data.reshape(data.shape[0], data.shape[-1])

        # Finally create DataFrame
        df = pd.DataFrame(data, columns=FEATURE_NAMES)

    # Fill missing columns
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = DEFAULTS[col]

    # Ensure correct column order
    df = df[FEATURE_NAMES]

    # Predict
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    churn_text = "Possible Churn" if pred == 1 else "Churn Not Possible"
    return f"{churn_text} with a probability of {proba:.2f}"