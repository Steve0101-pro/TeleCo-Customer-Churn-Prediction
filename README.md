# 🔮 TeleCo Customer Churn Prediction(End-to-End ML Project)

A Machine Learning system that predicts whether a telecom customer is likely to churn (leave the service) based on their account and service usage data.

The project demonstrates a complete ML pipeline, including model training, hyperparameter tuning, experiment tracking, API deployment, and an interactive web interface.

---

# Project Overview

Customer churn is a major challenge for telecom companies. Predicting churn allows companies to:

- Identify customers likely to leave

- Take preventive retention actions

- Improve customer satisfaction

- Increase revenue

This project builds and compares multiple machine learning models to predict churn and deploys the best-performing model as a production-ready application.

---

# Features

    -Data preprocessing and feature engineering

    -Multiple machine learning models

    -Hyperparameter tuning using Optuna

    Experiment tracking with MLflow

    Model serving using FastAPI

    Interactive web app using Streamlit

    Clean and modular ML pipeline

---

# 🖥️ Tech Stack

**Programming**

- Python

**Machine Learning**

- Scikit-learn

- XGBoost

**Optimization**

- Optuna

**Experiment Tracking**

- MLflow

**Backend API**

- FastAPI

**Frontend App**

- Streamlit

**Data Processing**

- Pandas

- NumPy

Machine Learning Models Used

The following models were trained and evaluated:

- **Logistic Regression**

- **Linear Regression**

- **Random Forest**

- **XGBoost**

The best model was selected based on evaluation metrics and tuned using Optuna.

---

# clone project

```bash

git clone https://github.com/your-username/teleco-churn-prediction.git
cd teleco-churn-prediction

```

# 📁 Project Structure

```text
TeleCo-Churn-Prediction
TeleCo-Churn-Prediction
│
├── data
│   └── telecom_churn.csv
│
├── notebooks
│   └── EDA_and_model_training.ipynb
│
├── artifacts
│   └── churn_model1.pkl
│
├── src
│   │
│   ├── app
│   │   ├── app.py        # FastAPI backend
│   │   └── main.py       # Streamlit frontend
│   │
│   └── inference.py      # Prediction logic
│
├── mlruns                # MLflow experiment tracking
│
├── requirements.txt
│
└── README.md
```

# run fastapi

```bash
uvicorn src.app.app:app --reload
```

# run streamlit

```bash
streamlit run src/app/main.py
```
