from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn

# FORCE SAME MLflow LOCATION
mlflow.set_tracking_uri("file:///C:/AI-ML-Portfolio/mlruns")

# --------------------------------------------------
# LOAD MODEL + PREPROCESSOR
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MODEL_URI = "models:/LoanDefaultModel/1"
model = mlflow.sklearn.load_model(MODEL_URI)

preprocessor = joblib.load(os.path.join(BASE_DIR, "models", "preprocessor.pkl"))

# --------------------------------------------------
# APP
# --------------------------------------------------

app = FastAPI(title="Loan Default Prediction API")

@app.get("/")
def home():
    return {"message": "API is running"}

# --------------------------------------------------
# PREDICT
# --------------------------------------------------

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    # 🔥 APPLY SAME PREPROCESSING
    df_processed = preprocessor.transform(df)

    # Predict
    prediction = model.predict(df_processed)[0]

    try:
        proba = model.predict_proba(df_processed)[0][1]
    except:
        proba = None

    return {
        "prediction": int(prediction),
        "probability": float(proba) if proba else None
    }