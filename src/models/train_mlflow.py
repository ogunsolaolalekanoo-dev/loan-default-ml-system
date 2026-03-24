import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# FORCE MLflow PATH (CRITICAL FIX)
# --------------------------------------------------
BASE_TRACKING_DIR = "C:/AI-ML-Portfolio/mlruns"
mlflow.set_tracking_uri(f"file:///{BASE_TRACKING_DIR}")
print("MLflow tracking URI:", mlflow.get_tracking_uri())

mlflow.set_experiment("Loan_Default_Experiment")

# --------------------------------------------------
# IMPORTS
# --------------------------------------------------
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# --------------------------------------------------
# PATH FIX
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

from src.features.preprocess import preprocess_data

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
X_train, X_test, y_train, y_test, preprocessor = preprocess_data()

# SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# --------------------------------------------------
# MODELS
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(eval_metric="logloss", n_jobs=-1)
}

best_roc = 0
best_model = None
best_name = ""

# --------------------------------------------------
# TRAIN + LOG
# --------------------------------------------------
for name, model in models.items():

    with mlflow.start_run(run_name=name):

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        roc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("roc_auc", roc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log parameter
        mlflow.log_param("model_type", name)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        cm_path = f"{name}_cm.png"
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(cm_path)

        # Log model
        mlflow.sklearn.log_model(model, name="model")

        print(f"{name} ROC-AUC: {roc:.4f}")

        # Track best model
        if roc > best_roc:
            best_roc = roc
            best_model = model
            best_name = name

# --------------------------------------------------
# REGISTER BEST MODEL
# --------------------------------------------------
print(f"\nBest Model: {best_name}")

with mlflow.start_run(run_name="Best_Model"):

    mlflow.sklearn.log_model(
        best_model,
        name="best_model",
        registered_model_name="LoanDefaultModel"
    )