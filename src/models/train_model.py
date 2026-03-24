import sys
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

# --------------------------------------------------
# FIX PATH
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

from src.features.preprocess import preprocess_data

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
X_train, X_test, y_train, y_test, preprocessor = preprocess_data()

# --------------------------------------------------
# CLASS IMBALANCE HANDLING
# --------------------------------------------------

print("\nOriginal Class Distribution:")
print(pd.Series(y_train).value_counts())

# SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_train).value_counts())

# scale_pos_weight for XGBoost
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

print("\nScale Pos Weight:", scale_pos_weight)

# --------------------------------------------------
# MODELS + PARAM GRIDS
# --------------------------------------------------

models = {

    "Logistic Regression": (
        LogisticRegression(max_iter=1000),
        {
            "C": [0.1, 1, 10]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(n_jobs=-1),
        {
            "n_estimators": [100],
            "max_depth": [None, 10]
        }
    ),

    "Gradient Boosting": (
        GradientBoostingClassifier(),
        {
            "n_estimators": [100],
            "learning_rate": [0.1]
        }
    ),

    "XGBoost": (
        XGBClassifier(
            eval_metric="logloss",
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight
        ),
        {
            "n_estimators": [100],
            "learning_rate": [0.1],
            "max_depth": [3, 5]
        }
    )
}

# --------------------------------------------------
# STRATIFIED K-FOLD
# --------------------------------------------------

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

results = []

# --------------------------------------------------
# TRAIN + GRID SEARCH
# --------------------------------------------------

for name, (model, params) in models.items():

    print(f"\n🚀 Training {name}...")

    grid = GridSearchCV(
        model,
        params,
        cv=skf,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    print(f"Best Params for {name}: {grid.best_params_}")

    # Predictions
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Metrics
    roc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    results.append({
        "Model": name,
        "ROC-AUC": roc,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn
    })

# --------------------------------------------------
# RESULTS TABLE
# --------------------------------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="ROC-AUC", ascending=False)

print("\n📊 MODEL COMPARISON:")
print(results_df)

# --------------------------------------------------
# PLOT
# --------------------------------------------------

os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)

plt.figure()
results_df.set_index("Model")[["ROC-AUC", "F1"]].plot(kind="bar")
plt.title("Model Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "reports", "model_comparison.png"))

print("\n📈 Chart saved")

# --------------------------------------------------
# BEST MODEL
# --------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]
print(f"\n🔥 Best Model: {best_model_name}")

best_model, params = models[best_model_name]

grid = GridSearchCV(best_model, params, cv=skf, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)

final_model = grid.best_estimator_

# --------------------------------------------------
# THRESHOLD TUNING
# --------------------------------------------------

probs = final_model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.1, 0.9, 0.05)

threshold_results = []

for t in thresholds:

    preds = (probs >= t).astype(int)

    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds)
    acc = accuracy_score(y_test, preds)

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    threshold_results.append({
        "Threshold": t,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn
    })

threshold_df = pd.DataFrame(threshold_results)

print("\n📊 THRESHOLD TUNING:")
print(threshold_df)

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

joblib.dump(final_model, os.path.join(BASE_DIR, "models", "best_model.pkl"))
joblib.dump(preprocessor, os.path.join(BASE_DIR, "models", "preprocessor.pkl"))

print("\n💾 Best model saved successfully")