import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Get project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Path to cleaned data
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "loan_default_clean.csv")


def preprocess_data():

    print("Loading clean data from:", CLEAN_DATA_PATH)

    df = pd.read_csv(CLEAN_DATA_PATH)

    # Separate features and target
    X = df.drop("Default", axis=1)
    y = df["Default"]

    # Identify feature types
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print("\nNumeric Features:", numeric_features)
    print("\nCategorical Features:", categorical_features)

    # Build preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # Fit and transform
    X_transformed = preprocessor.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    preprocess_data()