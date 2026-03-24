import pandas as pd
import os

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Build paths
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "loan_default.csv")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "loan_default_clean.csv")


def clean_data():

    print("Looking for file at:", RAW_PATH)

    df = pd.read_csv(RAW_PATH)

    # Drop ID
    df = df.drop(columns=["LoanID"])

    # Remove duplicates
    df = df.drop_duplicates()

    # Convert numeric safely
    df["DTIRatio"] = pd.to_numeric(df["DTIRatio"], errors="coerce")

    # Drop missing values
    df = df.dropna()

    print("Clean shape:", df.shape)

    # Create processed folder if it doesn't exist
    os.makedirs(os.path.join(BASE_DIR, "data", "processed"), exist_ok=True)

    # Save clean data
    df.to_csv(PROCESSED_PATH, index=False)

    print("Saved clean data to:", PROCESSED_PATH)


if __name__ == "__main__":
    clean_data()