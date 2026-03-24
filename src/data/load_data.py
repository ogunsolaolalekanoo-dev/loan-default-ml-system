import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    df = load_data("A1-loan-default-pipeline/data/raw/loan_default.csv")

    print("Shape:", df.shape)
    print("\nColumns:\n", df.columns)
    print("\nSample:\n", df.head())