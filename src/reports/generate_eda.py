import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# PATH SETUP (SAME AS YOUR MODEL SCRIPT)
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "loan_default_clean.csv")
REPORT_PATH = os.path.join(BASE_DIR, "reports")

# Create reports folder (same place as model chart)
os.makedirs(REPORT_PATH, exist_ok=True)

sns.set_style("whitegrid")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# 1. TARGET DISTRIBUTION
# --------------------------------------------------

plt.figure()
sns.countplot(x='Default', data=df)
plt.title("Default Distribution")
plt.savefig(os.path.join(REPORT_PATH, "target_distribution.png"))
plt.close()

# --------------------------------------------------
# 2. NUMERICAL DISTRIBUTIONS
# --------------------------------------------------

num_cols = [
    'Age', 'Income', 'LoanAmount', 'CreditScore',
    'MonthsEmployed', 'NumCreditLines',
    'InterestRate', 'LoanTerm', 'DTIRatio'
]

df[num_cols].hist(bins=20, figsize=(12,10))
plt.suptitle("Numerical Distributions")
plt.savefig(os.path.join(REPORT_PATH, "numerical_distributions.png"))
plt.close()

# --------------------------------------------------
# 3. CORRELATION HEATMAP
# --------------------------------------------------

plt.figure(figsize=(10,8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(REPORT_PATH, "correlation_heatmap.png"))
plt.close()

# --------------------------------------------------
# 4. BOXPLOTS
# --------------------------------------------------

features = ['CreditScore', 'Income', 'LoanAmount', 'DTIRatio']

for feature in features:
    plt.figure()
    sns.boxplot(x='Default', y=feature, data=df)
    plt.title(f"{feature} vs Default")
    plt.savefig(os.path.join(REPORT_PATH, f"{feature}_vs_default.png"))
    plt.close()

# --------------------------------------------------
# 5. CATEGORICAL ANALYSIS
# --------------------------------------------------

cat_cols = ['Education', 'EmploymentType', 'LoanPurpose']

for col in cat_cols:
    plt.figure(figsize=(8,5))
    sns.countplot(x=col, hue='Default', data=df)
    plt.xticks(rotation=45)
    plt.title(f"{col} vs Default")
    plt.savefig(os.path.join(REPORT_PATH, f"{col}_vs_default.png"))
    plt.close()

# --------------------------------------------------
# 6. DEFAULT RATE (VERY IMPORTANT)
# --------------------------------------------------

for col in cat_cols:
    rate = df.groupby(col)['Default'].mean().sort_values()

    plt.figure()
    rate.plot(kind='bar')
    plt.title(f"Default Rate by {col}")
    plt.ylabel("Default Rate")
    plt.savefig(os.path.join(REPORT_PATH, f"default_rate_{col}.png"))
    plt.close()

print("\n✅ EDA saved in:", REPORT_PATH)


