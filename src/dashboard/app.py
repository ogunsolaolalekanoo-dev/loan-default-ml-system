import streamlit as st
import requests

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

st.set_page_config(page_title="Loan Default Predictor", layout="centered")

st.title("💳 Loan Default Prediction Dashboard")

st.write("Enter applicant details to assess loan default risk.")

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------

age = st.slider("Age", 18, 100, 30)
income = st.number_input("Income", value=50000)
loan_amount = st.number_input("Loan Amount", value=10000)
credit_score = st.slider("Credit Score", 300, 850, 650)
months_employed = st.slider("Months Employed", 0, 120, 12)
num_credit_lines = st.slider("Number of Credit Lines", 0, 10, 2)
interest_rate = st.slider("Interest Rate (%)", 1.0, 30.0, 10.0)
loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
dti = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)

education = st.selectbox("Education", ["High School", "Bachelor's", "Master's"])
employment = st.selectbox("Employment Type", ["Full-time", "Part-time", "Unemployed"])
marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
mortgage = st.selectbox("Has Mortgage", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
purpose = st.selectbox("Loan Purpose", ["Personal", "Auto", "Business", "Home"])
cosigner = st.selectbox("Has Co-Signer", ["Yes", "No"])

# --------------------------------------------------
# THRESHOLD SLIDER (🔥 KEY FEATURE)
# --------------------------------------------------

threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5)

# --------------------------------------------------
# PREDICT BUTTON
# --------------------------------------------------

if st.button("Predict"):

    input_data = {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti,
        "Education": education,
        "EmploymentType": employment,
        "MaritalStatus": marital,
        "HasMortgage": mortgage,
        "HasDependents": dependents,
        "LoanPurpose": purpose,
        "HasCoSigner": cosigner
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=input_data
        )

        result = response.json()

        probability = result["probability"]

        # Apply threshold
        prediction = 1 if probability >= threshold else 0

        # --------------------------------------------------
        # OUTPUT
        # --------------------------------------------------

        st.subheader("Prediction Result")

        st.write(f"**Default Probability:** {probability:.2f}")

        if prediction == 1:
            st.error("⚠️ High Risk of Default")
        else:
            st.success("✅ Low Risk of Default")

    except Exception as e:
        st.error(f"Error: {e}")