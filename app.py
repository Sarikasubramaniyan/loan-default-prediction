import streamlit as st
import pandas as pd
import pickle

# Load trained model and column names
model = pickle.load(open("loan_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))  # ✔️ This is already present

st.set_page_config(page_title="Loan Default Predictor", page_icon="💳")
st.title("💳 Loan Default Prediction App")
st.write("Fill the form to predict if the person is likely to default on their loan.")

# 🔽 Input form
def user_input_features():
    LoanAmount = st.number_input("Loan Amount", value=5000)
    Age = st.number_input("Age", value=30)
    Income = st.number_input("Monthly Income", value=25000)
    CreditScore = st.slider("Credit Score", 300, 850, value=650)
    MonthsEmployed = st.number_input("Months Employed", value=36)
    NumCreditLines = st.number_input("Number of Credit Lines", value=5)
    InterestRate = st.slider("Interest Rate (%)", 0.0, 25.0, step=0.1, value=10.5)
    LoanTerm = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
    DTIRatio = st.slider("DTI Ratio", 0.0, 1.0, step=0.01, value=0.3)

    Education = st.selectbox("Education Level", ["High School", "Bachelors", "Masters", "PhD"])
    EmploymentType = st.selectbox("Employment Type", ["Salaried", "Self-employed", "Unemployed"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    HasMortgage = st.selectbox("Has Mortgage?", ["Yes", "No"])
    HasDependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    LoanPurpose = st.selectbox("Loan Purpose", ["Home", "Auto", "Education", "Personal", "Business"])
    HasCoSigner = st.selectbox("Has Co-Signer?", ["Yes", "No"])

    # Create dictionary
    data = {
        "LoanAmount": LoanAmount,
        "Age": Age,
        "Income": Income,
        "CreditScore": CreditScore,
        "MonthsEmployed": MonthsEmployed,
        "NumCreditLines": NumCreditLines,
        "InterestRate": InterestRate,
        "LoanTerm": LoanTerm,
        "DTIRatio": DTIRatio,
        "Education": Education,
        "EmploymentType": EmploymentType,
        "MaritalStatus": MaritalStatus,
        "HasMortgage": HasMortgage,
        "HasDependents": HasDependents,
        "LoanPurpose": LoanPurpose,
        "HasCoSigner": HasCoSigner
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 🧠 Encode categorical features (just like training)
categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage',
                    'HasDependents', 'LoanPurpose', 'HasCoSigner']

for col in categorical_cols:
    input_df[col] = pd.factorize(input_df[col])[0]

# 🛡️ Reorder columns to match training data
input_df = input_df[model_columns]

# 🔮 Make prediction
if st.button("🔍 Predict Loan Default Risk"):
    prediction = model.predict(input_df)
    result = "⚠️ High Risk of Default" if prediction[0] == 1 else "✅ Low Risk of Default"
    st.success(f"Prediction: {result}")
