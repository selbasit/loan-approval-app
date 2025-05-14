import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and transformer
model = pickle.load(open("loan_model.pkl", "rb"))
transformer = pickle.load(open("column_transformer.pkl", "rb"))

# Title
st.title("Loan Approval Prediction App")
st.write("Provide applicant and loan details to predict loan approval status.")

# Input fields
age = st.number_input("Applicant Age", 18, 100, 30)
income = st.number_input("Annual Income (USD)", 1000, 1000000, 50000, step=1000)
emp_length = st.number_input("Employment Length (years)", 0, 50, 5)
home = st.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"])
loan_amt = st.number_input("Loan Amount (USD)", 500, 50000, 10000, step=500)
int_rate = st.number_input("Interest Rate (%)", 0.0, 40.0, 10.0, step=0.1)
loan_intent = st.selectbox("Loan Purpose", ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"])
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
cred_hist = st.number_input("Credit History Length (years)", 0, 50, 5)
default = st.selectbox("Default on File", ["N", "Y"])

# Predict button
if st.button("Predict"):
    loan_percent_income = loan_amt / income if income > 0 else 0

    input_dict = {
        'person_age': [age],
        'person_income': [income],
        'person_emp_length': [emp_length],
        'loan_amnt': [loan_amt],
        'loan_int_rate': [int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_cred_hist_length': [cred_hist],
        'person_home_ownership': [home],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade],
        'cb_person_default_on_file': [default]
    }

    input_df = pd.DataFrame(input_dict)

    # Apply preprocessing
    input_transformed = transformer.transform(input_df)

    # Predict
    prediction = model.predict(input_transformed)[0]

    # Result
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
