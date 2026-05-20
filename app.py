import streamlit as st
import numpy as np
import pickle

# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="wide"
)

# =========================
# LOAD FILES
# =========================

model = pickle.load(open("model_r.pkl", "rb"))

scaler = pickle.load(open("scaler.pkl", "rb"))

encoder = pickle.load(open("encoder.pkl", "rb"))



st.title(" Credit Risk Prediction System")

st.markdown("### Enter Customer Details")

st.write("")


col1, col2, col3 = st.columns(3)



with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    income = st.number_input(
        "Income",
        min_value=0.0,
        value=50000.0
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=700
    )



with col2:

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=100000.0
    )

    loan_term = st.number_input(
        "Loan Term (Months)",
        min_value=1,
        value=24
    )

    existing_loans = st.number_input(
        "Existing Loans",
        min_value=0,
        value=1
    )



with col3:

    debt_income_ratio = st.number_input(
        "Debt Income Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.30
    )

    previous_defaults = st.number_input(
        "Previous Defaults",
        min_value=0,
        value=0
    )

    bank_account_age = st.number_input(
        "Bank Account Age",
        min_value=0,
        value=5
    )

st.write("")
st.write("")


if st.button("Predict Risk"):

    # Input Data
    input_data = np.array([[
        age,
        income,
        credit_score,
        loan_amount,
        loan_term,
        existing_loans,
        debt_income_ratio,
        previous_defaults,
        bank_account_age
    ]])

    # Scale Data
    input_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_data)

    # Decode Prediction
    result = encoder.inverse_transform(prediction)

    st.write("## Prediction Result")

    # Display Result
    if result[0] == "High":

        st.error(" High Risk Customer")

    elif result[0] == "Low":

        st.success(" Low Risk Customer")

    else:

        st.warning(" Medium Risk Customer")
st.markdown("---")
st.caption("Machine Learning Credit Risk Prediction App")