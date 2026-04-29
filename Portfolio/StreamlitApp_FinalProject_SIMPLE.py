import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Default Prediction App", layout="wide")

st.title("Loan Default Prediction App")
st.write(
    "This app predicts whether a borrower is likely to default using a simplified "
    "machine learning model trained on five important loan-risk features."
)

# Load local model
import os

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "model.joblib")

model = joblib.load(model_path)

with st.form("prediction_form"):
    st.subheader("Borrower / Loan Inputs")

    col1, col2 = st.columns(2)

    with col1:
        int_rate = st.number_input(
            "Interest Rate",
            min_value=0.0,
            max_value=40.0,
            value=13.0,
            step=0.1
        )

        fico_range_low = st.number_input(
            "FICO Range Low",
            min_value=300.0,
            max_value=850.0,
            value=680.0,
            step=5.0
        )

        loan_income_ratio = st.number_input(
            "Loan-to-Income Ratio",
            min_value=0.0,
            max_value=5.0,
            value=0.30,
            step=0.01
        )

    with col2:
        grade_encoded = st.number_input(
            "Grade Encoded",
            min_value=0.0,
            max_value=6.0,
            value=2.0,
            step=1.0,
            help="A=0, B=1, C=2, D=3, E=4, F=5, G=6"
        )

        dti = st.number_input(
            "Debt-to-Income Ratio",
            min_value=0.0,
            max_value=50.0,
            value=18.0,
            step=0.5
        )

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    input_df = pd.DataFrame([{
        "int_rate": int_rate,
        "grade_encoded": grade_encoded,
        "fico_range_low": fico_range_low,
        "dti": dti,
        "loan_income_ratio": loan_income_ratio
    }])

   # Make sure input columns match the model's expected training columns
try:
    expected_features = list(model.named_steps["imputer"].feature_names_in_)
    input_df = input_df.reindex(columns=expected_features, fill_value=0)
except Exception:
    pass

prediction = model.predict(input_df)[0]

    st.subheader("Prediction")

    if prediction == 1:
        st.error("High Risk / Likely to Default")
    else:
        st.success("Low Risk / Not Likely to Default")

    st.write("Input used for prediction:")
    st.dataframe(input_df)
