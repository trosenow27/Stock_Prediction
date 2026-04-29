import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loan Default Prediction App", layout="wide")

st.title("Loan Default Prediction App")
st.write(
    "This app predicts whether a borrower is likely to default using a simplified "
    "machine learning model trained on five important loan-risk features."
)

# Load local model
model = joblib.load("model.joblib")

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

    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    st.subheader("Prediction")

    if prediction == 1:
        st.error("High Risk / Likely to Default")
    else:
        st.success("Low Risk / Not Likely to Default")

    st.write(f"Probability of default: {prediction_proba[1]:.2%}")
    st.write("Input used for prediction:")
    st.dataframe(input_df)

    # SHAP explanation for this single prediction
    st.subheader("SHAP Explanation")

    try:
        # The model is a pipeline: imputer -> scaler -> logistic regression
        transformed_input = model[:-1].transform(input_df)
        feature_names = input_df.columns.tolist()

        # Explain the final Logistic Regression model
        final_classifier = model.named_steps["model"]

        explainer = shap.LinearExplainer(final_classifier, transformed_input)
        shap_values = explainer(transformed_input)

        explanation = shap.Explanation(
            values=shap_values.values[0],
            base_values=shap_values.base_values[0],
            data=input_df.iloc[0].values,
            feature_names=feature_names
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(fig)

        st.caption(
            "The SHAP plot shows which features pushed this individual prediction toward "
            "higher or lower default risk."
        )

    except Exception as e:
        st.warning("SHAP plot could not be generated.")
        st.write(e)
