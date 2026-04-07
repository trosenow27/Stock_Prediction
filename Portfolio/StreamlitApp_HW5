import os
import sys
import warnings
import tempfile
import tarfile

import joblib
import boto3
import sagemaker
import shap
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer
from sklearn.pipeline import Pipeline as SklearnPipeline

warnings.filterwarnings("ignore")
st.set_page_config(page_title="S&P 500 Return Prediction", layout="wide")

# =========================
# EDIT THESE TO MATCH YOUR FILES
# =========================
MODEL_INFO = {
    "endpoint": st.secrets["aws_credentials"]["AWS_ENDPOINT"],
    "pipeline_tar": "finalized_pca_model.tar.gz",   # change if your tar.gz has a different name
    "explainer_file": "explainer_pca.shap",         # change if you used explainer_kpca.shap
    "s3_pipeline_prefix": "sklearn-pipeline-deployment",
    "s3_explainer_prefix": "explainer"
}

AWS_REGION = "us-east-1"

# =========================
# AWS SECRETS
# =========================
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = MODEL_INFO["endpoint"]


# =========================
# AWS SESSION
# =========================
@st.cache_resource
def get_boto_session():
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name=AWS_REGION
    )


@st.cache_resource
def get_sm_session():
    boto_session = get_boto_session()
    return sagemaker.Session(boto_session=boto_session)


# =========================
# LOAD PIPELINE FROM S3
# =========================
@st.cache_resource
def load_pipeline_from_s3():
    session = get_boto_session()
    s3_client = session.client("s3")

    local_tar = os.path.join(tempfile.gettempdir(), MODEL_INFO["pipeline_tar"])
    s3_key = f"{MODEL_INFO['s3_pipeline_prefix']}/{MODEL_INFO['pipeline_tar']}"

    if not os.path.exists(local_tar):
        s3_client.download_file(
            Bucket=aws_bucket,
            Key=s3_key,
            Filename=local_tar
        )

    extract_dir = os.path.join(tempfile.gettempdir(), "extracted_model")
    os.makedirs(extract_dir, exist_ok=True)

    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=extract_dir)
        joblib_files = [name for name in tar.getnames() if name.endswith(".joblib")]

    if not joblib_files:
        raise FileNotFoundError("No .joblib file found inside the model tar.gz archive.")

    local_joblib = os.path.join(extract_dir, os.path.basename(joblib_files[0]))
    pipeline = joblib.load(local_joblib)
    return pipeline


# =========================
# LOAD SHAP EXPLAINER FROM S3
# =========================
@st.cache_resource
def load_shap_explainer():
    session = get_boto_session()
    s3_client = session.client("s3")

    local_explainer = os.path.join(tempfile.gettempdir(), MODEL_INFO["explainer_file"])
    s3_key = f"{MODEL_INFO['s3_explainer_prefix']}/{MODEL_INFO['explainer_file']}"

    if not os.path.exists(local_explainer):
        s3_client.download_file(
            Bucket=aws_bucket,
            Key=s3_key,
            Filename=local_explainer
        )

    with open(local_explainer, "rb") as f:
        explainer = shap.Explainer.load(f)

    return explainer


# =========================
# MODEL METADATA
# =========================
@st.cache_data
def get_expected_columns():
    pipeline = load_pipeline_from_s3()

    if hasattr(pipeline, "feature_names_in_"):
        return list(pipeline.feature_names_in_)

    raise AttributeError(
        "The saved pipeline does not expose feature_names_in_. "
        "Use the original training column names manually if needed."
    )


# =========================
# CALL SAGEMAKER ENDPOINT
# =========================
def call_model_api(input_df: pd.DataFrame):
    predictor = Predictor(
        endpoint_name=aws_endpoint,
        sagemaker_session=get_sm_session(),
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )

    try:
        raw_pred = predictor.predict(input_df.values.astype(float))
        pred_array = np.array(raw_pred).reshape(-1)
        return pred_array, 200
    except Exception as e:
        return str(e), 500


# =========================
# LOCAL SHAP EXPLANATION
# =========================
def explain_prediction(input_df: pd.DataFrame):
    pipeline = load_pipeline_from_s3()
    explainer = load_shap_explainer()

    model = pipeline.named_steps["model"]
    preprocessing_pipeline = SklearnPipeline(steps=pipeline.steps[:-1])
    transformed = preprocessing_pipeline.transform(input_df)

    try:
        feature_names = preprocessing_pipeline.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(transformed.shape[1])]

    transformed_df = pd.DataFrame(transformed, columns=feature_names)

    shap_values = explainer(transformed_df)

    st.subheader("SHAP Waterfall Plot for First Row")
    fig = plt.figure(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    vals = np.abs(shap_values[0].values)
    top_idx = int(np.argmax(vals))
    top_feature = shap_values[0].feature_names[top_idx]
    top_value = shap_values[0].values[top_idx]

    st.info(
        f"Most influential transformed feature: {top_feature} "
        f"(SHAP contribution: {top_value:.4f})"
    )


# =========================
# UI
# =========================
st.title("S&P 500 Future Return Prediction")
st.write(
    "Upload a CSV file containing the exact feature columns used to train the model. "
    "The app will send the data to your AWS SageMaker endpoint and show predictions."
)

try:
    expected_columns = get_expected_columns()
except Exception as e:
    st.error(f"Could not load model metadata: {e}")
    st.stop()

with st.expander("Expected input columns"):
    st.write(f"Total required columns: {len(expected_columns)}")
    st.dataframe(pd.DataFrame({"required_columns": expected_columns}), use_container_width=True)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    st.subheader("Uploaded Data Preview")
    st.dataframe(input_df.head(), use_container_width=True)

    missing_cols = [col for col in expected_columns if col not in input_df.columns]
    extra_cols = [col for col in input_df.columns if col not in expected_columns]

    if missing_cols:
        st.error(
            "Your CSV is missing required columns:\n\n"
            + ", ".join(missing_cols[:25])
            + (" ..." if len(missing_cols) > 25 else "")
        )
        st.stop()

    if extra_cols:
        st.warning(
            "Extra columns found. They will be ignored:\n\n"
            + ", ".join(extra_cols[:25])
            + (" ..." if len(extra_cols) > 25 else "")
        )

    input_df = input_df[expected_columns].copy()

    # force numeric
    for col in input_df.columns:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    if input_df.isna().all(axis=1).any():
        st.warning(
            "One or more rows are entirely missing after numeric conversion. "
            "The model may fail or produce poor results."
        )

    if st.button("Run Prediction"):
        preds, status = call_model_api(input_df)

        if status != 200:
            st.error(f"Prediction failed: {preds}")
            st.stop()

        result_df = input_df.copy()
        result_df["prediction"] = preds

        st.subheader("Predictions")
        st.dataframe(result_df[["prediction"]], use_container_width=True)

        try:
            explain_prediction(input_df.iloc[[0]])
        except Exception as e:
            st.warning(f"Prediction worked, but SHAP explanation failed: {e}")
else:
    st.info("Upload a CSV file to begin.")
