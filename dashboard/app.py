# dashboard/app.py
import streamlit as st
from src.utils import data_path, model_path, results_path
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Breast Cancer Detection Dashboard", layout="wide")

st.title("AI-Enhanced Early Disease Detection — Breast Cancer")

DATA_FILE = data_path("data.csv")
PROC_PRED_FILE = data_path("predictions") / "predictions.csv"
METRICS_FILE = results_path("metrics.txt")
MODEL_FILE = model_path()

# Sidebar: Data preview toggles
st.sidebar.header("Controls")
show_data = st.sidebar.checkbox("Show dataset preview", value=True)
show_metrics = st.sidebar.checkbox("Show model metrics", value=True)
show_predictions = st.sidebar.checkbox("Show latest predictions", value=True)

# Load original data
if DATA_FILE.exists():
    df = pd.read_csv(DATA_FILE)
else:
    st.error("Original dataset not found in data/data.csv")
    df = pd.DataFrame()

if show_data and not df.empty:
    st.subheader("Dataset preview")
    st.dataframe(df.head(200))
    st.write("Dataset shape:", df.shape)
    st.write("Columns:", list(df.columns))

# Visualization: class distribution
if not df.empty:
    if 'diagnosis' in df.columns:
        st.subheader("Class distribution (diagnosis)")
        df_local = df.copy()
        df_local['diagnosis'] = df_local['diagnosis'].map({'M':'Malignant','B':'Benign'})
        st.bar_chart(df_local['diagnosis'].value_counts())

# Model metrics
if show_metrics:
    st.subheader("Model performance metrics")
    if Path(METRICS_FILE).exists():
        with open(METRICS_FILE, "r") as f:
            metrics_text = f.read()
        st.text(metrics_text)
    else:
        st.info("No metrics found — run training to generate metrics.")

# Prediction form (manual single-sample prediction)
st.subheader("Manual prediction (enter feature values)")
# We need feature names: use processed X_train or original df minus diagnosis
proc_X = None
proc_dir = data_path("processed")
if (proc_dir / "X_train.csv").exists():
    proc_X = pd.read_csv(proc_dir / "X_train.csv")
elif 'diagnosis' in df.columns:
    proc_X = df.drop(columns=['diagnosis'])

if proc_X is None or proc_X.shape[1] == 0:
    st.warning("No feature columns found for prediction form.")
else:
    features = list(proc_X.columns[:15])  # show first 15 features for manual input to keep form manageable
    with st.form("predict_form"):
        inputs = {}
        for feat in features:
            # Use mean as default
            default_val = float(proc_X[feat].mean()) if feat in proc_X else 0.0
            inputs[feat] = st.number_input(feat, value=default_val, format="%.6f")
        submitted = st.form_submit_button("Predict")

        if submitted:
            if not Path(MODEL_FILE).exists():
                st.error("Trained model not found. Please run training.")
            else:
                model = joblib.load(MODEL_FILE)
                sample_df = pd.DataFrame([inputs])
                # If scaler exists, use it:
                scaler_file = data_path("scaler.pkl")
                if scaler_file.exists():
                    scaler = joblib.load(scaler_file)
                    sample_df = pd.DataFrame(scaler.transform(sample_df), columns=sample_df.columns)
                pred = model.predict(sample_df)[0]
                prob = model.predict_proba(sample_df)[0,1]
                label = "Malignant" if pred==1 else "Benign"
                st.success(f"Prediction: **{label}** (probability malignant = {prob:.3f})")

# Show latest predictions file
if show_predictions:
    st.subheader("Latest predictions (from data/predictions/predictions.csv)")
    if Path(PROC_PRED_FILE).exists():
        preds_df = pd.read_csv(PROC_PRED_FILE)
        st.dataframe(preds_df.head(200))
        csv = preds_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions.csv", csv, "predictions.csv", "text/csv")
    else:
        st.info("No prediction file found. Run prediction job first.")

st.markdown("---")
st.caption("Tip: Use the batch scripts in batch_jobs/ to run the pipeline automatically (cron in Linux/WSL or Task Scheduler on Windows).")
