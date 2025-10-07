# src/predict.py
import pandas as pd
import joblib
from src.utils import data_path, model_path, ensure_dirs
from pathlib import Path

def predict_and_save(input_file=None):
    ensure_dirs()
    model_file = model_path()
    if not Path(model_file).exists():
        raise FileNotFoundError("Model not found. Run training first.")

    # Choose input processed file if not provided
    proc_dir = data_path("processed")
    if input_file is None:
        input_file = proc_dir / "X_test.csv"
    else:
        input_file = Path(input_file)

    scaler_file = data_path("scaler.pkl")
    scaler = joblib.load(scaler_file) if Path(scaler_file).exists() else None
    model = joblib.load(model_file)

    df = pd.read_csv(input_file)
    # If scaler exists, assume df is already scaled; but we stored scaled CSVs, so skip re-scaling
    preds = model.predict(df)
    probs = model.predict_proba(df)[:,1]

    out = pd.DataFrame(df)
    out["prediction"] = preds
    out["probability"] = probs

    pred_file = data_path("predictions") / "predictions.csv"
    out.to_csv(pred_file, index=False)
    print(f"Predictions saved to {pred_file}")

if __name__ == "__main__":
    predict_and_save()
