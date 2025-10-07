# src/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import data_path, ensure_dirs
from pathlib import Path
import joblib

def preprocess(input_file=None):
    ensure_dirs()
    if input_file is None:
        input_file = data_path("data.csv")
    else:
        input_file = Path(input_file)
    df = pd.read_csv(input_file)
    print("Original shape:", df.shape)

    # Drop 'id' if present, drop any Unnamed columns
    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed") or c.lower()=="id"]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print("Dropped columns:", drop_cols)

    # Ensure target column 'diagnosis' exists
    if 'diagnosis' not in df.columns:
        raise ValueError("target column 'diagnosis' not found in dataset")

    # Drop rows with missing values (or you can impute)
    df = df.dropna().reset_index(drop=True)
    print("After dropna shape:", df.shape)

    # Encode target: M -> 1, B -> 0
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    if df['diagnosis'].isnull().any():
        raise ValueError("Some diagnosis values could not be mapped to 0/1")

    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    # Split (stratify to keep class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Save processed files
    processed_dir = data_path("processed")
    X_train_scaled.to_csv(processed_dir / "X_train.csv", index=False)
    X_test_scaled.to_csv(processed_dir / "X_test.csv", index=False)
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)

    # Save scaler for inference
    joblib.dump(scaler, data_path("scaler.pkl"))
    print("Preprocessing complete. Files saved to data/processed and scaler at data/scaler.pkl")

if __name__ == "__main__":
    preprocess()
