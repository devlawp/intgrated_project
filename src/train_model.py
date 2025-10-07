# src/train_model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib
from src.utils import data_path, model_path, results_path, ensure_dirs

def train_and_save():
    ensure_dirs()
    proc = data_path("processed")
    X_train = pd.read_csv(proc / "X_train.csv")
    X_test = pd.read_csv(proc / "X_test.csv")
    y_train = pd.read_csv(proc / "y_train.csv")
    y_test = pd.read_csv(proc / "y_test.csv")
    # y files may be single-column; ensure Series
    y_train = y_train.iloc[:,0]
    y_test = y_test.iloc[:,0]

    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Save metrics
    results_file = results_path("metrics.txt")
    with open(results_file, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))

    # Save model
    model_file = model_path()
    joblib.dump(model, model_file)
    print(f"Model trained and saved to {model_file}")
    print(f"Metrics saved to {results_file}")

if __name__ == "__main__":
    train_and_save()
