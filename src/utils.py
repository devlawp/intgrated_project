# src/utils.py
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def data_path(*parts):
    return ROOT.joinpath("data", *parts)

def model_path(filename="trained_model.pkl"):
    return ROOT.joinpath("model", filename)

def results_path(*parts):
    return ROOT.joinpath("results", *parts)

def ensure_dirs():
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "predictions").mkdir(parents=True, exist_ok=True)
    (ROOT / "model").mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
