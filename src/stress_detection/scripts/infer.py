from typing import Any
import joblib
import os

def get_latest_model(configs: dict) -> Any:
    models_path = configs.infer.paths.models
    pkl_files = [
        os.path.join(models_path, f)
        for f in os.listdir(models_path)
        if f.endswith('.pkl') and os.path.isfile(os.path.join(models_path, f))
    ]
    if not pkl_files:
        raise FileNotFoundError("No model files found in the specified directory.")
    latest_model_file = max(pkl_files, key=os.path.getmtime)
    return joblib.load(latest_model_file)

def trigger_api(configs: dict):
    ...