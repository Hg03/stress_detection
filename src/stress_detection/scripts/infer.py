from stress_detection.scripts.utils import WorkFrom, WorkLifeBalance, LivesWithFamily, WorkingState, load_config
from fastapi import FastAPI, Form
from typing import Any, Tuple
from typing import Annotated
import joblib
import ibis
import os
ibis.options.interactive = True

app = FastAPI()

def get_latest_model_and_preprocessor() -> Tuple[Any, Any]:
    configs = load_config("inference")
    models_path = configs.infer.paths.models
    pkl_files = [
        os.path.join(models_path, f)
        for f in os.listdir(models_path)
        if f.endswith('.pkl') and os.path.isfile(os.path.join(models_path, f))
    ]
    preprocessor_pkl_files = [f for f in pkl_files if 'preprocess' in f]
    model_files = [f for f in pkl_files if 'preprocess' not in f]
    if not pkl_files:
        raise FileNotFoundError("No model files found in the specified directory.")
    latest_model_file = max(model_files, key=os.path.getmtime)
    latest_preprocessor_file = max(preprocessor_pkl_files, key=os.path.getmtime)
    return joblib.load(latest_preprocessor_file), joblib.load(latest_model_file)

def trigger_api(configs: dict):
    print('..')


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/infer/")
async def infer(
    employee_id: Annotated[str, Form(description="EMP_123")],
    avg_working_hours_per_day: Annotated[float, Form(..., ge=0, le=24)],
    work_from: Annotated[WorkFrom, Form()],
    work_pressure: Annotated[int, Form(..., ge=1, le=5)],
    manager_support: Annotated[int, Form(..., ge=1, le=5)],
    sleeping_habit: Annotated[int, Form(..., ge=1, le=5)],
    exercise_habit: Annotated[int, Form(..., ge=1, le=5)],
    job_satisfaction: Annotated[int, Form(..., ge=1, le=5)],
    work_life_balance: Annotated[WorkLifeBalance, Form()],
    social_person: Annotated[int, Form(..., ge=1, le=5)],
    lives_with_family: Annotated[LivesWithFamily, Form()],
    working_state: Annotated[WorkingState, Form()]
) -> dict:
    input_values = ibis.memtable([{
        "employee_id": employee_id,
        "avg_working_hours_per_day": avg_working_hours_per_day,
        "work_from": work_from,
        "work_pressure": work_pressure,
        "manager_support": manager_support,
        "sleeping_habit": sleeping_habit,
        "exercise_habit": exercise_habit,
        "job_satisfaction": job_satisfaction,
        "work_life_balance": work_life_balance,
        "social_person": social_person,
        "lives_with_family": lives_with_family,
        "working_state": working_state,
        "stress_level": 0
    }])
    preprocessor, best_model = get_latest_model_and_preprocessor()
    preprocessed_values = preprocessor.transform(input_values)
    preds = best_model.predict(preprocessed_values.drop(["employee_id", "stress_level"], axis=1))
    return {"stress_level": preds[0]}