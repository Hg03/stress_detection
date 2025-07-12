from stress_detection.feature_store.feature_definition import create_schemas
from stress_detection.scripts.utils import model_mappings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from feast import FeatureStore
import ibis.expr.types as ir
from typing import Any
import warnings
import secrets
import dagshub
import joblib
import mlflow
import ibis
import os
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def from_feast(configs: dict) -> ir.Table:
    store = FeatureStore(repo_path=configs.training.paths.feature_store)

    train_path = os.path.join(configs.training.paths.feature_store, configs.training.paths.training_data)
    test_path = os.path.join(configs.training.paths.feature_store, configs.training.paths.testing_data)
    train_entity_table = ibis.read_parquet(train_path)
    test_entity_table = ibis.read_parquet(test_path)

    train_entity_expr = train_entity_table[[configs.training.columns.unique_key, configs.training.columns.event_timestamp]]
    test_entity_expr = test_entity_table[[configs.training.columns.unique_key, configs.training.columns.event_timestamp]]

    train_entity_df = train_entity_expr.execute()
    test_entity_df = test_entity_expr.execute()

    train_features = [f"training_features:{field.name}" for field in create_schemas() if field.name != configs.training.columns.unique_key]
    test_features = [f"testing_features:{field.name}" for field in create_schemas(train_or_test="test") if field.name != configs.training.columns.unique_key]

    # Fetch historical features
    train_feature_df = store.get_historical_features(
        entity_df=train_entity_df,
        features=train_features
    ).to_df()
    test_feature_df = store.get_historical_features(
        entity_df=test_entity_df,
        features=test_features
    ).to_df()

    training_data = ibis.memtable(train_feature_df)
    testing_data = ibis.memtable(test_feature_df)
    return training_data , testing_data

def get_X_y(preprocessed_data: ir.Table, to_drop: list[str], target_col: str) -> tuple[ir.Table, ir.Table]:
    return (
        preprocessed_data.drop(to_drop).drop(target_col).execute(),
        preprocessed_data.select(target_col).execute()
    )

def make_model_pipeline(configs: dict, models: list[str]) -> GridSearchCV:
    param_grid = []
    for model in models:
        model_instance = model_mappings(model_name=model)
        model_params = configs.training.models.get(model, {})
        model_param_dict = {
            "classifier": [model_instance]
        }
        model_param_dict.update(model_params)
        param_grid.append(model_param_dict)
    
    pipeline = Pipeline([("classifier", LogisticRegression())])
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=2,
        scoring='accuracy',
        verbose=1
    )


def tune_and_train(configs: dict, models: list[str], preprocessed_train: ir.Table, preprocessed_test: ir.Table) -> dict:
    to_drop = configs.training.columns.to_drop
    target_col = configs.training.columns.target 
    X_train, y_train = get_X_y(preprocessed_data=preprocessed_train, to_drop=to_drop, target_col=target_col)
    X_test, y_test = get_X_y(preprocessed_data=preprocessed_test, to_drop=to_drop, target_col=target_col)
    model_pipeline = make_model_pipeline(configs=configs, models=models)
    model_pipeline.fit(X_train, y_train[configs.training.columns.target].to_numpy())
    return {"model": model_pipeline, "X_train": X_train, "y_train": y_train , "X_test": X_test, "y_test": y_test}
     

def evaluate_model(configs: dict, model_artifacts: Any) -> dict:
    dagshub.init(repo_owner=os.getenv("DAGSHUB_USERNAME"), repo_name='stress_detection', mlflow=True)
    dagshub.auth.add_app_token(token=os.getenv('DAGSHUB_TOKEN'))
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.sklearn.autolog(log_datasets=False, log_post_training_metrics=False, log_model_signatures=False, log_models=False)
    with mlflow.start_run():
        model = model_artifacts["model"].best_estimator_
        X_train, y_train = model_artifacts["X_train"], model_artifacts["y_train"]
        X_test, _ = model_artifacts["X_test"], model_artifacts["y_test"]

        # Fit and predict
        model.fit(X_train, y_train[configs.training.columns.target].to_numpy())
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        os.makedirs(configs.training.paths.models, exist_ok=True)
        model_path = os.path.join(configs.training.paths.models, f"model_{secrets.token_hex(4)}.pkl")
        joblib.dump(model, model_path)
        print("Training predictions:", train_preds.shape)
        print("Testing predictions:", test_preds.shape)
