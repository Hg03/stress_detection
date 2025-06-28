from feast import FeatureStore
import ibis
import ibis.expr.types as ir
from typing import Any
import os
from stress_detection.feature_store.feature_definition import create_schemas


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

    train_features = [f"employee_training_features:{field.name}" for field in create_schemas() if field.name != configs.training.columns.unique_key]
    test_features = [f"employee_testing_features:{field.name}" for field in create_schemas(train_or_test="test") if field.name != configs.training.columns.unique_key]

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
        preprocessed_data.drop(to_drop).drop(target_col),
        preprocessed_data.select(target_col)
    )
def tune_and_train(configs: dict, preprocessed_train: ir.Table, preprocessed_test: ir.Table) -> Any:
    to_drop = configs.training.columns.to_drop
    target_col = configs.training.columns.target 
    X_train, y_train = get_X_y(preprocessed_data=preprocessed_train, to_drop=to_drop, target_col=target_col)
    X_test, y_test = get_X_y(preprocessed_data=preprocessed_test, to_drop=to_drop, target_col=target_col)
     

def evaluate_model(configs: dict, model: Any, preprocessed_train: ir.Table, preprocessed_test: ir.Table) -> dict:
    return {}
