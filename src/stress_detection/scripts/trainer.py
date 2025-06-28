from feast import FeatureStore
import ibis
from typing import Any
import os
from stress_detection.feature_store.feature_definition import create_schemas


def from_feast(configs: dict) -> ibis.expr.types.relations.Table:
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

def tune_and_train(configs: dict, preprocessed_train: ibis.table, preprocessed_test: ibis.table) -> Any:
    return 1

def evaluate_model(configs: dict, model: Any, preprocessed_train: ibis.table, preprocessed_test: ibis.table) -> dict:
    return {}
