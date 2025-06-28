from feast import FeatureStore
import ibis
from typing import Any
import os
from stress_detection.feature_store.feature_definition import create_schemas


def from_feast(configs: dict) -> ibis.expr.types.relations.Table:
    store = FeatureStore(repo_path=configs.training.paths.feature_store)

    # Load entity dataframe using Ibis
    path = os.path.join(configs.training.paths.feature_store, "data/training_data.parquet")
    ibis_entity_table = ibis.read_parquet(path)

    # Select only necessary columns for Feast (entity key + event_timestamp)
    entity_expr = ibis_entity_table[["employee_id", "event_timestamp"]]

    # Execute to Pandas (Feast expects a Pandas DataFrame)
    entity_df = entity_expr.execute()

    # Dynamically construct feature references
    train_features = [f"employee_training_features:{field.name}" for field in create_schemas() if field.name != "employee_id"]
    # test_features = [f"employee_testing_features:{field.name}" for field in create_schemas(train_or_test="test") if field.name != "employee_id"]
    # Fetch historical features
    train_feature_df = store.get_historical_features(
        entity_df=entity_df,
        features=train_features
    ).to_df()
    # test_feature_df = store.get_historical_features(
    #     entity_df=entity_df,
    #     features=test_features
    # ).to_df()

    # Convert the resulting DataFrame to an Ibis table
    training_data = ibis.memtable(train_feature_df)
    # testing_data = ibis.memtable(test_feature_df)

    return training_data #, testing_data

def tune_and_train(configs: dict, preprocessed_train: ibis.table, preprocessed_test: ibis.table) -> Any:
    return 1

def evaluate_model(configs: dict, model: Any, preprocessed_train: ibis.table, preprocessed_test: ibis.table) -> dict:
    return {}
