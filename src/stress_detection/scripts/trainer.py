from feast import FeatureStore
import ibis
from typing import Any


def from_feast(configs: dict) -> tuple[ibis.table, ibis.table]:
    store = FeatureStore(repo_path=configs.training.paths.feature_store)
    feature_vector = store.get_online_features(
        features=["employee_training_features:avg_working_hours_per_day"],
        entity_rows=[{"employee_id": eid} for eid in ["EMP0002"]]
                                                        )
    training_data = ibis.table(feature_vector.to_dict(), "training")             
    return training_data

def tune_and_train(configs: dict, preprocessed_train: ibis.table, preprocessed_test: ibis.table) -> Any:
    return 1

def evaluate_model(configs: dict, model: Any, preprocessed_train: ibis.table, preprocessed_test: ibis.table) -> dict:
    return {}
