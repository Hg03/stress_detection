import ibis
from typing import Any

schema = {
    'id': 'int64',
    'name': 'string',
    'created_at': 'timestamp'
}
mock_data = ibis.table(schema, name='mock_data')

def from_feast(configs: dict) -> tuple[ibis.table, ibis.table]:
    return mock_data, mock_data

def tune_and_train(configs: dict, preprocessed_train: ibis.table, preprocessed_test: ibis.table) -> Any:
    return 1

def evaluate_model(configs: dict, model: Any, preprocessed_train: ibis.table, preprocessed_test: ibis.table) -> dict:
    return {}