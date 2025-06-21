import ibis

schema = {
    'id': 'int64',
    'name': 'string',
    'created_at': 'timestamp'
}
mock_data = ibis.table(schema, name='mock_data')

def from_supabase(configs: dict) -> ibis.table:
    return mock_data


def split_data(configs: dict) -> tuple[ibis.table, ibis.table]:
    return mock_data, mock_data

def preprocess_data(configs: dict, train: ibis.table, test: ibis.table) -> tuple[ibis.table, ibis.table]:
    return train, test

def to_feast(configs: dict, preprocessed_train: ibis.table, preprocessed_test: ibis.table) -> None:
    ...