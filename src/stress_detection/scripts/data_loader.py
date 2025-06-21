from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import ibis_ml as ml
from supabase import create_client
from dotenv import load_dotenv
from tqdm import tqdm
import ibis
import os

schema = {
    'id': 'int64',
    'name': 'string',
    'created_at': 'timestamp'
}
mock_data = ibis.table(schema, name='mock_data')

def from_supabase(configs: dict) -> ibis.table:
    load_dotenv()
    conn = create_client(supabase_url=os.getenv("supabase_url"), supabase_key=os.getenv("supabase_key"))
    json_data = []
    batch_size, offset = configs.data_loading.batch_size, configs.data_loading.offset
    total_rows = conn.table(configs.data_loading.raw_data_table_name).select("count", count="exact").execute().count
    # Create progress bar
    progress_bar = tqdm(total=total_rows,desc="Loading data from Supabase",unit=" rows")
    while True:
        response = conn.table(configs.data_loading.raw_data_table_name).select("*").limit(batch_size).offset(offset).execute()
        batch = response.data
        if not batch:
            break
        json_data.extend(batch)
        offset+=batch_size
        progress_bar.update(len(batch))
    progress_bar.close()
    raw_data = ibis.memtable(json_data)
    col_map = {col.lower(): col for col in raw_data.columns}
    raw_data = raw_data.rename(col_map)
    raw_data = raw_data.filter(raw_data[configs.data_loading.columns.target].notnull())
    return raw_data


def split_data(configs: dict, raw_data: ibis.table) -> tuple[ibis.table, ibis.table]:
    train, test = ml.train_test_split(raw_data, test_size=configs.preprocess_data.split_ratio, unique_key=configs.data_loading.columns.unique_key, random_seed=0)
    return train, test

def preprocess_data(configs: dict, train: ibis.table, test: ibis.table) -> tuple[ibis.table, ibis.table]:
    imputer = ml.ImputeMean(ml.numeric())
    encoder = ml.OrdinalEncode(ml.string())
    recipe = ml.Recipe(imputer, encoder)
    pipe = Pipeline([('preprocessor', recipe)]).set_output(transform="pandas")
    return ibis.memtable(pipe.fit_transform(train)), ibis.memtable(pipe.transform(test))

def to_feast(configs: dict, preprocessed_train: ibis.table, preprocessed_test: ibis.table) -> None:
    print("we are here")