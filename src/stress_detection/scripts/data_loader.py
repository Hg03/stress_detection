from sklearn.pipeline import Pipeline
from supabase import create_client
from dotenv import load_dotenv
from feast import FeatureStore
from datetime import datetime
import ibis.expr.types as ir
from tqdm import tqdm
import ibis_ml as ml
import warnings
import secrets
import joblib
import ibis
import os
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



def from_supabase(configs: dict) -> ir.Table:
    load_dotenv()
    conn = create_client(supabase_url=os.getenv("SUPABASE_URL"), supabase_key=os.getenv("SUPABASE_KEY"))
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

def split_data(configs: dict, raw_data: ir.Table) -> tuple[ir.Table, ir.Table]:
    train, test = ml.train_test_split(raw_data, test_size=configs.preprocess_data.split_ratio, unique_key=configs.data_loading.columns.unique_key, random_seed=0)
    return train, test

def preprocess_data(configs: dict, train: ir.Table, test: ir.Table) -> tuple[ir.Table, ir.Table]:
    num_imputer = ml.ImputeMean(ml.numeric())
    cat_imputer = ml.ImputeMode(ml.string())

    onehot_cols = configs.preprocess_data.columns.get("nominal", [])
    ordinal_cols = configs.preprocess_data.columns.get("ordinal", [])
    
    encoder_steps = []
    if onehot_cols:
        encoder_steps.append(ml.OneHotEncode(onehot_cols))
    if ordinal_cols:
        encoder_steps.append(ml.OrdinalEncode(ordinal_cols))

    recipe = ml.Recipe(num_imputer, cat_imputer, *encoder_steps)

    pipe = Pipeline([
        ('preprocessor', recipe)
    ]).set_output(transform="pandas")
    os.makedirs(configs.data_loading.paths.models, exist_ok=True)
    preprocessing_pipeline_path = os.path.join(configs.data_loading.paths.models, f"preprocess_{secrets.token_hex(4)}.pkl")
    joblib.dump(pipe.fit(train), preprocessing_pipeline_path)
    return ibis.memtable(pipe.fit_transform(train)), ibis.memtable(pipe.transform(test))

def to_feast(configs: dict, preprocessed_train: ir.Table, preprocessed_test: ir.Table) -> None:
    # Add the column to your Ibis tables
    preprocessed_train = preprocessed_train.mutate(event_timestamp=ibis.now())
    preprocessed_test = preprocessed_test.mutate(event_timestamp=ibis.now())
    preprocessed_train.execute().to_parquet(os.path.join(configs.data_loading.paths.feature_store, "data/training_data.parquet"))
    preprocessed_test.execute().to_parquet(os.path.join(configs.data_loading.paths.feature_store, "data/testing_data.parquet"))
    store = FeatureStore(repo_path=configs.data_loading.paths.feature_store)
    from stress_detection.feature_store.feature_definition import entity, training_fv, testing_fv
    store.apply([entity, training_fv, testing_fv])
    store.materialize_incremental(end_date=datetime.utcnow())
