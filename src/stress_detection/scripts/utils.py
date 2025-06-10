from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import polars as pl
from dotenv import load_dotenv
from supabase import create_client
from tqdm import tqdm
import os

def load_config(pipeline: str):
    mappings = {"feature": "feature_config.yaml", "training": "training_config.yaml", "inference": "inference_config.yaml"}
    return OmegaConf.load(f"src/stress_detection/conf/{mappings[pipeline]}")

def from_supabase(config: DictConfig, type_of_data: str) -> pl.DataFrame:
    types_of_data = ["raw", "preprocessed"]
    load_dotenv()
    conn = create_client(supabase_url=os.getenv("supabase_url"), supabase_key=os.getenv("supabase_key"))
    if type_of_data == types_of_data[0]:
        json_data = []
        batch_size, offset = config.data_loading.batch_size, config.data_loading.offset
        total_rows = conn.table(config.data_loading.raw_data_table_name).select("count", count="exact").execute().count
        # Create progress bar
        progress_bar = tqdm(total=total_rows,desc="Loading data from Supabase",unit=" rows")
        while True:
            response = conn.table(config.data_loading.raw_data_table_name).select("*").limit(batch_size).offset(offset).execute()
            batch = response.data
            if not batch:
                break
            json_data.extend(batch)
            offset+=batch_size
            progress_bar.update(len(batch))
        progress_bar.close()
        raw_data = pl.DataFrame(json_data)
        raw_data = raw_data.drop(pl.col(config.data_loading.columns.to_drop))
        raw_data = raw_data.filter(pl.col(config.data_loading.columns.target).is_not_null())
        return raw_data
    elif type_of_data == types_of_data[1]:
        return pl.DataFrame()

def split_data(config: DictConfig, data: pl.DataFrame, type_of_data: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    types_of_data = ["raw", "preprocessed"]
    if type_of_data == types_of_data[0]:
        # Add row indices before sampling
        data_with_idx = data.with_row_index("original_idx")
        
        # Sample for training
        train_data = data_with_idx.sample(fraction=config.preprocess_data.split_ratio, shuffle=True)
        
        # Get test data by filtering out the training indices
        train_indices = train_data.select("original_idx")
        test_data = data_with_idx.filter(~pl.col("original_idx").is_in(train_indices.to_series()))
        
        # Remove the index column from both datasets
        train_data = train_data.drop("original_idx")
        test_data = test_data.drop("original_idx")
        
        return train_data, test_data
    elif type_of_data == types_of_data[1]:
        return pl.DataFrame(), pl.DataFrame()
    

if __name__ == "__main__":
    config=load_config(pipeline="feature")
    data = from_supabase(config=config, type_of_data="raw")
    print("raw")
    split_data_ = split_data(config=config, data=data, type_of_data="raw")