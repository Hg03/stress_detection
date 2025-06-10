from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import polars as pl

def load_config(pipeline: str):
    mappings = {"feature": "feature_config.yaml", "training": "training_config.yaml", "inference": "inference_config.yaml"}
    return OmegaConf.load(f"src/stress_detection/conf/{mappings[pipeline]}")

def from_supabase(config: DictConfig, type_of_data: str) -> pl.DataFrame:
    types_of_data = ["raw", "preprocessed"]
    return pl.DataFrame()

def split_data(config: DictConfig, data: pl.DataFrame, type_of_data: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    types_of_data = ["raw", "preprocessed"]
    return (pl.DataFrame(), pl.DataFrame())