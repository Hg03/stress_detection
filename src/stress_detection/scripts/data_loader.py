from omegaconf.dictconfig import DictConfig
import polars as pl


def preprocess_data(config: DictConfig, train: pl.DataFrame, test: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    return (pl.DataFrame(), pl.DataFrame())

def to_supabase(config: DictConfig, train: pl.DataFrame, test: pl.DataFrame) -> None:
    ...