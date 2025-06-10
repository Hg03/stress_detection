from omegaconf.dictconfig import DictConfig
from typing import Any
import polars as pl

def training_data(config: DictConfig, train: pl.DataFrame, test: pl.DataFrame):
    ...

def save_to_dagshub(config: DictConfig, model: Any) -> None:
    ...