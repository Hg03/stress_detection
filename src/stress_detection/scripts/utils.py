from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from typing import Any

def load_config(pipeline: str) -> DictConfig:
    mappings = {"feature": "feature_config.yaml", "training": "training_config.yaml", "inference": "inference_config.yaml"}
    return OmegaConf.load(f"src/stress_detection/conf/{mappings[pipeline]}")

def model_mappings(model_name: str) -> Any:
    model_map = {"svm": SVC(), "rf": RandomForestClassifier(), "gb": GradientBoostingClassifier()}
    return model_map.get(model_name)