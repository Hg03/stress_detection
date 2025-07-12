from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from omegaconf.dictconfig import DictConfig
from omegaconf import OmegaConf
from sklearn.svm import SVC
from enum import Enum
from typing import Any

class WorkFrom(str, Enum):
    home = "Home"
    office = "Office"
    hybrid = "Hybrid"

class WorkLifeBalance(str, Enum):
    no = "No"
    yes = "Yes"

class LivesWithFamily(str, Enum):
    yes = "Yes"
    no = "No"

class WorkingState(str, Enum):
    delhi = "Delhi"
    hyderabad = "Hyderabad"
    karnataka = "Karnataka"
    pune = "Pune"
    chennai = "Chennai"

def load_config(pipeline: str) -> DictConfig:
    mappings = {"feature": "feature_config.yaml", "training": "training_config.yaml", "inference": "inference_config.yaml"}
    return OmegaConf.load(f"src/stress_detection/conf/{mappings[pipeline]}")

def model_mappings(model_name: str) -> Any:
    model_map = {"svm": SVC(), "rf": RandomForestClassifier(), "gb": GradientBoostingClassifier()}
    return model_map.get(model_name)