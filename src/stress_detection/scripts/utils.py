from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

def load_config(pipeline: str) -> DictConfig:
    mappings = {"feature": "feature_config.yaml", "training": "training_config.yaml", "inference": "inference_config.yaml"}
    return OmegaConf.load(f"src/stress_detection/conf/{mappings[pipeline]}")