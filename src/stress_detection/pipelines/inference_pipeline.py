from stress_detection.scripts.infer import trigger_api

class infer_orchestrator:
    def __init__(self, inference_configs: dict):
        self.configs = inference_configs

    def execute(self):
        trigger_api(self.configs)