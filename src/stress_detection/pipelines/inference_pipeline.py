from metaflow import FlowSpec, step
from stress_detection.scripts.utils import load_config
from stress_detection.scripts.infer import call_fastapi
from loguru import logger

class inference_orchestrator(FlowSpec):
    
    @step
    def start(self):
        logger.info("Inference Pipeline Initialized !!")
        self.next(self.load_inference_config)

    @step
    def load_inference_config(self):
        logger.info("Loading Inference Pipeline Config...")
        self.inference_config = load_config(pipeline="inference")
        self.next(self.get_the_model)

    @step
    def get_the_model(self):
        logger.info("Taking model from dagshub mlflow") 
        self.next(self.trigger_api)   

    @step
    def trigger_api(self):
        logger.info("Initialize Inference FastAPI UI...")
        call_fastapi(config=self.inference_config)
        self.next(self.end)

    @step
    def end(self):
        logger.info("Inference Orchestrator Pipeline Running !!")

if __name__ == "__main__":
    inference_orchestrator()