from metaflow import FlowSpec, step
from stress_detection.scripts.utils import load_config
from stress_detection.pipelines.feature_pipeline import fe_orchestrator
from stress_detection.pipelines.training_pipeline import train_orchestrator
from stress_detection.pipelines.inference_pipeline import infer_orchestrator

class StressDetectionFlow(FlowSpec):

    @step
    def start(self):
        self.feature_configs = load_config("feature")
        self.training_configs = load_config("training")
        self.inference_configs = load_config("inference")
        self.next(self.feature_pipeline)
    
    @step
    def feature_pipeline(self):
        fe_orchestrator(feature_configs=self.feature_configs).execute()
        self.next(self.training_pipeline)

    @step
    def training_pipeline(self):
        train_orchestrator(training_configs=self.training_configs).execute()
        self.next(self.inference_pipeline)

    @step
    def inference_pipeline(self):
        infer_orchestrator(inference_configs=self.inference_configs).execute()
        self.next(self.end)

    @step
    def end(self):
        print("Stress Detection Flow completed successfully.")

if __name__ == "__main__":
    StressDetectionFlow()