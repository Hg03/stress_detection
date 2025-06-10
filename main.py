# run_all_pipelines.py

from stress_detection.pipelines.feature_pipeline import feature_orchestrator
from stress_detection.pipelines.training_pipeline import training_orchestrator
from stress_detection.pipelines.inference_pipeline import inference_orchestrator

def run_flow(flow_cls):
    """
    Run a Metaflow flow programmatically
    """
    flow = flow_cls()
    flow.run(exit=False)

if __name__ == "__main__":
    print("Running Feature Pipeline...")
    run_flow(feature_orchestrator)

    print("Running Training Pipeline...")
    run_flow(training_orchestrator)

    print("Running Inference Pipeline...")
    run_flow(inference_orchestrator)

    print("All pipelines completed successfully.")
