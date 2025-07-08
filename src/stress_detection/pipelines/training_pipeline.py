from stress_detection.scripts.trainer import from_feast, tune_and_train, evaluate_model

class train_orchestrator:
    def __init__(self, training_configs: dict, models: list = ["svm"]):
        self.configs = training_configs
        self.models = models

    def execute(self):
        preprocessed_train, preprocessed_test = from_feast(self.configs)
        model_artifacts = tune_and_train(self.configs, self.models, preprocessed_train, preprocessed_test)
        scores = evaluate_model(self.configs, model_artifacts, preprocessed_train, preprocessed_test)
        print(scores)

if __name__ == "__main__":
    from stress_detection.scripts.utils import load_config
    orchestrator = train_orchestrator(training_configs=load_config("training"))
    orchestrator.execute()
