from stress_detection.scripts.trainer import from_feast, tune_and_train, evaluate_model

class train_orchestrator:
    def __init__(self, training_configs: dict):
        self.configs = training_configs

    def execute(self):
        preprocessed_train, preprocessed_test = from_feast(self.configs)
        model = tune_and_train(self.configs, preprocessed_train, preprocessed_test)
        scores = evaluate_model(self.configs, model, preprocessed_train, preprocessed_test)
