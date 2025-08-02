from stress_detection.scripts.data_loader import from_supabase, split_data, preprocess_data, to_feast

class fe_orchestrator:
    # Feature Engineering Orchestrator
    def __init__(self, feature_configs: dict):
        self.configs = feature_configs

    def execute(self):
        print("Starting feature engineering pipeline...")
        raw_data = from_supabase(self.configs)
        train, test = split_data(self.configs, raw_data)
        preprocessed_train, preprocessed_test = preprocess_data(self.configs, train, test)
        to_feast(self.configs, preprocessed_train, preprocessed_test)
        print("Feature engineering pipeline executed successfully.")

if __name__ == "__main__":
    from stress_detection.scripts.utils import load_config
    orchestrator = fe_orchestrator(feature_configs=load_config("feature"))
    orchestrator.execute()