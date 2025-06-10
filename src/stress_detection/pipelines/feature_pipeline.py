from metaflow import FlowSpec, step
from stress_detection.scripts.utils import load_config, from_supabase, split_data
from stress_detection.scripts.data_loader import preprocess_data, to_supabase
from loguru import logger

class feature_orchestrator(FlowSpec):
    
    @step
    def start(self):
        logger.info("Feature Pipeline Initialized !!")
        self.next(self.load_feature_config)

    @step
    def load_feature_config(self):
        logger.info("Loading Feature Pipeline Config...")
        self.feature_config = load_config(pipeline="feature")
        self.next(self.load_raw_data)

    @step
    def load_raw_data(self):
        logger.info("Raw Data load...")
        raw_data = from_supabase(config=self.feature_config, type_of_data="raw")
        self.train, self.test = split_data(config=self.feature_config, data=raw_data, type_of_data="raw")
        self.next(self.preprocess)

    @step
    def preprocess(self):
        logger.info("Preprocessing Data...")
        self.preprocessed_train, self.preprocessed_test = preprocess_data(config=self.feature_config, train=self.train, test=self.test)
        self.next(self.save_data)

    @step
    def save_data(self):
        logger.info("Loading the preprocessed data with train test label extra to Supabase...")
        to_supabase(config=self.feature_config, train=self.preprocessed_train, test=self.preprocessed_train)
        self.next(self.end)

    @step
    def end(self):
        logger.info("Feature Orchestrator Pipeline Completed !!")


if __name__ == "__main__":
    feature_orchestrator()