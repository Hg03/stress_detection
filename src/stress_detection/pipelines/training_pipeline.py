from metaflow import FlowSpec, step
from stress_detection.scripts.utils import load_config, from_supabase, split_data
from stress_detection.scripts.trainer import training_data, save_to_dagshub
from loguru import logger

class training_orchestrator(FlowSpec):
    
    @step
    def start(self):
        logger.info("Training Pipeline Initialized !!")
        self.next(self.load_training_config)

    @step
    def load_training_config(self):
        logger.info("Loading Training Pipeline Config...")
        self.training_config = load_config(pipeline="training")
        self.next(self.load_train_data)

    @step
    def load_train_data(self):
        logger.info("Training Data load...")
        raw_data = from_supabase(config=self.training_config, type_of_data="preprocessed")
        self.preprocessed_train, self.preprocessed_test = split_data(config=self.training_config, data=raw_data, type_of_data="preprocessed")
        self.next(self.train)

    @step
    def train(self):
        logger.info("Training Data...")
        self.model = training_data(config=self.training_config, train=self.preprocessed_train, test=self.preprocessed_test)
        self.next(self.save_model)

    @step
    def save_model(self):
        logger.info("Loading the best model to dagshub mlflow...")
        save_to_dagshub(config=self.training_config, model=self.model)
        self.next(self.end)

    @step
    def end(self):
        logger.info("Training Orchestrator Pipeline Completed !!")

if __name__ == "__main__":
    training_orchestrator()