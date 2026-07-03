import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    """End-to-end training: ingestion -> transformation -> model training."""

    def run(self):
        try:
            logging.info(">>> Training pipeline started")

            train_path, test_path = DataIngestion().initiate_data_ingestion()
            train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(
                train_path, test_path
            )
            best_name, best_score = ModelTrainer().initiate_model_trainer(
                train_arr, test_arr
            )

            logging.info(
                f">>> Training pipeline finished. Best: {best_name} (AUC={best_score:.4f})"
            )
            return best_name, best_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    TrainPipeline().run()