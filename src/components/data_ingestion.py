import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    source_data_path: str = os.path.join("notebook", "data", "Dataset.csv")
    raw_data_path:    str = os.path.join("artifacts", "raw.csv")
    train_data_path:  str = os.path.join("artifacts", "train.csv")
    test_data_path:   str = os.path.join("artifacts", "test.csv")
    target_column:    str = "admission_status"
    sample_size:      int = 200000     # set to None to use all 1,000,000 rows
    test_size:      float = 0.2
    random_state:     int = 42


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            cfg = self.ingestion_config

            df = pd.read_csv(cfg.source_data_path)
            logging.info(f"Read dataset: {df.shape[0]} rows, {df.shape[1]} columns")

            # optional stratified down-sampling for fast iteration on this large file
            if cfg.sample_size and cfg.sample_size < len(df):
                df, _ = train_test_split(
                    df,
                    train_size=cfg.sample_size,
                    stratify=df[cfg.target_column],
                    random_state=cfg.random_state,
                )
                logging.info(f"Down-sampled to {df.shape[0]} rows (stratified, ratio preserved)")

            os.makedirs(os.path.dirname(cfg.raw_data_path), exist_ok=True)
            df.to_csv(cfg.raw_data_path, index=False, header=True)
            logging.info("Saved raw data to artifacts")

            train_set, test_set = train_test_split(
                df,
                test_size=cfg.test_size,
                stratify=df[cfg.target_column],     # keep the 88/12 balance in both splits
                random_state=cfg.random_state,
            )
            logging.info("Performed stratified train-test split")

            train_set.to_csv(cfg.train_data_path, index=False, header=True)
            test_set.to_csv(cfg.test_data_path, index=False, header=True)
            logging.info("Saved train and test data to artifacts")

            return cfg.train_data_path, cfg.test_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print("Train saved to:", train_path)
    print("Test saved to:", test_path)