import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.target_column = "admission_status"
        self.drop_columns = ["student_id"]          # ID -> no signal, drop it

    def get_data_transformer_object(self, numeric_features, categorical_features):
        """Build the ColumnTransformer: scale numerics, one-hot the categoricals."""
        try:
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore")),
            ])

            logging.info(f"Numeric features: {numeric_features}")
            logging.info(f"Categorical features: {categorical_features}")

            preprocessor = ColumnTransformer(transformers=[
                ("num", num_pipeline, numeric_features),
                ("cat", cat_pipeline, categorical_features),
            ])
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data")

            # drop the ID column
            train_df = train_df.drop(columns=self.drop_columns)
            test_df = test_df.drop(columns=self.drop_columns)

            # split features / target
            input_train = train_df.drop(columns=[self.target_column])
            target_train = train_df[self.target_column]
            input_test = test_df.drop(columns=[self.target_column])
            target_test = test_df[self.target_column]

            # derive column types from the training features (no hardcoding)
            numeric_features = input_train.select_dtypes(exclude="object").columns.tolist()
            categorical_features = input_train.select_dtypes(include="object").columns.tolist()

            preprocessor = self.get_data_transformer_object(
                numeric_features, categorical_features
            )

            logging.info("Fitting preprocessor on training data")
            input_train_arr = preprocessor.fit_transform(input_train)
            input_test_arr = preprocessor.transform(input_test)

            # OneHotEncoder can return a sparse matrix -> make dense for stacking
            if hasattr(input_train_arr, "toarray"):
                input_train_arr = input_train_arr.toarray()
                input_test_arr = input_test_arr.toarray()

            # attach target as the last column
            train_arr = np.c_[input_train_arr, np.array(target_train)]
            test_arr = np.c_[input_test_arr, np.array(target_test)]

            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessor,
            )
            logging.info("Saved preprocessor object")

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    train_path, test_path = DataIngestion().initiate_data_ingestion()
    train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(
        train_path, test_path
    )
    print("Train array shape:", train_arr.shape)
    print("Test array shape:", test_arr.shape)
    print("Preprocessor saved to artifacts/preprocessor.pkl")