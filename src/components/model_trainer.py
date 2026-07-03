import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting features and target from arrays")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1].astype(int)
            X_test,  y_test  = test_arr[:, :-1],  test_arr[:, -1].astype(int)

            # imbalance handling for XGBoost: weight = (#negatives / #positives)
            n_pos = (y_train == 1).sum()
            n_neg = (y_train == 0).sum()
            scale_pos_weight = n_neg / n_pos

            models = {
                "LogisticRegression": LogisticRegression(
                    max_iter=2000, class_weight="balanced"),
                "DecisionTree": DecisionTreeClassifier(
                    class_weight="balanced", random_state=42),
                "RandomForest": RandomForestClassifier(
                    class_weight="balanced", n_jobs=-1, random_state=42),
                "HistGradientBoosting": HistGradientBoostingClassifier(
                    class_weight="balanced", random_state=42),
                "XGBoost": XGBClassifier(
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss", n_jobs=-1, random_state=42),
            }

            params = {
                "LogisticRegression": {
                    "C": [0.01, 0.1, 1, 10],
                    "penalty": ["l2"],
                    "solver": ["lbfgs"],
                },
                "DecisionTree": {
                    "max_depth": [5, 10, 20, None],
                    "min_samples_split": [2, 10, 50],
                },
                "RandomForest": {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20, None],
                },
                "HistGradientBoosting": {
                    "learning_rate": [0.05, 0.1],
                    "max_iter": [100, 200],
                    "max_depth": [None, 10],
                },
                "XGBoost": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 6],
                    "learning_rate": [0.05, 0.1],
                },
            }

            logging.info("Tuning + evaluating models with GridSearchCV")
            report, fitted = evaluate_models(
                X_train, y_train, X_test, y_test, models, params)
            logging.info(f"Tuned model report (ROC-AUC): {report}")

            best_model_name = max(report, key=report.get)
            best_model_score = report[best_model_name]
            best_model = fitted[best_model_name]          # the *tuned* fitted model

            if best_model_score < 0.5:
                raise CustomException("No good model found", sys)

            logging.info(
                f"Best model: {best_model_name} (ROC-AUC={best_model_score:.4f})")
            save_object(self.config.trained_model_file_path, best_model)

            # print the full scoreboard + a detailed report on the winner
            print("\n--- ROC-AUC scoreboard ---")
            for name, score in sorted(report.items(), key=lambda x: -x[1]):
                print(f"{name:22s} {score:.4f}")

            print(f"\nBest model: {best_model_name}  (ROC-AUC {best_model_score:.4f})")
            print(classification_report(y_test, best_model.predict(X_test)))

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    train_path, test_path = DataIngestion().initiate_data_ingestion()
    train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(
        train_path, test_path)
    ModelTrainer().initiate_model_trainer(train_arr, test_arr)