import os
import sys
import dill

from sklearn.metrics import roc_auc_score

from src.exception import CustomException


def save_object(file_path, obj):
    """Pickle any Python object (e.g. the fitted preprocessor) to disk."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load a pickled object back from disk (used at prediction time)."""
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """Fit each model and score it by test ROC-AUC. Returns {name: auc}."""
    try:
        report = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                y_test_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_test_proba = model.decision_function(X_test)
            report[name] = roc_auc_score(y_test, y_test_proba)
        return report
    except Exception as e:
        raise CustomException(e, sys)