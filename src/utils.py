import os
import sys
import dill

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV 
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


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    For each model, GridSearchCV over its param grid (3-fold, scored by ROC-AUC),
    refit with the best params, then score on the test set.
    Returns {name: test_auc} and the dict of fitted best models.
    """
    try:
        report = {}
        fitted = {}

        for name, model in models.items():
            grid = params.get(name, {})

            if grid:                       # tune if a grid is provided
                gs = GridSearchCV(
                    model, grid,
                    scoring="roc_auc",
                    cv=3,
                    n_jobs=-1,
                    verbose=1,
                )
                gs.fit(X_train, y_train)
                best = gs.best_estimator_
                print(f"{name}: best params -> {gs.best_params_}")
            else:                          # no grid: fit as-is
                model.fit(X_train, y_train)
                best = model

            if hasattr(best, "predict_proba"):
                y_test_proba = best.predict_proba(X_test)[:, 1]
            else:
                y_test_proba = best.decision_function(X_test)

            report[name] = roc_auc_score(y_test, y_test_proba)
            fitted[name] = best

        return report, fitted
    except Exception as e:
        raise CustomException(e, sys)