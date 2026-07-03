import os
import sys
import dill

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