import os
import sys

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    """Load the saved preprocessor + model and score a new applicant."""

    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        try:
            logging.info("Loading preprocessor and model")
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)

            data_scaled = preprocessor.transform(features)

            if hasattr(data_scaled, "toarray"):
                data_scaled = data_scaled.toarray()

            pred = model.predict(data_scaled)
            proba = model.predict_proba(data_scaled)[:, 1]
            return pred, proba

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Wraps one applicant's form values into a pandas DataFrame
    shaped exactly like the training features.
    """

    def __init__(
        self,
        age: int,
        gender: str,
        state: str,
        family_income: int,
        high_school_gpa: float,
        sat_score: int,
        act_score: int,
        attendance_rate: float,
        ap_courses: int,
        extracurricular_count: int,
        volunteer_hours: int,
        leadership_positions: int,
        coding_projects: int,
        social_media_hours: float,
        online_certifications: int,
        essay_score: float,
        recommendation_score: float,
        interview_score: float,
    ):
        self.age = age
        self.gender = gender
        self.state = state
        self.family_income = family_income
        self.high_school_gpa = high_school_gpa
        self.sat_score = sat_score
        self.act_score = act_score
        self.attendance_rate = attendance_rate
        self.ap_courses = ap_courses
        self.extracurricular_count = extracurricular_count
        self.volunteer_hours = volunteer_hours
        self.leadership_positions = leadership_positions
        self.coding_projects = coding_projects
        self.social_media_hours = social_media_hours
        self.online_certifications = online_certifications
        self.essay_score = essay_score
        self.recommendation_score = recommendation_score
        self.interview_score = interview_score

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            data = {
                "age": [self.age],
                "gender": [self.gender],
                "state": [self.state],
                "family_income": [self.family_income],
                "high_school_gpa": [self.high_school_gpa],
                "sat_score": [self.sat_score],
                "act_score": [self.act_score],
                "attendance_rate": [self.attendance_rate],
                "ap_courses": [self.ap_courses],
                "extracurricular_count": [self.extracurricular_count],
                "volunteer_hours": [self.volunteer_hours],
                "leadership_positions": [self.leadership_positions],
                "coding_projects": [self.coding_projects],
                "social_media_hours": [self.social_media_hours],
                "online_certifications": [self.online_certifications],
                "essay_score": [self.essay_score],
                "recommendation_score": [self.recommendation_score],
                "interview_score": [self.interview_score],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys)