from flask import Flask, render_template, request

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("index.html", prediction=None, probability=None)

    # POST: read form values, run prediction
    data = CustomData(
        age=int(request.form["age"]),
        gender=request.form["gender"],
        state=request.form["state"],
        family_income=int(request.form["family_income"]),
        high_school_gpa=float(request.form["high_school_gpa"]),
        sat_score=int(request.form["sat_score"]),
        act_score=int(request.form["act_score"]),
        attendance_rate=float(request.form["attendance_rate"]),
        ap_courses=int(request.form["ap_courses"]),
        extracurricular_count=int(request.form["extracurricular_count"]),
        volunteer_hours=int(request.form["volunteer_hours"]),
        leadership_positions=int(request.form["leadership_positions"]),
        coding_projects=int(request.form["coding_projects"]),
        social_media_hours=float(request.form["social_media_hours"]),
        online_certifications=int(request.form["online_certifications"]),
        essay_score=float(request.form["essay_score"]),
        recommendation_score=float(request.form["recommendation_score"]),
        interview_score=float(request.form["interview_score"]),
    )

    df = data.get_data_as_dataframe()
    pred, proba = PredictPipeline().predict(df)

    return render_template(
        "index.html",
        prediction=int(pred[0]),
        probability=round(float(proba[0]) * 100, 2),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)