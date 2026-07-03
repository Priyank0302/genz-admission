# Gen-Z College Admission Predictor

An end-to-end machine learning system that predicts whether a Gen-Z applicant will be admitted to college, based on academic, extracurricular, and lifestyle features. Built as a modular, production-style project with a Flask front-end.

**Best model:** Logistic Regression &nbsp;|&nbsp; **ROC-AUC:** 0.8805 &nbsp;|&nbsp; **Non-admit recall:** 0.81

---

## Project overview

- **Problem:** Binary classification — predict admission (1) or rejection (0)
- **Dataset:** [Gen-Z College Admission Dataset](https://www.kaggle.com/datasets/sharmajicoder/genn-z-college-admission-dataset) (Kaggle) — 1,000,000 rows, 20 columns
- **Approach:** Modular ML pipeline (ingestion → transformation → training → tuning) with a Flask app for interactive predictions
- **Deliverable:** A web app where you enter an applicant's profile and get a predicted admission probability

## Key results

| Model                | ROC-AUC (tuned) |
|----------------------|-----------------|
| **Logistic Regression** | **0.8805**   |
| XGBoost              | 0.8774          |
| HistGradientBoosting | 0.8765          |
| Random Forest        | 0.8703          |
| Decision Tree        | 0.8304          |

The dataset is imbalanced (88.1% admit / 11.9% reject), so models are evaluated on **ROC-AUC** and **minority-class recall**, not accuracy. Logistic Regression wins because the underlying relationship between features and admission is largely linear/additive.

## Tech stack

Python 3.11, pandas, NumPy, scikit-learn, XGBoost, Flask, dill

## Project structure

```
genz-admission/
├── notebook/
│   ├── data/Dataset.csv              # raw Kaggle data
│   └── 1_EDA.ipynb                   # exploratory analysis + baseline
├── src/
│   ├── logger.py                     # timestamped logs
│   ├── exception.py                  # custom errors with file + line
│   ├── utils.py                      # save/load pickles, GridSearchCV runner
│   ├── components/
│   │   ├── data_ingestion.py         # read CSV + stratified split
│   │   ├── data_transformation.py    # ColumnTransformer + save preprocessor
│   │   └── model_trainer.py          # race 5 models, pick best by AUC
│   └── pipeline/
│       ├── train_pipeline.py         # one entry point to retrain
│       └── predict_pipeline.py       # load artifacts + score new applicants
├── templates/                        # home.html, index.html
├── artifacts/                        # generated: train/test csv, model.pkl, preprocessor.pkl
├── app.py                            # Flask server
├── requirements.txt
├── setup.py
└── README.md
```

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/Priyank0302/genz-admission.git
cd genz-admission
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The `-e .` line in `requirements.txt` installs the local `src/` folder as an editable package, so `from src.components...` imports work anywhere.

### 2. Get the data

Download the [Kaggle dataset](https://www.kaggle.com/datasets/sharmajicoder/genn-z-college-admission-dataset) and place it at `notebook/data/Dataset.csv`.

### 3. Train

Run the full training pipeline (ingestion → transformation → tuned model training):

```bash
python src/pipeline/train_pipeline.py
```

This produces `artifacts/model.pkl` and `artifacts/preprocessor.pkl`.

### 4. Launch the app

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000), fill in the applicant profile, and hit **Predict admission**.

## Design decisions

A few choices worth flagging, since they differ from a typical tutorial build:

- **Stratified train/test split** preserves the 88/12 class ratio in both sets — non-negotiable with imbalanced data.
- **Class weighting** (`class_weight="balanced"` for sklearn models, `scale_pos_weight` for XGBoost) prevents the boosters from collapsing into "predict admit for everyone."
- **Down-sampling to 200k rows** during ingestion keeps iteration fast on a laptop; the full 1M is available via a config flag.
- **`HistGradientBoostingClassifier` over KNN** — KNN is prohibitively slow at prediction time on this data volume.
- **Preprocessor is pickled separately** so prediction-time data is transformed with the exact same scalers and encoders as training. This is the single most important production habit in the whole project.

## What's not included (yet)

- Cloud deployment (AWS Elastic Beanstalk / Docker)
- CI/CD via GitHub Actions
- Experiment tracking (MLflow)
- Data-drift monitoring

## Credits

Project structure inspired by [Krish Naik's End-to-End ML Project series](https://github.com/krishnaik06/mlproject), adapted for a classification problem on a much larger, imbalanced dataset with a modernized model lineup and evaluation strategy.