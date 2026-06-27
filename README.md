# Student Performance Prediction

A machine learning project that predicts student academic performance
(final grade and pass/fail status) based on study habits, attendance,
demographics, and lifestyle factors — now with a premium Streamlit
AI dashboard built on top of the existing trained models.

##project live Link : https://student-performance-prediction-dlhtne9fjtqyzzokylv7hs.streamlit.app/
## 🚀 Run the Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard loads your **existing** `models/regression_model.pkl` and
`models/classification_model.pkl` — it never retrains anything.

### Dashboard Pages
- **📊 Dashboard** — KPI cards (Accuracy, Precision, Recall, F1) parsed live from
  `outputs/metrics_report.txt`, dataset snapshot, recent prediction history.
- **🎯 Predict** — Single-student form with sliders/selectboxes, instant prediction
  (grade, Pass/Fail badge, confidence, probability chart, gauge chart), AI-generated
  recommendations, PDF report download. Also supports bulk CSV upload for batch
  predictions with CSV export.
- **📈 Analytics** — Actual vs Predicted chart, Confusion Matrix, Pass/Fail pie chart,
  Feature Importance chart — all computed by running the existing models over the
  existing dataset (no retraining).
- **ℹ️ About Model** — Architecture summary and raw metrics report.
- **📥 Download Report** — PDF of the latest prediction, CSV of prediction history,
  and the raw metrics text file.

### New Project Structure
```
student_performance_prediction/
├── app.py                       # Main Streamlit dashboard (NEW)
├── utils.py                     # Model loading, prediction, metrics parsing (NEW)
├── components/                  # Modular UI components (NEW)
│   ├── cards.py                 # Hero header, KPI cards, badges, insight cards
│   ├── charts.py                # All Plotly charts (gauge, FI, pie, confusion matrix...)
│   ├── forms.py                 # Reusable student input form
│   └── pdf_report.py            # PDF report generator (fpdf2)
├── assets/
│   └── style.css                # Dark glassmorphism / neon purple-blue theme (NEW)
├── data/
│   ├── generate_dataset.py      # Creates the synthetic dataset (unchanged)
│   └── student_performance.csv  # 1500-row dataset (unchanged)
├── models/
│   ├── regression_model.pkl     # Existing trained model — NOT retrained
│   └── classification_model.pkl # Existing trained model — NOT retrained
├── outputs/
│   ├── actual_vs_predicted.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── metrics_report.txt       # Parsed automatically for KPI cards
├── train_model.py               # Original training script (unchanged)
├── predict.py                   # Original CLI prediction script (unchanged)
├── requirements.txt             # Updated with streamlit, plotly, fpdf2
└── README.md
```

> `train_model.py` and `predict.py` are untouched — the dashboard is a pure
> integration layer that loads the same `.pkl` files.



## Dataset

1500 synthetic student records with these features:

| Feature | Description |
|---|---|
| age | Student age (15-18) |
| gender | Male / Female |
| study_hours_per_week | Hours spent studying per week |
| attendance_percentage | Class attendance % |
| previous_grade | Prior academic score (0-100) |
| sleep_hours | Average nightly sleep |
| parental_support | Low / Medium / High |
| extracurricular_activities | Yes / No |
| internet_access | Yes / No |
| part_time_job | Yes / No |

Targets:
- **final_grade** — continuous score (0-100), used for regression
- **pass_fail** — Pass/Fail (grade >= 40), used for classification

The dataset is generated with realistic correlations (e.g., more study
hours and better attendance push the grade up, a part-time job pulls it
down slightly) plus random noise so the problem isn't trivially easy.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. (Optional) Regenerate the dataset
```bash
python3 data/generate_dataset.py
```

### 2. Train the models
```bash
python3 train_model.py
```
This trains a Random Forest Regressor (final_grade) and a Random Forest
Classifier (pass_fail), prints metrics, and saves models + plots.

### 3. Predict on a new student
Edit the `new_student` dictionary inside `predict.py` with the student's
details, then run:
```bash
python3 predict.py
```

## Model Performance (on held-out 20% test set)

**Regression (final grade prediction)**
- MAE: ~6.2 points
- RMSE: ~7.5 points
- R²: ~0.71

**Classification (pass/fail prediction)**
- Accuracy: ~92%
- Precision (Pass): ~94%
- Recall (Pass): ~97%
- F1-score (Pass): ~95%

(Exact numbers are in `outputs/metrics_report.txt` after training — they
may vary slightly with random seeds and library versions.)

## Approach

1. **Data preprocessing** — numeric features scaled with `StandardScaler`,
   categorical features one-hot encoded, all wrapped in a single
   scikit-learn `ColumnTransformer` + `Pipeline` so preprocessing is
   never separated from the model.
2. **Models** — `RandomForestRegressor` and `RandomForestClassifier`
   (class-balanced) chosen for strong baseline performance and built-in
   feature importance, with no heavy tuning required.
3. **Evaluation** — train/test split (80/20), standard regression
   metrics (MAE, RMSE, R²) and classification metrics (accuracy,
   precision, recall, F1, confusion matrix).
4. **Explainability** — feature importance plot shows which factors
   (study hours, previous grade, attendance, etc.) drive predictions.

## Ideas for Extending This Project

- Try gradient boosting (XGBoost / LightGBM) for potentially better accuracy
- Add SHAP values for per-student explainability
- Build a simple Streamlit/Flask app around `predict.py` for live input
- Use a real-world dataset (e.g., UCI Student Performance dataset) instead
  of the synthetic one
- Add cross-validation and hyperparameter tuning (GridSearchCV)
