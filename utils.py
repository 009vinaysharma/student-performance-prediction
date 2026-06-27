"""
utils.py
Core utility functions for the Streamlit dashboard.
Loads the EXISTING trained models (.pkl) from /models — never retrains.
"""

import os
import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

REG_MODEL_PATH = os.path.join(MODELS_DIR, "regression_model.pkl")
CLF_MODEL_PATH = os.path.join(MODELS_DIR, "classification_model.pkl")
METRICS_PATH = os.path.join(OUTPUTS_DIR, "metrics_report.txt")
DATASET_PATH = os.path.join(DATA_DIR, "student_performance.csv")

NUMERIC_FEATURES = [
    "age", "study_hours_per_week", "attendance_percentage",
    "previous_grade", "sleep_hours"
]
CATEGORICAL_FEATURES = [
    "gender", "parental_support", "extracurricular_activities",
    "internet_access", "part_time_job"
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ----------------------------------------------------------------------
# Model loading (cached so .pkl files are read from disk only once)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    """Loads the existing pre-trained regression and classification models."""
    reg_model, clf_model = None, None
    errors = []

    if os.path.exists(REG_MODEL_PATH):
        try:
            reg_model = joblib.load(REG_MODEL_PATH)
        except Exception as e:
            errors.append(f"Regression model failed to load: {e}")
    else:
        errors.append(f"Missing file: {REG_MODEL_PATH}")

    if os.path.exists(CLF_MODEL_PATH):
        try:
            clf_model = joblib.load(CLF_MODEL_PATH)
        except Exception as e:
            errors.append(f"Classification model failed to load: {e}")
    else:
        errors.append(f"Missing file: {CLF_MODEL_PATH}")

    return reg_model, clf_model, errors


@st.cache_data(show_spinner=False)
def load_dataset():
    """Loads the existing dataset (used only for charts, not retraining)."""
    if os.path.exists(DATASET_PATH):
        try:
            return pd.read_csv(DATASET_PATH)
        except Exception:
            return None
    return None


# ----------------------------------------------------------------------
# Metrics report parsing
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def parse_metrics_report():
    """
    Parses outputs/metrics_report.txt and returns a dict of metrics.
    Falls back to sensible defaults if the file is missing or malformed.
    """
    metrics = {
        "mae": None, "rmse": None, "r2": None,
        "accuracy": None, "precision": None, "recall": None, "f1": None,
        "raw_text": None, "found": False
    }

    if not os.path.exists(METRICS_PATH):
        return metrics

    try:
        with open(METRICS_PATH, "r") as f:
            text = f.read()
        metrics["raw_text"] = text
        metrics["found"] = True

        patterns = {
            "mae": r"MAE\s*:\s*([\d.]+)",
            "rmse": r"RMSE\s*:\s*([\d.]+)",
            "r2": r"R\^?2\s*:\s*([\d.]+)",
            "accuracy": r"Accuracy\s*:\s*([\d.]+)",
            "precision": r"Precision\s*:\s*([\d.]+)",
            "recall": r"Recall\s*:\s*([\d.]+)",
            "f1": r"F1-?score\s*:\s*([\d.]+)",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                metrics[key] = float(match.group(1))
    except Exception:
        pass

    return metrics


# ----------------------------------------------------------------------
# Prediction helpers
# ----------------------------------------------------------------------
def build_input_dataframe(form_data: dict) -> pd.DataFrame:
    """Converts form input dict into a single-row DataFrame matching model schema."""
    row = {feat: form_data.get(feat) for feat in ALL_FEATURES}
    return pd.DataFrame([row])


def predict_student(reg_model, clf_model, input_df: pd.DataFrame):
    """
    Runs both existing models on a single student row.
    Returns dict with grade, status, confidence, and class probabilities.
    """
    result = {
        "predicted_grade": None,
        "status": None,
        "confidence": None,
        "probabilities": {},
        "error": None,
    }
    try:
        if reg_model is not None:
            grade = reg_model.predict(input_df)[0]
            result["predicted_grade"] = float(np.clip(grade, 0, 100))

        if clf_model is not None:
            status = clf_model.predict(input_df)[0]
            proba = clf_model.predict_proba(input_df)[0]
            classes = clf_model.classes_
            prob_map = {c: float(p) for c, p in zip(classes, proba)}
            result["status"] = status
            result["probabilities"] = prob_map
            result["confidence"] = float(max(proba))
    except Exception as e:
        result["error"] = str(e)

    return result


def predict_bulk(reg_model, clf_model, df: pd.DataFrame) -> pd.DataFrame:
    """Runs predictions on a bulk-uploaded dataframe. Returns df with extra columns."""
    df = df.copy()

    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Uploaded CSV is missing required columns: {missing}")

    X = df[ALL_FEATURES]

    if reg_model is not None:
        df["predicted_final_grade"] = np.clip(reg_model.predict(X), 0, 100).round(1)
    if clf_model is not None:
        df["predicted_status"] = clf_model.predict(X)
        proba = clf_model.predict_proba(X)
        df["confidence"] = proba.max(axis=1).round(3)

    return df


# ----------------------------------------------------------------------
# Feature importance (extracted directly from the existing regression model)
# ----------------------------------------------------------------------
def get_feature_importance(reg_model):
    """Extracts feature importances from the existing trained RandomForest pipeline."""
    try:
        preprocessor = reg_model.named_steps["preprocessor"]
        model = reg_model.named_steps["model"]

        cat_names = list(
            preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
        )
        feature_names = NUMERIC_FEATURES + cat_names
        importances = model.feature_importances_

        fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
        return fi_df
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])


# ----------------------------------------------------------------------
# Evaluation data for charts (re-uses existing models, no retraining)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_eval_data(_reg_model, _clf_model):
    """
    Runs the existing models over the full existing dataset to produce
    actual-vs-predicted and confusion-matrix chart data.
    No training happens here — purely inference for visualization.
    """
    df = load_dataset()
    if df is None or _reg_model is None or _clf_model is None:
        return None

    X = df[ALL_FEATURES]
    y_true_grade = df["final_grade"]
    y_true_status = df["pass_fail"]

    y_pred_grade = _reg_model.predict(X)
    y_pred_status = _clf_model.predict(X)

    return {
        "y_true_grade": y_true_grade,
        "y_pred_grade": y_pred_grade,
        "y_true_status": y_true_status,
        "y_pred_status": y_pred_status,
        "pass_fail_counts": df["pass_fail"].value_counts(),
    }


# ----------------------------------------------------------------------
# AI-style recommendations
# ----------------------------------------------------------------------
def generate_recommendations(form_data: dict, predicted_grade: float, status: str):
    """Generates simple rule-based, human-readable recommendations."""
    tips = []

    study_hours = form_data.get("study_hours_per_week", 0)
    attendance = form_data.get("attendance_percentage", 0)
    sleep_hours = form_data.get("sleep_hours", 0)
    previous_grade = form_data.get("previous_grade", 0)
    parental_support = form_data.get("parental_support", "Medium")
    part_time_job = form_data.get("part_time_job", "No")
    extracurricular = form_data.get("extracurricular_activities", "No")

    if study_hours < 10:
        tips.append("📚 Increase study hours — aim for at least 12–15 hours per week for steady improvement.")
    elif study_hours < 18:
        tips.append("📈 Good study routine — pushing slightly higher could boost your grade further.")

    if attendance < 75:
        tips.append("🏫 Improve attendance — consistent class attendance strongly correlates with better grades.")

    if sleep_hours < 6.5:
        tips.append("😴 Sleep 7–8 hours per night — adequate rest improves focus and retention.")
    elif sleep_hours > 9.5:
        tips.append("⏰ Slightly reduce oversleeping — balance rest with productive study time.")

    if previous_grade < 50:
        tips.append("🎯 Focus on foundational topics to strengthen previous weak areas.")
    else:
        tips.append("✅ Maintain your previous grade level with consistent revision.")

    if parental_support == "Low":
        tips.append("🤝 Seek additional academic support — tutoring or study groups can help fill the gap.")

    if part_time_job == "Yes":
        tips.append("⚖️ Balance part-time work with study time to avoid burnout.")

    if extracurricular == "No":
        tips.append("🎨 Consider light extracurricular involvement — it's linked with better overall engagement.")

    if status == "Fail" or (predicted_grade is not None and predicted_grade < 40):
        tips.insert(0, "⚠️ Predicted result is at risk — prioritize the recommendations below immediately.")
    elif predicted_grade is not None and predicted_grade >= 80:
        tips.insert(0, "🌟 Excellent trajectory — keep up the current habits!")

    return tips[:6] if tips else ["✅ Keep up the great work — no major risk factors detected."]
