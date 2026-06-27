"""
train_model.py
Trains two models on the student performance dataset:
  1. Regression model -> predicts final_grade (continuous, 0-100)
  2. Classification model -> predicts pass_fail (Pass/Fail)

Saves:
  - models/regression_model.pkl
  - models/classification_model.pkl
  - models/preprocessor.pkl
  - outputs/feature_importance.png
  - outputs/actual_vs_predicted.png
  - outputs/confusion_matrix.png
  - outputs/metrics_report.txt
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

DATA_PATH = "data/student_performance.csv"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "age", "study_hours_per_week", "attendance_percentage",
    "previous_grade", "sleep_hours"
]
CATEGORICAL_FEATURES = [
    "gender", "parental_support", "extracurricular_activities",
    "internet_access", "part_time_job"
]

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows, {df.shape[1]} columns\n")

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y_reg = df["final_grade"]
y_clf = df["pass_fail"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
])

# ---------- Train/test split (shared indices for fair comparison) ----------
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=RANDOM_STATE
)

# =========================================================
# 1. REGRESSION MODEL  -> predict final_grade
# =========================================================
print("=" * 60)
print("REGRESSION MODEL: Predicting final_grade")
print("=" * 60)

reg_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, max_depth=8, random_state=RANDOM_STATE))
])

reg_pipeline.fit(X_train, y_reg_train)
y_pred_reg = reg_pipeline.predict(X_test)

mae = mean_absolute_error(y_reg_test, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_reg))
r2 = r2_score(y_reg_test, y_pred_reg)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R^2  : {r2:.3f}\n")

# =========================================================
# 2. CLASSIFICATION MODEL -> predict pass_fail
# =========================================================
print("=" * 60)
print("CLASSIFICATION MODEL: Predicting pass_fail")
print("=" * 60)

clf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=RANDOM_STATE,
        class_weight="balanced"
    ))
])

clf_pipeline.fit(X_train, y_clf_train)
y_pred_clf = clf_pipeline.predict(X_test)

acc = accuracy_score(y_clf_test, y_pred_clf)
prec = precision_score(y_clf_test, y_pred_clf, pos_label="Pass")
rec = recall_score(y_clf_test, y_pred_clf, pos_label="Pass")
f1 = f1_score(y_clf_test, y_pred_clf, pos_label="Pass")

print(f"Accuracy  : {acc:.3f}")
print(f"Precision : {prec:.3f}")
print(f"Recall    : {rec:.3f}")
print(f"F1-score  : {f1:.3f}\n")
print(classification_report(y_clf_test, y_pred_clf))

# =========================================================
# Save models
# =========================================================
joblib.dump(reg_pipeline, "models/regression_model.pkl")
joblib.dump(clf_pipeline, "models/classification_model.pkl")
print("Saved trained models to models/")

# =========================================================
# Plots
# =========================================================
sns.set_style("whitegrid")

# Actual vs Predicted (regression)
plt.figure(figsize=(6, 6))
plt.scatter(y_reg_test, y_pred_reg, alpha=0.5, color="#4C72B0")
plt.plot([0, 100], [0, 100], "r--", lw=2)
plt.xlabel("Actual Final Grade")
plt.ylabel("Predicted Final Grade")
plt.title(f"Actual vs Predicted Final Grade (R2={r2:.2f})")
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png", dpi=150)
plt.close()

# Confusion matrix (classification)
cm = confusion_matrix(y_clf_test, y_pred_clf, labels=["Pass", "Fail"])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pass", "Fail"], yticklabels=["Pass", "Fail"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Pass/Fail")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()

# Feature importance (from regression model)
feature_names = (
    NUMERIC_FEATURES +
    list(reg_pipeline.named_steps["preprocessor"]
         .named_transformers_["cat"]
         .get_feature_names_out(CATEGORICAL_FEATURES))
)
importances = reg_pipeline.named_steps["model"].feature_importances_
fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=True)

plt.figure(figsize=(7, 6))
plt.barh(fi_df["feature"], fi_df["importance"], color="#55A868")
plt.xlabel("Importance")
plt.title("Feature Importance (Regression Model)")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
plt.close()

# =========================================================
# Save text report
# =========================================================
with open("outputs/metrics_report.txt", "w") as f:
    f.write("STUDENT PERFORMANCE PREDICTION - MODEL METRICS\n")
    f.write("=" * 50 + "\n\n")
    f.write("REGRESSION MODEL (final_grade)\n")
    f.write(f"  MAE  : {mae:.2f}\n")
    f.write(f"  RMSE : {rmse:.2f}\n")
    f.write(f"  R^2  : {r2:.3f}\n\n")
    f.write("CLASSIFICATION MODEL (pass_fail)\n")
    f.write(f"  Accuracy  : {acc:.3f}\n")
    f.write(f"  Precision : {prec:.3f}\n")
    f.write(f"  Recall    : {rec:.3f}\n")
    f.write(f"  F1-score  : {f1:.3f}\n\n")
    f.write(classification_report(y_clf_test, y_pred_clf))

print("\nAll outputs saved to outputs/ and models/")
