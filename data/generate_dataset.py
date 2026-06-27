"""
generate_dataset.py
Generates a synthetic but realistic "Student Performance" dataset and
saves it as data/student_performance.csv

Features:
    - study_hours_per_week
    - attendance_percentage
    - previous_grade (0-100)
    - sleep_hours
    - parental_support (Low/Medium/High)
    - extracurricular_activities (Yes/No)
    - internet_access (Yes/No)
    - part_time_job (Yes/No)
    - gender
    - age

Target:
    - final_grade (0-100, continuous)  -> for regression
    - pass_fail (Pass/Fail, threshold 40) -> for classification
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1500

study_hours = np.clip(np.random.normal(15, 7, N), 0, 45)
attendance = np.clip(np.random.normal(80, 12, N), 30, 100)
previous_grade = np.clip(np.random.normal(65, 15, N), 0, 100)
sleep_hours = np.clip(np.random.normal(7, 1.5, N), 3, 11)
parental_support = np.random.choice(["Low", "Medium", "High"], N, p=[0.25, 0.45, 0.30])
extracurricular = np.random.choice(["Yes", "No"], N, p=[0.4, 0.6])
internet_access = np.random.choice(["Yes", "No"], N, p=[0.85, 0.15])
part_time_job = np.random.choice(["Yes", "No"], N, p=[0.3, 0.7])
gender = np.random.choice(["Male", "Female"], N)
age = np.random.randint(15, 19, N)

support_map = {"Low": -4, "Medium": 0, "High": 5}
support_effect = np.array([support_map[s] for s in parental_support])
job_effect = np.where(part_time_job == "Yes", -3, 0)
extra_effect = np.where(extracurricular == "Yes", 2, 0)
internet_effect = np.where(internet_access == "Yes", 2, -3)

noise = np.random.normal(0, 6, N)

final_grade = (
    0.55 * previous_grade
    + 1.1 * study_hours
    + 0.25 * attendance
    + 1.5 * sleep_hours
    + support_effect
    + job_effect
    + extra_effect
    + internet_effect
    + noise
    - 26
)
final_grade = np.clip(final_grade, 0, 100).round(1)
pass_fail = np.where(final_grade >= 40, "Pass", "Fail")

df = pd.DataFrame({
    "age": age,
    "gender": gender,
    "study_hours_per_week": study_hours.round(1),
    "attendance_percentage": attendance.round(1),
    "previous_grade": previous_grade.round(1),
    "sleep_hours": sleep_hours.round(1),
    "parental_support": parental_support,
    "extracurricular_activities": extracurricular,
    "internet_access": internet_access,
    "part_time_job": part_time_job,
    "final_grade": final_grade,
    "pass_fail": pass_fail,
})

out_path = "data/student_performance.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df.head())
print("\nPass/Fail distribution:")
print(df["pass_fail"].value_counts())
