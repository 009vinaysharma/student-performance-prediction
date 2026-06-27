"""
predict.py
Loads the trained models and predicts final_grade + pass/fail
for a new student record.

Usage:
    python3 predict.py
"""

import joblib
import pandas as pd

reg_model = joblib.load("models/regression_model.pkl")
clf_model = joblib.load("models/classification_model.pkl")

# Example new student — edit these values to test your own
new_student = pd.DataFrame([{
    "age": 16,
    "gender": "Female",
    "study_hours_per_week": 12,
    "attendance_percentage": 78,
    "previous_grade": 60,
    "sleep_hours": 6.5,
    "parental_support": "Medium",
    "extracurricular_activities": "Yes",
    "internet_access": "Yes",
    "part_time_job": "No",
}])

predicted_grade = reg_model.predict(new_student)[0]
predicted_status = clf_model.predict(new_student)[0]
pass_prob = clf_model.predict_proba(new_student)[0]
classes = clf_model.classes_

print("Input student profile:")
print(new_student.to_string(index=False))
print()
print(f"Predicted final grade : {predicted_grade:.1f} / 100")
print(f"Predicted status       : {predicted_status}")
print("Class probabilities    :", dict(zip(classes, pass_prob.round(3))))
