"""
components/forms.py
Reusable student input form for single prediction.
"""

import streamlit as st


def render_student_form(defaults: dict = None, key_prefix: str = "form"):
    """
    Renders the student input form. Returns a dict of form values.
    `defaults` lets the Reset button restore initial values.
    """
    defaults = defaults or {}

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "🎂 Age", min_value=14, max_value=22,
            value=defaults.get("age", 16), key=f"{key_prefix}_age"
        )
        gender = st.selectbox(
            "👤 Gender", ["Male", "Female"],
            index=["Male", "Female"].index(defaults.get("gender", "Male")),
            key=f"{key_prefix}_gender"
        )
        study_hours = st.slider(
            "📚 Study Hours / Week", 0.0, 45.0,
            value=float(defaults.get("study_hours_per_week", 15.0)), step=0.5,
            key=f"{key_prefix}_study"
        )
        attendance = st.slider(
            "🏫 Attendance %", 30.0, 100.0,
            value=float(defaults.get("attendance_percentage", 80.0)), step=1.0,
            key=f"{key_prefix}_attendance"
        )

    with col2:
        previous_grade = st.slider(
            "📝 Previous Grade", 0.0, 100.0,
            value=float(defaults.get("previous_grade", 65.0)), step=1.0,
            key=f"{key_prefix}_prevgrade"
        )
        sleep_hours = st.slider(
            "😴 Sleep Hours / Night", 3.0, 11.0,
            value=float(defaults.get("sleep_hours", 7.0)), step=0.5,
            key=f"{key_prefix}_sleep"
        )
        parental_support = st.selectbox(
            "🤝 Parental Support", ["Low", "Medium", "High"],
            index=["Low", "Medium", "High"].index(defaults.get("parental_support", "Medium")),
            key=f"{key_prefix}_support"
        )

    with col3:
        extracurricular = st.selectbox(
            "🎨 Extracurricular Activities", ["Yes", "No"],
            index=["Yes", "No"].index(defaults.get("extracurricular_activities", "No")),
            key=f"{key_prefix}_extra"
        )
        internet_access = st.selectbox(
            "🌐 Internet Access", ["Yes", "No"],
            index=["Yes", "No"].index(defaults.get("internet_access", "Yes")),
            key=f"{key_prefix}_internet"
        )
        part_time_job = st.selectbox(
            "💼 Part-Time Job", ["Yes", "No"],
            index=["Yes", "No"].index(defaults.get("part_time_job", "No")),
            key=f"{key_prefix}_job"
        )

    return {
        "age": age,
        "gender": gender,
        "study_hours_per_week": study_hours,
        "attendance_percentage": attendance,
        "previous_grade": previous_grade,
        "sleep_hours": sleep_hours,
        "parental_support": parental_support,
        "extracurricular_activities": extracurricular,
        "internet_access": internet_access,
        "part_time_job": part_time_job,
    }
