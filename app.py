"""
app.py
Premium AI SaaS-style Streamlit dashboard for the Student Performance
Prediction project.

IMPORTANT: This file only INTEGRATES the existing trained models
(models/regression_model.pkl, models/classification_model.pkl).
It never retrains anything — train_model.py and predict.py are untouched.

Run with:
    streamlit run app.py
"""

import os
import io
import pandas as pd
import streamlit as st

from utils import (
    load_models, load_dataset, parse_metrics_report,
    build_input_dataframe, predict_student, predict_bulk,
    get_feature_importance, get_eval_data, generate_recommendations,
    ALL_FEATURES,
)
from components.cards import (
    render_hero, render_kpi_row, render_badge, render_insight_card,
    render_section_title, render_divider, render_glass_card_open, render_glass_card_close,
)
from components.charts import (
    gauge_chart, feature_importance_chart, probability_chart,
    pass_fail_pie, actual_vs_predicted_chart, confusion_matrix_chart,
)
from components.forms import render_student_form
from components.pdf_report import generate_pdf_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Student Performance AI Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    css_path = os.path.join(BASE_DIR, "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()

# =========================================================
# LOAD EXISTING MODELS + METRICS (cached, no retraining)
# =========================================================
reg_model, clf_model, model_errors = load_models()
metrics = parse_metrics_report()
dataset_df = load_dataset()

if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:
    st.markdown(
        "<h2 style='font-family:Poppins;font-weight:800;"
        "background:linear-gradient(135deg,#6366f1,#ec4899);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>"
        "🎓 EduPredict AI</h2>",
        unsafe_allow_html=True,
    )
    st.caption("Student Performance Intelligence")
    render_divider()

    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🎯 Predict", "📈 Analytics", "ℹ️ About Model", "📥 Download Report"],
        label_visibility="collapsed",
    )

    render_divider()
    st.markdown("<p class='small-muted'>Model Status</p>", unsafe_allow_html=True)
    if reg_model is not None and clf_model is not None:
        st.success("Models loaded ✅", icon="✅")
    else:
        st.error("Model loading issue ⚠️")
        for err in model_errors:
            st.caption(err)

    render_divider()
    st.markdown(
        "<p class='small-muted'>Built with Streamlit + Plotly<br>"
        "Powered by your trained RandomForest models</p>",
        unsafe_allow_html=True,
    )

# =========================================================
# PAGE: DASHBOARD
# =========================================================
if page == "📊 Dashboard":
    render_hero(
        "Student Performance Prediction",
        "AI-powered analytics dashboard for forecasting academic outcomes and unlocking actionable insights.",
    )

    render_section_title("📌 Model Performance Overview")

    kpi_metrics = {
        "Accuracy": (f"{metrics['accuracy']*100:.1f}%" if metrics["accuracy"] else "N/A", "🎯"),
        "Precision": (f"{metrics['precision']*100:.1f}%" if metrics["precision"] else "N/A", "🧪"),
        "Recall": (f"{metrics['recall']*100:.1f}%" if metrics["recall"] else "N/A", "🔁"),
        "F1 Score": (f"{metrics['f1']*100:.1f}%" if metrics["f1"] else "N/A", "⚖️"),
    }
    render_kpi_row(kpi_metrics)

    if not metrics["found"]:
        st.warning(
            "outputs/metrics_report.txt not found — KPI cards show N/A. "
            "Run train_model.py once to generate it (no retraining needed if it already exists)."
        )

    st.write("")
    col1, col2 = st.columns(2)

    with col1:
        render_glass_card_open()
        st.markdown("##### 🧮 Regression Metrics (Final Grade)")
        st.metric("MAE", f"{metrics['mae']:.2f}" if metrics["mae"] else "N/A")
        st.metric("RMSE", f"{metrics['rmse']:.2f}" if metrics["rmse"] else "N/A")
        st.metric("R² Score", f"{metrics['r2']:.3f}" if metrics["r2"] else "N/A")
        render_glass_card_close()

    with col2:
        render_glass_card_open()
        st.markdown("##### 🧩 Dataset Snapshot")
        if dataset_df is not None:
            st.metric("Total Students", len(dataset_df))
            st.metric("Pass Rate", f"{(dataset_df['pass_fail']=='Pass').mean()*100:.1f}%")
            st.metric("Avg Final Grade", f"{dataset_df['final_grade'].mean():.1f}")
        else:
            st.info("Dataset not found in data/ folder.")
        render_glass_card_close()

    render_divider()
    render_section_title("🕓 Recent Predictions")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history[::-1])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.info("No predictions yet. Head to the **Predict** page to try the model.")

# =========================================================
# PAGE: PREDICT
# =========================================================
elif page == "🎯 Predict":
    render_hero("Predict Student Performance", "Fill in the student profile to generate an instant AI prediction.")

    tab1, tab2 = st.tabs(["🧍 Single Student", "📂 Bulk CSV Upload"])

    # ---------------- Single Prediction ----------------
    with tab1:
        if "reset_trigger" not in st.session_state:
            st.session_state.reset_trigger = 0

        render_section_title("Student Profile")
        form_data = render_student_form(key_prefix=f"single_{st.session_state.reset_trigger}")

        c1, c2 = st.columns([3, 1])
        with c1:
            predict_clicked = st.button("🚀 Predict Performance", use_container_width=True)
        with c2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.reset_trigger += 1
                st.rerun()

        if predict_clicked:
            if reg_model is None or clf_model is None:
                st.error("Models are not loaded. Check the models/ folder.")
            else:
                input_df = build_input_dataframe(form_data)
                result = predict_student(reg_model, clf_model, input_df)

                if result["error"]:
                    st.error(f"Prediction failed: {result['error']}")
                else:
                    st.session_state.last_result = result
                    st.session_state.last_form = form_data
                    st.session_state.history.append({
                        "Age": form_data["age"],
                        "Gender": form_data["gender"],
                        "Predicted Grade": round(result["predicted_grade"], 1),
                        "Status": result["status"],
                        "Confidence": f"{result['confidence']*100:.1f}%",
                    })

        if st.session_state.get("last_result"):
            result = st.session_state.last_result
            form_data = st.session_state.last_form

            render_divider()
            render_section_title("🔮 Prediction Result")

            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                render_glass_card_open()
                st.markdown("**Predicted Final Grade**")
                st.markdown(
                    f"<h2 style='margin:0'>{result['predicted_grade']:.1f} / 100</h2>",
                    unsafe_allow_html=True,
                )
                render_glass_card_close()
            with res_col2:
                render_glass_card_open()
                st.markdown("**Pass / Fail Status**")
                render_badge(result["status"])
                render_glass_card_close()
            with res_col3:
                render_glass_card_open()
                st.markdown("**Confidence Score**")
                st.markdown(
                    f"<h2 style='margin:0'>{result['confidence']*100:.1f}%</h2>",
                    unsafe_allow_html=True,
                )
                render_glass_card_close()

            st.write("")
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.plotly_chart(gauge_chart(result["predicted_grade"]), use_container_width=True)
            with chart_col2:
                st.plotly_chart(probability_chart(result["probabilities"]), use_container_width=True)

            render_divider()
            render_section_title("💡 AI Insights & Recommendations")
            tips = generate_recommendations(form_data, result["predicted_grade"], result["status"])
            for tip in tips:
                render_insight_card(tip)

            render_divider()
            pdf_bytes = generate_pdf_report(form_data, result, tips)
            st.download_button(
                "📄 Download PDF Report",
                data=pdf_bytes,
                file_name="student_prediction_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    # ---------------- Bulk CSV ----------------
    with tab2:
        render_section_title("Bulk Prediction via CSV")
        st.markdown(
            f"<p class='small-muted'>CSV must contain columns: {', '.join(ALL_FEATURES)}</p>",
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                bulk_df = pd.read_csv(uploaded_file)
                if reg_model is None or clf_model is None:
                    st.error("Models are not loaded.")
                else:
                    result_df = predict_bulk(reg_model, clf_model, bulk_df)
                    st.success(f"Predicted {len(result_df)} student records.")
                    st.dataframe(result_df, use_container_width=True, hide_index=True)

                    csv_buffer = io.StringIO()
                    result_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "⬇️ Download Predictions as CSV",
                        data=csv_buffer.getvalue(),
                        file_name="bulk_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    if "pass_fail" not in bulk_df.columns:
                        pie_counts = result_df["predicted_status"].value_counts()
                        st.plotly_chart(pass_fail_pie(pie_counts), use_container_width=True)
            except Exception as e:
                st.error(f"Could not process CSV: {e}")

# =========================================================
# PAGE: ANALYTICS
# =========================================================
elif page == "📈 Analytics":
    render_hero("Model Analytics", "Deep-dive into how the existing trained models behave across the dataset.")

    if reg_model is None or clf_model is None or dataset_df is None:
        st.warning("Models or dataset not available — analytics charts require both.")
    else:
        eval_data = get_eval_data(reg_model, clf_model)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                actual_vs_predicted_chart(eval_data["y_true_grade"], eval_data["y_pred_grade"]),
                use_container_width=True,
            )
        with col2:
            st.plotly_chart(
                confusion_matrix_chart(eval_data["y_true_status"], eval_data["y_pred_status"]),
                use_container_width=True,
            )

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(pass_fail_pie(eval_data["pass_fail_counts"]), use_container_width=True)
        with col4:
            fi_df = get_feature_importance(reg_model)
            if not fi_df.empty:
                st.plotly_chart(feature_importance_chart(fi_df), use_container_width=True)
            else:
                st.info("Feature importance unavailable for this model type.")

# =========================================================
# PAGE: ABOUT MODEL
# =========================================================
elif page == "ℹ️ About Model":
    render_hero("About This Model", "Technical overview of the existing trained pipeline.")

    render_glass_card_open()
    st.markdown("""
##### 🧠 Model Architecture
- **Regression Model:** RandomForestRegressor — predicts `final_grade` (0–100)
- **Classification Model:** RandomForestClassifier (class-balanced) — predicts `Pass` / `Fail`
- **Preprocessing:** `ColumnTransformer` with `StandardScaler` (numeric) + `OneHotEncoder` (categorical), wrapped in a single scikit-learn `Pipeline`

##### 📥 Input Features
Age, Gender, Study Hours/Week, Attendance %, Previous Grade, Sleep Hours,
Parental Support, Extracurricular Activities, Internet Access, Part-Time Job

##### 📦 Files Used (unchanged from your original project)
- `models/regression_model.pkl`
- `models/classification_model.pkl`
- `outputs/metrics_report.txt`
- `data/student_performance.csv`

This dashboard only loads and serves your **already-trained** models —
no retraining occurs anywhere in this app.
    """)
    render_glass_card_close()

    if metrics["found"]:
        render_divider()
        render_section_title("📄 Raw Metrics Report")
        st.code(metrics["raw_text"], language="text")

# =========================================================
# PAGE: DOWNLOAD REPORT
# =========================================================
elif page == "📥 Download Report":
    render_hero("Download Reports", "Export your prediction results and model metrics.")

    render_glass_card_open()
    st.markdown("##### 📄 Last Single Prediction Report")
    if st.session_state.get("last_result"):
        pdf_bytes = generate_pdf_report(
            st.session_state.last_form,
            st.session_state.last_result,
            generate_recommendations(
                st.session_state.last_form,
                st.session_state.last_result["predicted_grade"],
                st.session_state.last_result["status"],
            ),
        )
        st.download_button(
            "📄 Download Latest Prediction (PDF)",
            data=pdf_bytes,
            file_name="student_prediction_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.info("Make a prediction on the Predict page first.")
    render_glass_card_close()

    st.write("")
    render_glass_card_open()
    st.markdown("##### 🗂️ Prediction History (CSV)")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        csv_buffer = io.StringIO()
        hist_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Download Prediction History (CSV)",
            data=csv_buffer.getvalue(),
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.info("No prediction history yet.")
    render_glass_card_close()

    st.write("")
    render_glass_card_open()
    st.markdown("##### 📊 Model Metrics Report (TXT)")
    if metrics["found"]:
        st.download_button(
            "⬇️ Download metrics_report.txt",
            data=metrics["raw_text"],
            file_name="metrics_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        st.info("metrics_report.txt not found in outputs/.")
    render_glass_card_close()
