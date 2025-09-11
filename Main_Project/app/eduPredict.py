import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
from datetime import datetime
import time
import warnings

# Suppress version compatibility warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")
warnings.filterwarnings("ignore", category=UserWarning, module="pickle")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", message=".*If you are loading a serialized model.*", category=UserWarning)


st.set_page_config(
    page_title="Education Prediction", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎓"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    body { font-family: 'Inter', sans-serif; }

    /* Main background (dark theme default) */
    .main {
        background: linear-gradient(135deg, #222831 0%, #393E46 100%);
        padding: 1rem 2rem;
        color: #DFD0B8;
    }

    /* Light theme override */
    @media (prefers-color-scheme: light) {
        .main {
            background: linear-gradient(135deg, #f9f9f9 0%, #e5e5e5 50%, #dfd0b8 100%);
            color: #222831;
        }
    }

    /* Header */
    .app-header {
       background: linear-gradient(90deg, #222831, #393E46);
       color: #DFD0B8;
       padding: .2rem;
       border-radius: 12px;
       box-shadow: 0 8px 30px rgba(0,0,0,0.4);
       margin: 10px auto;
       width: 50rem;
       max-width: 90%;
       text-align: center;
    }
    
    /* Sub-Header */
    .sub-header {
       background: linear-gradient(90deg, #222831, #393E46);
       color: #DFD0B8;
       padding: .2rem;
       border-radius: 12px;
       box-shadow: 0 8px 30px rgba(0,0,0,0.4);
       margin: 5px auto;
       width: 50rem;
       max-width: 90%;
       text-align: center;
    }
            
    /* suggestion */
    .suggestion {
       background: linear-gradient(90deg, #222831, #393E46);
       color: #DFD0B8;
       padding: 2rem;
       border-radius: 12px;
       box-shadow: 0 8px 30px rgba(0,0,0,0.4);
       width: 50rem;
       max-width: 90%;
    }

    /* s1 */
    .s1 {
       background: linear-gradient(90deg, #222831, #393E46);
       color: #DFD0B8;
       padding: 1rem;
       border-radius: 12px;
       box-shadow: 0 8px 30px rgba(0,0,0,0.4);
       width: 50rem;
       max-width: 90%;
    }
            
    /* suggestion-success */
    .suggestion-success {
       background: #2B5A1D;
       color: #DFD0B8;
       padding: .5rem;
       border-radius: 12px;
       box-shadow: 0 8px 30px rgba(0,0,0,0.4);
       margin: 5px;
       width: 50rem;
       max-width: 90%;
    }
            
    /* Login card */
    .login-card {
        max-width: 420px; margin: 50px auto; text-align: center;
        padding: 2rem; border-radius: 15px;
        background: linear-gradient(90deg, #222831, #393E46);
        box-shadow: 0 10px 35px rgba(0,0,0,0.25);
        color: #222831;
    }
    .login-card h2 { margin-bottom: 1rem; }

    /* Notice */
    .notice {
        background: #948979; padding: 12px 16px; border-radius: 8px;
        border-left: 4px solid #393E46;
        margin: 1rem 0; color: #222831;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #222831, #393E46) !important;
        color: #DFD0B8 !important; border-radius: 999px !important;
        padding: 0.6rem 1.4rem !important; font-weight: 600;
        border: none !important; box-shadow: 0 6px 20px rgba(0,0,0,0.25) !important;
    }

    /* Inputs / textareas */
    .stTextInput>div>div>input,
    .stTextInput input,
    .stTextArea>div>textarea {
      max-width: 380px !important;
      width: 100% !important;
      margin: 0 auto !important;
      display: block !important;
      box-sizing: border-box;
      background: linear-gradient(90deg, #222831, #393E46); !important;
      color: #DFD0B8 !important;
      border-radius: 8px !important;
      border: 1px solid #948979 !important;
      padding: .5rem .8rem !important;
    }

    /* Make the Sign In button match inputs */
    .stButton>button {
      max-width: 380px !important;
      width: 100% !important;
      margin: 0.6rem auto 0 auto !important;
      display: block !important;
    }
            
    /* Footer */
    .app-footer {
       background: linear-gradient(90deg, #222831, #393E46);
       color: #DFD0B8;
       padding: .2rem;
       border-radius: 12px;
       box-shadow: 0 8px 30px rgba(0,0,0,0.4);
       margin: 10px auto;
       width: 50rem;
       max-width: 90%;
       text-align: center;
    }
            
    /* 1) Ensure tab labels are centered */
    div[data-baseweb="tab-list"] {
        justify-content: center !important;
        position: relative !important;
    }

    /* 2) Hide Streamlit's default full-width gray line / border (covers several possible DOM variants) */
    div[data-baseweb="tab-border"],
    div[data-baseweb="tab-list"] + hr,
    div[data-baseweb="tab-list"] ~ hr,
    div[data-baseweb="tabs"] hr,
    hr[data-testid="stHorizontalRule"],
    .css-1v3fvcr > hr,
    .stTabs > hr {
        border: none !important;
        display: none !important;
        box-shadow: none !important;
    }

    /* 3) Add a custom, centered thin line under the tabs (limited width) */
    div[data-baseweb="tab-list"]::after {
        content: "";
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        bottom: -10px;
        width: min(60%, 800px);
        height: 1px;
        background: rgba(255,255,255,0.12);
        pointer-events: none;
    }

    /* 4) Slightly reduce padding so the active tab underline doesn't overlap the custom line */
    div[data-baseweb="tab-list"] > div {
        padding-bottom: 0.35rem !important;
    }

    </style>
""", unsafe_allow_html=True)


# Header
st.markdown(f"""
<div class="app-header">
    <h1>🎓 Education Prediction</h1>
    <p>Data-driven insights for academic excellence</p>
    <small style="display:block; text-align:right; padding:10px;">
        Session up time: <b> {datetime.now().strftime("%b %d, %Y %H:%M")}</b>
    </small>
</div>
""", unsafe_allow_html=True)

auth_users = {
    "student": "$7ud3n7",
    "teacher": "73@ch3r",
    "counselor": "C0un$3l0r"
}

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    # Login form
  col1, col2, col3 = st.columns([1, 0.8, 1])   # keep login form centered
  with col2:
    username = st.text_input("Username", key="user", placeholder="Enter Username")
    password = st.text_input("Password", type="password", key="pass", placeholder="Enter Password")
    
    if st.button("Login"):
        if username in auth_users and password == auth_users[username]:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid credentials")

if st.session_state.logged_in:
    role = st.session_state.username
    
    role_names = {
        "student": "Student",
        "teacher": "Teacher",
        "counselor": "Counselor"
    }
    
    current_role_name = role_names.get(role, role.capitalize())
    
    st.markdown(f"""
    <div class="sub-header"'>
        <h3>Welcome {current_role_name}!</h3>
        <small style="display:block; text-align:right; padding:10px;">
           Session: {datetime.now().strftime('%H:%M')}
        </small>
    </div>
    """, unsafe_allow_html=True)
    
    spacer_left, col1, col2, spacer_right = st.columns([3, 2, 2, 3])

    with col1:
        if st.button("Feedback", use_container_width=True):
            feedback_url = "https://docs.google.com/forms/d/e/1FAIpQLSfk2z-KC01lh1jxSk9BXTxX7JMPYTHbwkRcZN38TcrjQTzzaw/viewform?usp=header"
            st.experimental_open_url(feedback_url)

    with col2:
        if st.button("Logout", use_container_width=True):
           st.session_state.logged_in = False
           st.rerun()

    try:
        df = pd.read_csv("data/edu_cleaned.csv")
        anomaly_detector_model = joblib.load("models/anomaly_detector_model.pkl")
        trend_model = joblib.load("models/trend_model.pkl")

        models_dir = "models"
        candidate_files = [
            ("Tuned Logistic Regression", os.path.join(models_dir, "tuned_logistic_regression_model.pkl")),
            ("Tuned Random Forest", os.path.join(models_dir, "tuned_random_forest_model.pkl")),
            ("Tuned XGBoost", os.path.join(models_dir, "tuned_xgboost_model.pkl")),
            ("Baseline Random Forest", os.path.join(models_dir, "rf_model.pkl")),
        ]

        available_models = {}
        for display_name, path in candidate_files:
            if os.path.exists(path):
                try:
                    available_models[display_name] = joblib.load(path)
                except Exception:
                    pass

        default_model_name = None
        tuned_report_path = os.path.join("reports", "model_comparison_tuned.csv")
        if os.path.exists(tuned_report_path):
            try:
                comp_df = pd.read_csv(tuned_report_path)
                if {"Model", "F1 Score"}.issubset(comp_df.columns):
                    best_row = comp_df.sort_values("F1 Score", ascending=False).iloc[0]
                    name_map = {
                        "Tuned Logistic Regression": "Tuned Logistic Regression",
                        "Tuned Random Forest": "Tuned Random Forest",
                        "Tuned XGBoost": "Tuned XGBoost",
                    }
                    best_name_in_report = str(best_row["Model"]).strip()
                    if best_name_in_report in name_map and name_map[best_name_in_report] in available_models:
                        default_model_name = name_map[best_name_in_report]
            except Exception:
                pass

        if default_model_name is None:
            # Force only baseline model
            if "Baseline Random Forest" in available_models:
                default_model_name = "Baseline Random Forest"


        models_loaded = len(available_models) > 0 and anomaly_detector_model is not None and trend_model is not None
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please ensure 'data/edu_cleaned.csv' and model files exist in the correct directories.")
        models_loaded = False

    if models_loaded:
        def rebuild_target(row):
            if row["Target_Graduate"] == 1:
                return "Graduate"
            elif row["Target_Enrolled"] == 1:
                return "Enrolled"
            else:
                return "Dropout"

        df["Grade"] = df.apply(rebuild_target, axis=1)

        if role == "student":
            tab1, tab2, tab3 = st.tabs(["My Academic Outlook", "My Learning Insights", "Learning Suggestions"])
        elif role == "teacher":
            tab1, tab2, tab3 = st.tabs(["Academic Snapshot", "Class Collective Insights", "Risk Indicators"])
        else:
            tab1, tab2, tab3 = st.tabs(["Student Evaluation", "Institutional Analytics", "Target Planning"])

        with tab1:
            col_left, col_center, col_right = st.columns([1, 3, 1])
            with col_center:
             
             if role == "student":
                    st.markdown("""
                    <div class='metric-card'>
                        <h3 style='color: #DFD0B8; margin-bottom: 1rem;'>My Academic Outlook</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    age = st.slider("My Age", 15, 50, 20, key="student_age")
                    admission_grade = st.slider("My Admission Grade", 0.0, 200.0, 120.0, key="student_admission")
                    gender = st.selectbox("Gender", ["male", "female"], key="student_gender")
                    scholarship = st.selectbox("Scholarship Status", ["yes", "no"], key="student_scholarship")
                    tuition_paid = st.selectbox("Tuition Fees Status", ["yes", "no"], key="student_tuition")
                    sem1_grade = st.slider("My 1st Semester Grade", 0.0, 20.0, 12.0, key="student_sem1")
                    sem2_grade = st.slider("My 2nd Semester Grade", 0.0, 20.0, 12.0, key="student_sem2")
                    unemployment = st.slider("Unemployment Rate", 0.0, 20.0, 7.5, key="student_unemployment")
                    inflation = st.slider("Inflation Rate", 0.0, 10.0, 3.0, key="student_inflation")
                    gdp = st.slider("GDP", 0.0, 200000.0, 100000.0, key="student_gdp")
                    
                    model_names = list(available_models.keys()) if models_loaded else []
                    selected_model_name = None
                    selected_model = None
                    
                    if "Baseline Random Forest" in available_models:
                        selected_model_name = "Baseline Random Forest"
                        selected_model = available_models[selected_model_name]
                    else:
                        st.error("Baseline Random Forest model not found. Please ensure rf_model.pkl exists.")
                        selected_model_name, selected_model = None, None


                    if st.button("Predict My Academic Future", use_container_width=True, key="student_predict"):
                        base_input = df.drop(columns=[col for col in df.columns if "Target" in col or col == "Grade"])
                        model_columns = base_input.columns.tolist()
                        input_template = pd.DataFrame(columns=model_columns)
                        defaults = {}
                        for col in model_columns:
                            try:
                                if pd.api.types.is_numeric_dtype(df[col]):
                                    defaults[col] = float(df[col].median())
                                else:
                                    mode_series = df[col].mode()
                                    defaults[col] = mode_series.iloc[0] if not mode_series.empty else df[col].iloc[0]
                            except Exception:
                                defaults[col] = 0
                        input_template.loc[0] = defaults

                        input_template["Age at enrollment"] = age
                        input_template["Admission grade"] = admission_grade
                        input_template["Gender"] = 1 if gender == "male" else 0
                        input_template["Scholarship holder"] = 1 if scholarship == "yes" else 0
                        input_template["Tuition fees up to date"] = 1 if tuition_paid == "yes" else 0
                        input_template["Curricular units 1st sem (grade)"] = sem1_grade
                        input_template["Curricular units 2nd sem (grade)"] = sem2_grade
                        input_template["Unemployment rate"] = unemployment
                        input_template["Inflation rate"] = inflation
                        input_template["GDP"] = gdp

                        input_template = input_template[model_columns]

                        for col in input_template.columns:
                            if pd.api.types.is_integer_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                                input_template[col] = np.round(input_template[col]).astype(df[col].dtype)
                        prediction = selected_model.predict(input_template)[0]
                        probabilities = selected_model.predict_proba(input_template)[0]
                        confidence = round(probabilities[prediction] * 100, 2)
                        label_map = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
                        result = label_map[prediction]

                        if anomaly_detector_model.predict(input_template)[0] == -1:
                            st.error("Irregularities in academic profile detected! Kindly meet with your advisor.")

                        if result == "Dropout":
                            st.markdown(f"""
                            <div class='prediction-card error-card'>
                                <h2>Risk Alert: {result}</h2>
                                <h3>Confidence: {confidence}%</h3>
                                <p>Immediate intervention recommended</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif sem1_grade < 8 or sem2_grade < 8:
                            st.markdown(f"""
                            <div class='prediction-card warning-card'>
                                <h2>Warning: {result}</h2>
                                <h3>Confidence: {confidence}%</h3>
                                <p>Focus on improving your grades</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class='prediction-card success-card'>
                                <h2>Great News: {result}</h2>
                                <h3>Confidence: {confidence}%</h3>
                                <p>Keep up the excellent work!</p>
                            </div>
                            """, unsafe_allow_html=True)

                        st.info(f"Confidence Score: {confidence}%")

                        sem2_pred = trend_model.predict([[sem1_grade]])[0]
                        st.info(f"Predicted Semester-2 Grade: {round(sem2_pred, 2)}")

                        report = f"""
                        ═══════════════════════════════════════
                        🎓 Education Prediction STUDENT REPORT
                        ═══════════════════════════════════════
                        
                        Role: Student
                        Model: {selected_model_name}
                        Prediction: {result}
                        Confidence: {confidence}%
                        Anomaly: {"Yes" if anomaly_detector_model.predict(input_template)[0] == -1 else "No"}
                        Predicted Semester-2 Grade: {round(sem2_pred, 2)}
                        
                        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        EduPredict - AI-Powered Academic Success Predictor
                        
                        ═══════════════════════════════════════
                        """
                        st.download_button("Download My Report", report, 
                                         file_name="student_academic_report.txt", use_container_width=True)

             elif role == "teacher":
                    st.markdown("""
                    <div class='metric-card'>
                        <h3 style='color: #DFD0B8; margin-bottom: 1rem;'>Academic Snapshot Tool</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    student_name = st.selectbox("Student Name", ["Student"], key="Student Name")
                    age = st.slider("Student Age", 17, 60, 22, key="teacher_age_1")
                    admission_grade = st.slider("Admission Grade", 0.0, 200.0, 120.0, key="teacher_admission_1")
                    gender = st.selectbox("Gender", ["male", "female"], key="teacher_gender_1")
                    scholarship = st.selectbox("Scholarship Status", ["yes", "no"], key="teacher_scholarship_1")
                    tuition_paid = st.selectbox("Tuition Status", ["yes", "no"], key="teacher_tuition_1")
                    sem1_grade = st.slider("1st Sem Grade", 0.0, 20.0, 12.0, key="teacher_sem1_1")
                    sem2_grade = st.slider("2nd Sem Grade", 0.0, 20.0, 12.0, key="teacher_sem2_1")
                    unemployment = st.slider("Unemployment Rate", 0.0, 20.0, 7.5, key="teacher_unemployment_1")
                    inflation = st.slider("Inflation Rate", 0.0, 10.0, 3.0, key="teacher_inflation_1")
                    gdp = st.slider("GDP", 0.0, 200000.0, 100000.0, key="teacher_gdp_1")
                    
                    model_names = list(available_models.keys()) if models_loaded else []
                    selected_model_name = None
                    selected_model = None

                    if "Baseline Random Forest" in available_models:
                        selected_model_name = "Baseline Random Forest"
                        selected_model = available_models[selected_model_name]
                    else:
                        st.error("Baseline Random Forest model not found. Please ensure rf_model.pkl exists.")
                        selected_model_name, selected_model = None, None

                    
                    if st.button("Analyze Student Performance", use_container_width=True, key="teacher_predict"):
                        base_input = df.drop(columns=[col for col in df.columns if "Target" in col or col == "Grade"])
                        model_columns = base_input.columns.tolist()
                        input_template = pd.DataFrame(columns=model_columns)
                        defaults = {}
                        for col in model_columns:
                            try:
                                if pd.api.types.is_numeric_dtype(df[col]):
                                    defaults[col] = float(df[col].median())
                                else:
                                    mode_series = df[col].mode()
                                    defaults[col] = mode_series.iloc[0] if not mode_series.empty else df[col].iloc[0]
                            except Exception:
                                defaults[col] = 0
                        input_template.loc[0] = defaults
        
                        input_template["Student Name"] = 1 if student_name == "Select Student Name" else 0
                        input_template["Age at enrollment"] = age
                        input_template["Admission grade"] = admission_grade
                        input_template["Gender"] = 1 if gender == "male" else 0
                        input_template["Scholarship holder"] = 1 if scholarship == "yes" else 0
                        input_template["Tuition fees up to date"] = 1 if tuition_paid == "yes" else 0
                        input_template["Curricular units 1st sem (grade)"] = sem1_grade
                        input_template["Curricular units 2nd sem (grade)"] = sem2_grade
                        input_template["Unemployment rate"] = unemployment
                        input_template["Inflation rate"] = inflation
                        input_template["GDP"] = gdp
        
                        input_template = input_template[model_columns]
        
                        for col in input_template.columns:
                            if pd.api.types.is_integer_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                                input_template[col] = np.round(input_template[col]).astype(df[col].dtype)
                        prediction = selected_model.predict(input_template)[0]
                        probabilities = selected_model.predict_proba(input_template)[0]
                        confidence = round(probabilities[prediction] * 100, 2)
                        label_map = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
                        result = label_map[prediction]
        
                        if anomaly_detector_model.predict(input_template)[0] == -1:
                            st.error("Unusual student profile detected! Requires special attention.")
        
                        if result == "Dropout":
                            st.markdown(f"""
                            <div class='prediction-card error-card'>
                                <h2>High Risk: {result}</h2>
                                <h3>Confidence: {confidence}%</h3>
                                <p>Immediate intervention required</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif sem1_grade < 8 or sem2_grade < 8:
                            st.markdown(f"""
                            <div class='prediction-card warning-card'>
                                <h2>Moderate Risk: {result}</h2>
                                <h3>Confidence: {confidence}%</h3>
                                <p>Additional support recommended</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class='prediction-card success-card'>
                                <h2>Good Progress: {result}</h2>
                                <h3>Confidence: {confidence}%</h3>
                                <p>Student is on track</p>
                            </div>
                            """, unsafe_allow_html=True)
        
                        st.info(f"Confidence Score: {confidence}%")
        
                        sem2_pred = trend_model.predict([[sem1_grade]])[0]
                        st.info(f"Predicted Semester-2 Grade: {round(sem2_pred, 2)}")
        
                        report = f"""
                        ═══════════════════════════════════════
                        🎓 Education Prediction TEACHER ANALYSIS REPORT
                        ═══════════════════════════════════════
                        
                        Role: Teacher
                        Model: {selected_model_name}
                        Student Prediction: {result}
                        Confidence: {confidence}%
                        Anomaly: {"Yes" if anomaly_detector_model.predict(input_template)[0] == -1 else "No"}
                        Predicted Sem-2 Grade: {round(sem2_pred, 2)}
                        
                        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        EduPredict - AI-Powered Academic Success Predictor
                        
                        ═══════════════════════════════════════
                        """
                        st.download_button("Download Analysis Report", report, 
                                         file_name="teacher_student_analysis.txt", use_container_width=True)
               
             else:
                    st.markdown("""
                    <div class='metric-card'>
                        <h3 style='color: #DFD0B8; margin-bottom: 1rem;'>Student Evaluation Tool</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    stud_ratio = st.slider("Percentage of Student", 0, 100, 30, key="Percentage of Student")
                    age = st.slider("Student Age", 17, 60, 22, key="counselor_age")
                    admission_grade = st.slider("Admission Grade", 0.0, 200.0, 120.0, key="counselor_admission")
                    gender = st.selectbox("Gender", ["male", "female"], key="counselor_gender")
                    scholarship = st.selectbox("Scholarship Status", ["yes", "no"], key="counselor_scholarship")
                    tuition_paid = st.selectbox("Tuition Status", ["yes", "no"], key="counselor_tuition")
                    sem1_grade = st.slider("1st Sem Grade", 0.0, 20.0, 12.0, key="counselor_sem1")
                    sem2_grade = st.slider("2nd Sem Grade", 0.0, 20.0, 12.0, key="counselor_sem2")
                    unemployment = st.slider("Unemployment Rate", 0.0, 20.0, 7.5, key="counselor_unemployment")
                    inflation = st.slider("Inflation Rate", 0.0, 10.0, 3.0, key="counselor_inflation")
                    gdp = st.slider("GDP", 0.0, 200000.0, 100000.0, key="counselor_gdp")
                    
                    model_names = list(available_models.keys()) if models_loaded else []
                    selected_model_name = None
                    selected_model = None
                    
                    if "Baseline Random Forest" in available_models:
                        selected_model_name = "Baseline Random Forest"
                        selected_model = available_models[selected_model_name]
                    else:
                        st.error("Baseline Random Forest model not found. Please ensure rf_model.pkl exists.")
                        selected_model_name, selected_model = None, None


                    if st.button("Assess Student Risk", use_container_width=True, key="counselor_predict"):
                        base_input = df.drop(columns=[col for col in df.columns if "Target" in col or col == "Grade"])
                        model_columns = base_input.columns.tolist()
                        input_template = pd.DataFrame(columns=model_columns)
                        defaults = {}
                        for col in model_columns:
                            try:
                                if pd.api.types.is_numeric_dtype(df[col]):
                                    defaults[col] = float(df[col].median())
                                else:
                                    mode_series = df[col].mode()
                                    defaults[col] = mode_series.iloc[0] if not mode_series.empty else df[col].iloc[0]
                            except Exception:
                                defaults[col] = 0
                        input_template.loc[0] = defaults

                        input_template["Percentage of Student"] = stud_ratio
                        input_template["Age at enrollment"] = age
                        input_template["Admission grade"] = admission_grade
                        input_template["Gender"] = 1 if gender == "male" else 0
                        input_template["Scholarship holder"] = 1 if scholarship == "yes" else 0
                        input_template["Tuition fees up to date"] = 1 if tuition_paid == "yes" else 0
                        input_template["Curricular units 1st sem (grade)"] = sem1_grade
                        input_template["Curricular units 2nd sem (grade)"] = sem2_grade
                        input_template["Unemployment rate"] = unemployment
                        input_template["Inflation rate"] = inflation
                        input_template["GDP"] = gdp

                        input_template = input_template[model_columns]

                        for col in input_template.columns:
                            if pd.api.types.is_integer_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                                input_template[col] = np.round(input_template[col]).astype(df[col].dtype)
                        prediction = selected_model.predict(input_template)[0]
                        probabilities = selected_model.predict_proba(input_template)[0]
                        confidence = round(probabilities[prediction] * 100, 2)
                        label_map = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
                        result = label_map[prediction]

                        if anomaly_detector_model.predict(input_template)[0] == -1:
                            st.error("Unusual student profile detected! Requires specialized intervention.")

                        if result == "Dropout":
                            st.markdown(f"""
                            <div class='prediction-card error-card'>
                                <h2>Critical Risk: {result}</h2>
                                <h3>Confidence: {confidence}%</h3>
                                <p>Immediate intervention plan needed</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif sem1_grade < 8 or sem2_grade < 8:
                            st.markdown(f"""
                            <div class='prediction-card warning-card'>
                                <h2>Moderate Risk: {result}</h2>
                                <h3>Confidence: {confidence}%</h3>
                                <p>Intervention strategy recommended</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class='prediction-card success-card'>
                                <h2>Low Risk: {result}</h2>
                                <h3>Confidence: {confidence}%</h3>
                                <p>Student is performing well</p>
                            </div>
                            """, unsafe_allow_html=True)

                        st.info(f"Confidence Score: {confidence}%")

                        sem2_pred = trend_model.predict([[sem1_grade]])[0]
                        st.info(f"Predicted Semester-2 Grade: {round(sem2_pred, 2)}")

                        report = f"""
                        ═══════════════════════════════════════
                        🎓 Education Prediction COUNSELOR ASSESSMENT REPORT
                        ═══════════════════════════════════════
                        
                        Role: Counselor
                        Model: {selected_model_name}
                        Student Assessment: {result}
                        Confidence: {confidence}%
                        Anomaly: {"Yes" if anomaly_detector_model.predict(input_template)[0] == -1 else "No"}
                        Predicted Sem-2 Grade: {round(sem2_pred, 2)}
                        
                        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        EduPredict - AI-Powered Academic Success Predictor
                        
                        ═══════════════════════════════════════
                        """
                        st.download_button("Download Assessment Report", report, 
                                         file_name="counselor_assessment.txt", use_container_width=True)

        with tab2:
            col_left, col_center, col_right = st.columns([1, 3, 1])
            with col_center:

             if role == "student":
                st.subheader(" My Learning Insights")
                
                st.markdown("#### My Performance vs Peers")
                fig = px.pie(df, names="Grade", title="How I Compare to Other Students",
                                   color_discrete_sequence=["#948979", '#DFD0B8', '#393E46'])
                st.plotly_chart(fig, use_container_width=True, key="student_tab2_performance_pie")
                
                st.markdown("#### My Grade Trends")
                fig = px.line(df[["Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"]].reset_index(),
                              title="My Semester-wise Grade Progress",
                                   color_discrete_sequence=["#948979", '#DFD0B8', '#393E46'])
                st.plotly_chart(fig, use_container_width=True, key="student_tab2_grade_trends")
                
                st.markdown("#### My Success Probability")
                fig = px.box(df, x="Grade", y="Admission grade", title="Admission Grade vs Success Rate",
                                   color_discrete_sequence=["#948979", '#DFD0B8', '#393E46'])
                st.plotly_chart(fig, use_container_width=True, key="student_tab2_success_probability")
                
                st.markdown("#### My Course Performance")
                if "Course" in df.columns:
                    fig_course = px.histogram(df, x="Course", color="Grade", barmode="group", title="Course-wise Performance",
                                   color_discrete_sequence=["#948979", '#DFD0B8', '#393E46'])
                    st.plotly_chart(fig_course, use_container_width=True, key="student_tab2_course_performance")
                
                st.markdown("#### My Age Group Analysis")
                df["Age Group"] = pd.cut(df["Age at enrollment"], bins=[16, 20, 25, 30, 40, 60],
                                 labels=["17–20", "21–25", "26–30", "31–40", "41+"])
                dropout_by_age = df[df["Grade"] == "Dropout"]["Age Group"].value_counts(normalize=True).sort_index()
                fig_age = px.bar(x=dropout_by_age.index, y=dropout_by_age.values * 100,
                         labels={"x": "Age Group", "y": "Dropout %"}, title="Dropout Percentage by Age Group",
                                   color_discrete_sequence=["#948979", '#DFD0B8', '#393E46'])
                st.plotly_chart(fig_age, use_container_width=True, key="student_tab2_age_analysis")
            
             elif role == "teacher":
                st.subheader("Class Collective Insights")
                
                st.markdown("#### Class Performance Distribution")
                fig = px.pie(df, names="Grade", title="Overall Class Performance Distribution",
                                   color_discrete_sequence=["#948979", '#DFD0B8', '#393E46'])
                st.plotly_chart(fig, use_container_width=True, key="teacher_tab2_class_distribution")
                
                st.markdown("#### Class Performance Trends")
                fig = px.line(df[["Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"]].reset_index(),
                              title="Class Performance Trends Over Time",
                                   color_discrete_sequence=["#948979", '#DFD0B8', '#393E46'])
                st.plotly_chart(fig, use_container_width=True, key="teacher_tab2_class_trends")
                
                st.markdown("#### Course-wise Performance")
                if "Course" in df.columns:
                    fig_course = px.histogram(df, x="Course", color="Grade", barmode="group", title="Course-wise Class Performance",
                                   color_discrete_sequence=["#948979", '#DFD0B8', '#393E46'])
                    st.plotly_chart(fig_course, use_container_width=True, key="teacher_tab2_course_performance")
                
                st.markdown("#### Gender Performance Analysis")
                fig = px.box(df, x="Gender", y="Admission grade", title="Admission Grade by Gender",
                                   color_discrete_sequence=["#948979", '#DFD0B8', '#393E46'])
                st.plotly_chart(fig, use_container_width=True, key="teacher_tab2_gender_analysis")
                
                st.markdown("#### Scholarship Impact Analysis")
                df["Scholarship Label"] = df["Scholarship holder"].map({1: "Yes", 0: "No"})
                grade_scholar = df.groupby("Scholarship Label")[["Curricular units 1st sem (grade)", 
                                                         "Curricular units 2nd sem (grade)"]].mean().reset_index()
                fig_scholar = px.bar(grade_scholar, x="Scholarship Label", y=["Curricular units 1st sem (grade)", 
                                                                       "Curricular units 2nd sem (grade)"],
                             barmode="group", title="Average Grades: Scholarship vs Non-Scholarship",
                                   color_discrete_sequence=["#948979", '#DFD0B8', '#393E46'])
                st.plotly_chart(fig_scholar, use_container_width=True, key="teacher_tab2_scholarship_analysis")
            
             else:
                st.subheader("Institutional Analytics")
                
                st.markdown("#### Student Risk Distribution")
                fig = px.pie(df, names="Grade", title="Overall Student Risk Distribution",
                                   color_discrete_sequence=["#948979", '#DFD0B8', '#393E46'])
                st.plotly_chart(fig, use_container_width=True, key="counselor_tab2_risk_distribution")
                
                st.markdown("#### Performance Trends for Intervention")
                fig = px.line(df[["Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"]].reset_index(),
                              title="Performance Trends for Intervention Planning",
                                   color_discrete_sequence=["#948979", '#DFD0B8', '#393E46'])
                st.plotly_chart(fig, use_container_width=True, key="counselor_tab2_performance_trends")
                
                st.markdown("#### Correlation Heatmap (Risk Factors)")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr().round(2)
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        color_continuous_scale="RdBu",
                        zmin=-1,
                        zmax=1,
                        title="Risk Factor Correlations",
                    )
                    st.plotly_chart(fig_corr, use_container_width=True, key="counselor_tab2_correlation_heatmap")
    
                st.markdown("#### 3D Performance Analysis")
                if all(c in df.columns for c in [
                    "Admission grade", "Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"
                ]):
                    fig_3d = px.scatter_3d(
                        df,
                        x="Admission grade",
                        y="Curricular units 1st sem (grade)",
                        z="Curricular units 2nd sem (grade)",
                        color="Grade",
                        title="3D Performance Analysis for Intervention Planning",
                        color_discrete_sequence=["#948979", '#DFD0B8', '#393E46']
                    )
                    fig_3d.update_traces(marker=dict(size=4, opacity=0.85), selector=dict(type='scatter3d'))
                    fig_3d.update_layout(scene=dict(
                        xaxis_title='Admission grade',
                        yaxis_title='Sem-1 grade',
                        zaxis_title='Sem-2 grade'
                    ))
                    st.plotly_chart(fig_3d, use_container_width=True, key="counselor_tab2_3d_analysis")
    
                st.markdown("#### Dropout Risk Heatmap")
                if "Age Group" in df.columns and "Scholarship Label" in df.columns:
                    dropout_df = df.copy()
                    dropout_df["is_dropout"] = (dropout_df["Grade"] == "Dropout").astype(int)
                    pivot = dropout_df.pivot_table(
                        index="Age Group", columns="Scholarship Label", values="is_dropout", aggfunc="mean", observed=False
                    ).reindex(index=["17–20", "21–25", "26–30", "31–40", "41+"], columns=["Yes", "No"]) * 100
                    fig_dropout_heat = px.imshow(
                        pivot,
                        text_auto=True,
                        color_continuous_scale="YlOrRd",
                        origin="upper",
                        labels=dict(color="Dropout %"),
                        title="Dropout Risk by Age Group × Scholarship",
                        color_discrete_sequence=["#948979", '#DFD0B8', '#393E46']
                    )
                    st.plotly_chart(fig_dropout_heat, use_container_width=True, key="counselor_tab2_dropout_heatmap")

        with tab3:
            col_left, col_center, col_right = st.columns([1, 3, 1])
            with col_center:
             if role == "student":
                st.subheader("Learning Suggestions")
                
                st.markdown("#### Personalized Study Tips")
                st.markdown(f"""
                <div class="suggestion">
                <h4 style="text-align: center; font-weight: bold;">Personalized Recommendations Based on Your Academic Profile:<h4/>
                
                **Study Strategies:**
                - Concentrate on strengthening your weaker subjects
                - Develop a consistent study schedule
                - Participate in study groups to enhance understanding
                - Apply active learning methods
                
                **Performance Tracking:**
                - Review your grades frequently
                - Set realistic goals for each semester
                - Ask for guidance when necessary
                - Ensure regular class attendance
                
                **Tips for Academic Success:**
                - Keep your coursework well-organized
                - Maintain open communication with your teachers
                - Utilize all available learning resources
                - Stay consistent and motivated
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Success Stories")
                success_stories = [
                    "Students with comparable profiles reached an 85% success rate",
                    "Maintaining steady study habits results in improved performance",
                    "Consistent attendance boosts grades by up to 15%",
                    "Requesting support early helps avoid academic challenges"
                ]

                
                for story in success_stories:
                 st.markdown(f'<div class="suggestion-success"> {story}</div>', unsafe_allow_html=True)
                
                st.markdown("#### Useful Resources")
                st.markdown("""
                - **Tutoring & Academic Support**
                - **Counseling & Wellness Services**
                - **Scholarships & Financial Aid**
                - **Career Guidance & Development**
                """)

                
             elif role == "teacher":
                st.subheader("Risk Alerts & Indicators")
                
                st.markdown("#### High-Risk Students")
                high_risk = df[df['Grade'] == 'Dropout']
                st.markdown(f"""**<div class="s1">{len(high_risk)} students identified as high-risk</div>**""", unsafe_allow_html=True)
                
                if len(high_risk) > 0:
                    st.dataframe(high_risk[['Age at enrollment', 'Admission grade', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']].head(10))
                
                st.markdown("#### Risk Factors Analysis")
                st.markdown(f""" 
                <div class="suggestion">
                <h4 style="text-align: center; font-weight: bold;">Key Risk Factors to Monitor:</h4>
                
                **Academic Indicators:**
                - Low admission grades
                - Declining semester results
                - Irregular attendance
                - Incomplete or missing assignments
                
                **Personal Factors:**
                - Financial difficulties
                - Family responsibilities
                - Health-related issues
                - Transportation problems
                
                **Intervention Strategies:**
                - Early alert systems
                - Develop personalized tailored support plans
                - Regular check-ins
                - Connect students to helpful resources
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Intervention Success Metrics")
                intervention_metrics = [
                    "Early interventions boost success rates by up to 40%",
                    "Consistent check-ins lower dropout risk by 25%",
                    "Customized support plans enhance retention by 30%",
                    "Resource referrals benefit 60% of students at risk"
                ]

                
                for metric in intervention_metrics:
                    st.info(f"{metric}")
                
             else:
                st.subheader("Target Planning")
                
                st.markdown("#### Strategic Intervention Framework")
                st.markdown(f"""
                
                <div class="suggestion">
                <h4 style="text-align: center; font-weight: bold;">Comprehensive Intervention Strategy:<h4/>
                
                **Tier 1 - High Risk (Immediate Action):**
                - Individual counseling sessions
                - Intensive academic support programs
                - Financial assistance review
                - Family involvement
                
                **Tier 2 - Moderate Risk (Preventive):**
                - Regular monitoring
                - Study skills and learning workshops
                - Peer mentoring programs
                - Structured academic advising
                
                **Tier 3 - Low Risk (Maintenance):**
                - Regular check-ins with advisors
                - Success celebration
                - Goal setting support
                - Easy access to institutional resources
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Intervention Effectiveness")
                intervention_data = {
                    "Early Intervention": "85% success rate",
                    "Academic Support Programs": "70% improvement",
                    "Financial Aid Assistance": "60% retention increase",
                    "Mental Health Support": "75% positive outcomes",
                    "Peer Mentoring": "65% engagement boost"
                }
                
                for intervention, success_rate in intervention_data.items():
                    st.metric(f" {intervention}", success_rate)
                
                st.markdown("#### Long-term Success Tracking")
                st.success("""
                **Institutional Success Metrics:**
                
                **Retention Rates:**
                - Overall retention improved by 25%
                - At-risk student retention up 40%
                - Graduation rates increased by 15%
                
                **Academic Performance:**
                - Average GPA improved by 0.3 points
                - Course completion rates improved by 20%
                - Student satisfaction levels increased by 30%
                """)

        st.markdown("""
        <div class='app-footer'>
            <p><strong>All rights resvered by EduPredict</strong></p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.error("Cannot proceed without loading the required models and data files.")
        st.info("Please check if these files exist:")
        st.code("""
         Project Structure:
        ├── data/
        │   └── edu_cleaned.csv
        ├── models/
        │   ├── rf_model.pkl
        │   ├── anomaly_detector_model.pkl
        │   └── trend_model.pkl
        └── app/
            └── eduPredict.py
        """)

else:
    st.markdown("""
    <div class='app-footer'>
            <h3>All Account Credentials</h3>
            <h5><b>Student Credentials</b></h5>
            <p><b>Username:</b> student</p>
            <p><b>Password:</b> $7ud3n7</p>
            <hr>
            <h5><b>Teacher Credentials</b></h5>
            <p><b>Username:</b> teacher</p>
            <p><b>Password:</b> 73@ch3r</p>
            <hr>
            <h5><b>Counselor Credentials</b></h5>
            <p><b>Username:</b> counselor</p>
            <p><b>Password:</b> C0un$3l0r</p>
        </div>
    """, unsafe_allow_html=True)
