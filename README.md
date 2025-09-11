# EduPredict - Student Grade & Outcome Predictor

EduPredict is a machine learning–powered tool that predicts whether students are likely to **drop out, stay enrolled, or graduate**.  
It comes with an interactive **Streamlit dashboard** for visualization, insights, and reporting.

---

## Features

- Role-based login: Student, Teacher, Counselor  
- Predict student outcome: Dropout / Enrolled / Graduate  
- Interactive charts: Grade trends, dropout heatmap, scatter plots, violin plots, etc.  
- Anomaly detection for unusual cases  
- Next semester grade prediction  
- Downloadable reports (PDF/CSV)  
- Feedback integration with Google Forms  

---

## Getting Started

1. Clone the Repository
```
   git clone https://github.com/Developer-Elisha/EduPredict.git
```
```
   cd EduPredict/Main_Project
```

2. Install Requirements
```
   pip install -r requirements.txt
```
3. Run the App
```
   streamlit run app/eduPredict.py
```
4. Default Login Credentials
   Student → student / $7ud3n7
   Teacher → teacher / 73@ch3r
   Counselor → counselor / C0un$3l0r

---

## Project Structure
```
EduPredict/
│
├── Documentation/                 # User & Developer Documentation
│   ├── User_Documentation.pdf
│   └── Developer_Documentation.pdf
│
├── Feedback&Status/               # Feedback forms & project status reports
│   ├── ELISHA NOEL/
│   ├── MUHAMMAD SHEES/
│   ├── UTBAN ANWAR/
│   └── VINESH KUMAR/
│
├── GithubRepo/
│   └── Repo_&_Live_Link
│
├── Main_Project/
│   ├── app/
│   │   └── eduPredict.py
│   ├── data/
│   │   ├── edu_cleaned.csv
│   │   └── edu_raw.csv
│   ├── model/
│   │   ├── anomaly_detector_model.pkl
│   │   ├── rf_model.pkl
│   │   ├── trend_model.pkl
│   │   ├── tuned_logistic_regression_model.pkl
│   │   ├── tuned_random_forest_model.pkl
│   │   └── tuned_xgboost_model.pkl
│   ├── notebooks/
│   │   ├── AcademicED.ipynb
│   │   ├── Modeling.ipynb
│   │   ├── EDA_Report.html
│   │   ├── model_comparison.csv
│   │   └── model_comparison_tuned.csv
│
├── SnapShot/                      # Screenshots / UI previews
│
├── Video/                         # Demo video
│
└── README.md
```
---

## Dataset

- Source: UCI Machine Learning Repository  
- ~4500 student records  
- Target variable: Dropout / Enrolled / Graduate  

---

## Tech Stack

- Machine Learning: Python, scikit-learn, XGBoost  
- Visualization & UI: Streamlit, Plotly  
- Frontend Styling: HTML, CSS  

---

## Acknowledgements

- Our Faculty  
- Aptech Garden  
- E-Project Team  