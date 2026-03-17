import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import sqlite3
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from src.preprocess import preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DB_PATH = os.path.join("database", "student_dropout.db")
MODEL_PATH = os.path.join("models", "dropout_model.pkl")

st.set_page_config(page_title="Student Dropout Dashboard", layout="wide")

@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM student_records;", conn)
    conn.close()
    return df

df = load_data()
model = joblib.load(MODEL_PATH)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Overview", "EDA", "Model Performance", "Prediction", "Add Student Data"]
)

# ---------------- OVERVIEW ----------------
if page == "Overview":

    st.title("🎓 Student Dropout Risk Monitoring System")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Students", len(df))

    with col2:
        dropout_rate = df["dropped_out"].mean() * 100
        st.metric("Dropout Rate", f"{dropout_rate:.2f}%")

# ---------------- EDA ----------------
elif page == "EDA":

    st.title("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        stream_data = df.groupby("stream")["dropped_out"].mean()
        fig, ax = plt.subplots()
        stream_data.plot(kind="bar", ax=ax)
        st.pyplot(fig)

    with col2:
        year_data = df.groupby("academic_year")["dropped_out"].mean()
        fig2, ax2 = plt.subplots()
        year_data.plot(kind="bar", ax=ax2)
        st.pyplot(fig2)

# ---------------- MODEL PERFORMANCE ----------------
elif page == "Model Performance":

    st.title("📈 Model Performance")

    X_train, X_test, y_train, y_test = preprocess_data()

    # Logistic
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    log_pred = log_model.predict(X_test)

    log_acc = accuracy_score(y_test, log_pred)
    log_prec = precision_score(y_test, log_pred)
    log_rec = recall_score(y_test, log_pred)
    log_f1 = f1_score(y_test, log_pred)

    # Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    rf_acc = accuracy_score(y_test, rf_pred)
    rf_prec = precision_score(y_test, rf_pred)
    rf_rec = recall_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)

    # Decision Tree
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)

    dt_acc = accuracy_score(y_test, dt_pred)
    dt_prec = precision_score(y_test, dt_pred)
    dt_rec = recall_score(y_test, dt_pred)
    dt_f1 = f1_score(y_test, dt_pred)

    comparison_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "Decision Tree"],
        "Accuracy": [log_acc, rf_acc, dt_acc],
        "Precision": [log_prec, rf_prec, dt_prec],
        "Recall": [log_rec, rf_rec, dt_rec],
        "F1 Score": [log_f1, rf_f1, dt_f1]
    })

    st.dataframe(comparison_df)

    fig, ax = plt.subplots()
    comparison_df.set_index("Model")["Accuracy"].plot(kind="bar", ax=ax)
    st.pyplot(fig)

    best_model = comparison_df.loc[comparison_df["Accuracy"].idxmax(), "Model"]
    st.success(f"Best Model: {best_model}")

# ---------------- PREDICTION ----------------
elif page == "Prediction":

    st.title("🔮 Predict Dropout Risk")

    age = st.number_input("Age", 17, 30, 20)
    gender = st.selectbox("Gender", ["Male", "Female"])
    stream = st.selectbox("Stream", ["CSE", "ECE", "Mechanical", "Civil", "BBA", "BCom"])
    attendance = st.slider("Attendance %", 0, 100, 75)
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
    fees_paid = st.selectbox("Fees Paid %", [60, 75, 100])
    hostel = st.selectbox("Hostel", [0, 1])
    distance = st.slider("Distance", 0, 50, 10)
    scholarship = st.selectbox("Scholarship", [0, 1])
    year = st.number_input("Year", 2020, 2030, 2024)

    if st.button("Predict"):

        input_df = pd.DataFrame([[
            age,
            1 if gender == "Male" else 0,
            ["CSE","ECE","Mechanical","Civil","BBA","BCom"].index(stream),
            attendance,
            cgpa,
            fees_paid,
            hostel,
            distance,
            scholarship,
            year
        ]], columns=[
            "age","gender","stream","attendance","cgpa",
            "fees_paid","hostel","distance_km","scholarship","academic_year"
        ])

        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        if pred == 1:
            st.error(f"High Risk ⚠️ ({prob:.2f})")
        else:
            st.success(f"Low Risk ✅ ({prob:.2f})")

# ---------------- ADD DATA ----------------
elif page == "Add Student Data":

    st.title("➕ Add Student")

    age = st.number_input("Age", 17, 30, 20)
    gender = st.selectbox("Gender", ["Male", "Female"])
    stream = st.selectbox("Stream", ["CSE","ECE","Mechanical","Civil","BBA","BCom"])
    attendance = st.slider("Attendance", 0, 100, 75)
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
    fees = st.selectbox("Fees", [60,75,100])
    hostel = st.selectbox("Hostel", [0,1])
    distance = st.slider("Distance", 0, 50, 10)
    scholarship = st.selectbox("Scholarship", [0,1])
    dropped = st.selectbox("Dropped Out", [0,1])
    year = st.number_input("Year", 2020, 2035, 2026)

    if st.button("Insert"):

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        cur.execute("""
        INSERT INTO student_records
        (age, gender, stream, attendance, cgpa, fees_paid, hostel, distance_km, scholarship, dropped_out, academic_year)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (age, gender, stream, attendance, cgpa, fees, hostel, distance, scholarship, dropped, year))

        conn.commit()
        conn.close()

        st.success("Inserted Successfully")