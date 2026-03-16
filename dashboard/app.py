import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
import sqlite3
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# Paths
DB_PATH = os.path.join("database", "student_dropout.db")
MODEL_PATH = os.path.join("models", "dropout_model.pkl")

st.set_page_config(page_title="Student Dropout Dashboard", layout="wide")

# Load Data
@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM student_records;", conn)
    conn.close()
    return df

df = load_data()

# Load Model
model = joblib.load(MODEL_PATH)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "EDA", "Model Performance", "Prediction"])

# ------------------- OVERVIEW -------------------
if page == "Overview":
    st.title("🎓 Student Dropout Risk Monitoring System")

    st.write("Total Records:", len(df))
    dropout_rate = df["dropped_out"].mean() * 100
    st.write(f"Overall Dropout Rate: {dropout_rate:.2f}%")

# ------------------- EDA -------------------
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dropout by Stream")
        stream_data = df.groupby("stream")["dropped_out"].mean()
        fig, ax = plt.subplots()
        stream_data.plot(kind="bar", ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Dropout by Academic Year")
        year_data = df.groupby("academic_year")["dropped_out"].mean()
        fig2, ax2 = plt.subplots()
        year_data.plot(kind="bar", ax=ax2)
        st.pyplot(fig2)

# ------------------- MODEL PERFORMANCE -------------------
elif page == "Model Performance":
    st.title("📈 Model Performance")

    from src.preprocess import preprocess_data
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    X_train, X_test, y_train, y_test = preprocess_data()
    y_pred = model.predict(X_test)

    st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    st.write("Precision:", round(precision_score(y_test, y_pred), 4))
    st.write("Recall:", round(recall_score(y_test, y_pred), 4))
    st.write("F1 Score:", round(f1_score(y_test, y_pred), 4))

# ------------------- PREDICTION -------------------
elif page == "Prediction":
    st.title("🔮 Predict Dropout Risk")

    age = st.number_input("Age", 17, 30, 20)
    gender = st.selectbox("Gender", ["Male", "Female"])
    stream = st.selectbox("Stream", ["CSE", "ECE", "Mechanical", "Civil", "BBA", "BCom"])
    attendance = st.slider("Attendance %", 0, 100, 75)
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
    fees_paid = st.selectbox("Fees Paid %", [60, 75, 100])
    hostel = st.selectbox("Hostel Resident", [0, 1])
    distance = st.slider("Distance from Home (km)", 0, 50, 10)
    scholarship = st.selectbox("Scholarship", [0, 1])
    academic_year = st.number_input("Academic Year", 2020, 2030, 2024)

    if st.button("Predict"):
        input_df = pd.DataFrame([[
            age,
            1 if gender == "Male" else 0,
            ["CSE", "ECE", "Mechanical", "Civil", "BBA", "BCom"].index(stream),
            attendance,
            cgpa,
            fees_paid,
            hostel,
            distance,
            scholarship,
            academic_year
        ]], columns=[
            "age", "gender", "stream", "attendance",
            "cgpa", "fees_paid", "hostel",
            "distance_km", "scholarship", "academic_year"
        ])

        probability = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.error(f"High Dropout Risk ⚠️ (Probability: {probability:.2f})")
        else:
            st.success(f"Low Dropout Risk ✅ (Probability: {probability:.2f})")
