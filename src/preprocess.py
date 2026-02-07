import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

DB_PATH = os.path.join("database", "student_dropout.db")


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM student_records;", conn)
    conn.close()
    return df


def preprocess_data():
    df = load_data()

    # Drop ID column
    df = df.drop(columns=["student_id"])

    # Encode categorical variables
    le_gender = LabelEncoder()
    le_stream = LabelEncoder()

    df["gender"] = le_gender.fit_transform(df["gender"])
    df["stream"] = le_stream.fit_transform(df["stream"])

    X = df.drop(columns=["dropped_out"])
    y = df["dropped_out"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
