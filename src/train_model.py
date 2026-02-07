from src.preprocess import preprocess_data
from sklearn.linear_model import LogisticRegression
import joblib
import os

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_model():
    X_train, X_test, y_train, y_test = preprocess_data()

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("Logistic Regression trained successfully.")

    return model, X_test, y_test


if __name__ == "__main__":
    model, X_test, y_test = train_model()

    joblib.dump(model, os.path.join(MODEL_DIR, "dropout_model.pkl"))
    print("Model saved successfully.")
