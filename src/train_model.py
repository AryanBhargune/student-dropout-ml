from src.preprocess import preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_models():
    X_train, X_test, y_train, y_test = preprocess_data()

    # Initialize models
    logistic_model = LogisticRegression(max_iter=1000)
    rf_model = RandomForestClassifier()

    # Train models
    logistic_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    # Predictions
    log_pred = logistic_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    # Accuracy comparison
    log_acc = accuracy_score(y_test, log_pred)
    rf_acc = accuracy_score(y_test, rf_pred)

    print("Model Performance Comparison")
    print("-----------------------------")
    print(f"Logistic Regression Accuracy: {log_acc:.4f}")
    print(f"Random Forest Accuracy: {rf_acc:.4f}")

    # Select best model
    if rf_acc > log_acc:
        best_model = rf_model
        best_name = "Random Forest"
    else:
        best_model = logistic_model
        best_name = "Logistic Regression"

    print(f"\nBest Model Selected: {best_name}")

    # Save best model
    joblib.dump(best_model, os.path.join(MODEL_DIR, "dropout_model.pkl"))
    print("Best model saved successfully.")

    return best_model


if __name__ == "__main__":
    train_models()