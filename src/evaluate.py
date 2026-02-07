from src.preprocess import preprocess_data
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

MODEL_PATH = os.path.join("models", "dropout_model.pkl")


def evaluate_model():
    # Load model
    model = joblib.load(MODEL_PATH)

    # Get test data
    X_train, X_test, y_train, y_test = preprocess_data()

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Model Evaluation Results")
    print("------------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    evaluate_model()
