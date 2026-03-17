from src.preprocess import preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_models():
    X_train, X_test, y_train, y_test = preprocess_data()

    # Models
    log_model = LogisticRegression(max_iter=1000)
    rf_model = RandomForestClassifier()
    dt_model = DecisionTreeClassifier()

    # Train
    log_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)

    # Predict
    log_pred = log_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    dt_pred = dt_model.predict(X_test)

    # Accuracy
    log_acc = accuracy_score(y_test, log_pred)
    rf_acc = accuracy_score(y_test, rf_pred)
    dt_acc = accuracy_score(y_test, dt_pred)

    print("Model Comparison")
    print("----------------")
    print(f"Logistic Regression: {log_acc:.4f}")
    print(f"Random Forest      : {rf_acc:.4f}")
    print(f"Decision Tree      : {dt_acc:.4f}")

    # Select best model
    models = {
        "Logistic Regression": (log_model, log_acc),
        "Random Forest": (rf_model, rf_acc),
        "Decision Tree": (dt_model, dt_acc)
    }

    best_name = max(models, key=lambda x: models[x][1])
    best_model = models[best_name][0]

    print(f"\nBest Model Selected: {best_name}")

    # Save best model
    joblib.dump(best_model, os.path.join(MODEL_DIR, "dropout_model.pkl"))
    print("Best model saved successfully.")


if __name__ == "__main__":
    train_models()