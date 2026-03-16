from src.preprocess import preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_models():
    X_train, X_test, y_train, y_test = preprocess_data()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("\n", name)
        print("------------------------")
        print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
        print("Precision:", round(precision_score(y_test, y_pred), 4))
        print("Recall   :", round(recall_score(y_test, y_pred), 4))
        print("F1 Score :", round(f1_score(y_test, y_pred), 4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    evaluate_models()