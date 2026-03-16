from src.preprocess import preprocess_data

X_train, X_test, y_train, y_test = preprocess_data()

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
