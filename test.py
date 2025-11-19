import joblib
from sklearn.metrics import accuracy_score

def main():
    print(" Loading saved model...")
    data = joblib.load("model/savedmodel.pth")

    model = data["model"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print(" Running predictions...")
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f" Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()