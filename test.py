# test.py
import joblib
from sklearn.metrics import accuracy_score

def main():
    obj = joblib.load("models/savedmodel.pth")
    model = obj["model"]
    X_test = obj["X_test"]
    y_test = obj["y_test"]
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("TEST ACCURACY:", acc)

if __name__ == "__main__":
    main()
