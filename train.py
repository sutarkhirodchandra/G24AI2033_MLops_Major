from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

def main():
    print(" Loading Olivetti Faces dataset...")
    data = fetch_olivetti_faces()
    X, y = data.data, data.target

    print(" Splitting into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(" Training Decision Tree Classifier...")
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    print(" Saving model to model/savedmodel.pth...")
    os.makedirs("model", exist_ok=True)
    joblib.dump(
        {"model": clf, "X_test": X_test, "y_test": y_test},
        "model/savedmodel.pth"
    )

    print(" Training complete! Model saved successfully.")

if __name__ == "__main__":
    main()
