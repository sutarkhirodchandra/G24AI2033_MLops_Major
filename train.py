# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# --- Step 1: Create simple data ---
data = {
    "X": [1, 2, 3, 4, 5],
    "Y": [2, 4, 6, 8, 10]
}
df = pd.DataFrame(data)

# --- Step 2: Train-test split ---
X = df[["X"]]
y = df["Y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 3: Train model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Step 4: Save model ---
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved as 'model.pkl'")
