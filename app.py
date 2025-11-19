import os
import io
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
import joblib

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

# -----------------------------
# FLASK APP
# -----------------------------
app = Flask(__name__, template_folder=TEMPLATE_DIR)

print("Starting Flask app. Template folder:", TEMPLATE_DIR)
print("Current working directory:", os.getcwd())

# -----------------------------
# LOAD MODEL AT STARTUP
# -----------------------------
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

print(" Loading saved model for Flask app...")
artifact = joblib.load(MODEL_PATH)

# If artifact is dict: extract model
if isinstance(artifact, dict):
    model = artifact.get("model")
else:
    model = artifact

# -----------------------------
# ROUTES
# -----------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")
        if not file:
            return render_template("upload.html", result="No file uploaded.")

        # Read and preprocess image (resize to 64x64 grayscale)
        img = Image.open(io.BytesIO(file.read())).convert("L").resize((64, 64))
        arr = np.asarray(img).reshape(1, -1)

        # Predict
        pred = model.predict(arr)[0]

        return render_template("upload.html", result=str(pred))

    except Exception as e:
        print(" Prediction error:", e)
        return render_template("upload.html", result=f"Error: {e}")


# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
