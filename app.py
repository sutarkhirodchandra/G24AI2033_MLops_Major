from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    x_val = [[data["X"]]]
    y_pred = model.predict(x_val)
    return jsonify({"prediction": float(y_pred[0])})

if __name__ == "__main__":
    app.run(debug=True)
