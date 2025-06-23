from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("modelo_knn.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        feats = np.array(data["features"], dtype=float).reshape(1, -1)

        pred = model.predict(feats)[0]  # ya es 'saltamontes' o 'chicharra'
        return jsonify({"prediction": pred})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
