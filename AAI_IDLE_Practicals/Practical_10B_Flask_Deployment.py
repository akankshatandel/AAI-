# PRACTICAL 10B: Simple local deployment API using Flask
# Install: python -m pip install flask scikit-learn joblib
# Run Practical_10A first. Then run this file and open the URL shown in the console.
# Test in a browser: http://127.0.0.1:5000/predict?values=5.1,3.5,1.4,0.2

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("iris_model.joblib")
class_names = ["setosa", "versicolor", "virginica"]

@app.route("/")
def home():
    return (
        "Iris prediction API is running. "
        "Use /predict?values=5.1,3.5,1.4,0.2"
    )

@app.route("/predict")
def predict():
    raw = request.args.get("values", "")
    try:
        values = [float(x) for x in raw.split(",")]
        if len(values) != 4:
            raise ValueError("Enter exactly 4 comma-separated numbers.")
        prediction = int(model.predict(np.array(values).reshape(1, -1))[0])
        return jsonify({
            "prediction": class_names[prediction],
            "input": values
        })
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

if __name__ == "__main__":
    app.run(debug=False)
