from datetime import datetime
from flask import Flask, render_template, request
import csv
import numpy as np
import pickle
import pandas as pd


app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load feature list
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

# Logging Function
def log_prediction(input_dict, proba, threshold, result):
    import os
    filename = "prediction_log.csv"
    log_fields = list(input_dict.keys()) + ["Probability", "Threshold", "Result", "Timestamp"]

    log_values = list(input_dict.values()) + [
        round(proba, 4),
        threshold,
        result,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ]

    # Check if file exists to write header
    file_exists = os.path.isfile(filename)

    try:
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(log_fields)
            writer.writerow(log_values)
    except Exception as e:
        print(f"Logging failed: {e}")
    
@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Load feature columns
        feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

        # Create zeroed input dict
        input_dict = {col: 0 for col in feature_columns}

        # Populate form values
        input_dict["TransactionAmt"] = float(request.form["TransactionAmt"])
        input_dict["oldbalanceOrg"] = float(request.form["oldbalanceOrg"])
        input_dict["newbalanceOrig"] = float(request.form["newbalanceOrig"])

        txn_type = request.form["type"].upper()
        type_col = f"type_{txn_type}"
        if type_col in input_dict:
            input_dict[type_col] = 1

        # Create DataFrame
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[feature_columns]  # Align with training features

        # Scale features
        input_scaled = scaler.transform(input_df)

        # 🔥 Predict probability of fraud (class = 1)
        proba = model.predict_proba(input_scaled)[0][1]   # 👈 Add this

        # 🔧 Apply custom threshold
        threshold = 0.3                                    # 👈 Tweak this as needed
        prediction = 1 if proba > threshold else 0         # 👈 Use probability instead of default predict()

        # Show result
        result = "⚠️ Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"

        # Log the prediction
        log_prediction(input_dict, proba, threshold, result)

        return render_template("result.html", prediction=result, proba=round(proba, 4), threshold=threshold)

    except Exception as e:
        return f"<h3>Error: {str(e)}</h3>"

if __name__ == "__main__":
    app.run(debug=True)
