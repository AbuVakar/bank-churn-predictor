import os
from flask import Flask, request, render_template
import pandas as pd
import joblib
from utils.preprocessor import preprocess_input

app = Flask(__name__)

# Load model components
model = joblib.load("model/churn_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_names = joblib.load("model/feature_names.pkl")

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None

    if request.method == "POST":
        try:
            # 🎯 Match features from telecom dataset
            raw_input = {
                "gender": request.form["gender"],
                "SeniorCitizen": int(request.form["SeniorCitizen"]),
                "Partner": request.form["Partner"],
                "tenure": int(request.form["tenure"]),
                "MonthlyCharges": float(request.form["MonthlyCharges"]),
                "Contract": request.form["Contract"],
                "PaymentMethod": request.form["PaymentMethod"]
            }

            # Convert to DataFrame
            input_df = pd.DataFrame([raw_input])

            # Preprocess
            processed = preprocess_input(input_df)
            processed = processed.reindex(columns=feature_names, fill_value=0)
            scaled = scaler.transform(processed)

            # Predict
            result = model.predict(scaled)[0]
            prediction = "Will Churn 💔" if result == 1 else "Will Not Churn ✅"

        except Exception as e:
            prediction = f"❌ Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)