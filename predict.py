import pandas as pd
import pickle
from utils.preprocessor import preprocess_input

# === Step 1: Load Trained Components ===
with open("model/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# === Step 2: Define Sample Input Data ===
sample_data = {
    "Gender": 1,
    "Age": 35,
    "Tenure": 5,
    "Balance": 70000.0,
    "NumOfProducts": 2,
    "CreditScore": 650,
    "IsActiveMember": 1
}

input_df = pd.DataFrame([sample_data])

# === Step 3: Preprocess ===
processed = preprocess_input(input_df)

# Align with training features
processed = processed.reindex(columns=feature_names, fill_value=0)

# === Step 4: Scale Input ===
scaled = scaler.transform(processed)

# === Step 5: Predict ===
result = model.predict(scaled)[0]

# === Step 6: Output ===
print("🔍 Prediction Result:")
print("Will Churn 💔" if result == 1 else "Will Not Churn ✅")
