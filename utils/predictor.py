import pandas as pd
import pickle
from utils.preprocessor import preprocess_input

# Load saved model, scaler, and features
model = pickle.load(open("model/churn_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
feature_names = pickle.load(open("model/feature_names.pkl", "rb"))

def predict_churn(df):
    df = preprocess_input(df)
    df = df.reindex(columns=feature_names, fill_value=0)
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]
    return "Will Churn 💔" if prediction == 1 else "Will Not Churn ✅"
