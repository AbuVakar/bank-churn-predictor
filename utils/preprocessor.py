import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Use same logic as in train_model.py
def preprocess_input(df):
    df = df.copy()

    # Encode categorical features (same as training)
    label_enc = LabelEncoder()

    # List of categorical columns based on training
    cat_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", 
                "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
                "TechSupport", "StreamingTV", "StreamingMovies", "Contract", 
                "PaperlessBilling", "PaymentMethod"]

    for col in cat_cols:
        if col in df.columns:
            df[col] = label_enc.fit_transform(df[col])

    return df
