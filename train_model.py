import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv("data/churn.csv")

# Drop irrelevant columns
df.drop(["customerID"], axis=1, inplace=True)

# Encode categorical variables
le = LabelEncoder()
for column in df.select_dtypes(include=["object"]).columns:
    if column != "Churn":
        df[column] = le.fit_transform(df[column])
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Feature scaling
X = df.drop("Churn", axis=1)
y = df["Churn"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "model/churn_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(list(X.columns), "model/feature_names.pkl")

print("✅ Model trained and saved.")
