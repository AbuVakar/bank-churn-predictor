# 🏦 Bank Customer Churn Predictor using Machine Learning

A Machine Learning + Flask web app that predicts whether a **bank customer is likely to churn (leave the bank)** based on key financial and behavioral parameters like credit score, balance, tenure, and more.

## 🔗 Live Demo  
👉 [Click here to try it now](https://bank-churn-predictor-yltr.onrender.com)
![IProject Preview]:-static/project-preview.png
---

## 📌 Problem Statement

**Customer churn** is a major problem for banks, where losing a loyal customer increases the cost of acquiring new ones.  
This project helps predict churn based on patterns in the customer's financial behavior — allowing banks to **take early action and retain valuable clients.**

---

## 🚀 Features

- 📋 User-friendly web form to input customer details
- 🤖 ML model (trained on historical data) to predict churn
- 📈 Scaled & preprocessed features using `scikit-learn`
- 🔍 Live prediction: `Will Churn 💔` or `Will Not Churn ✅`
- 🎯 Deploy-ready Flask application

---

## 🧠 Tech Stack

| Technology        | Role                            |
|------------------|----------------------------------|
| Python 🐍         | Programming language             |
| Pandas / NumPy    | Data handling and preprocessing |
| Scikit-learn      | Model training & evaluation     |
| Flask             | Web app backend                 |
| HTML + CSS        | Frontend (user input form)      |
| Joblib            | Model saving/loading            |

---

## 📊 Model Inputs (Features Used)

| Feature           | Description                              |
|-------------------|-------------------------------------------|
| `CreditScore`     | Customer's credit rating (300-900)        |
| `Gender`          | Male = 1, Female = 0                      |
| `Age`             | Customer age in years                     |
| `Tenure`          | Years as a customer                       |
| `Balance`         | Account balance in ₹                      |
| `NumOfProducts`   | No. of bank products used (1 to 4)        |
| `IsActiveMember`  | Actively using account? (1=Yes, 0=No)     |

---

## 📦 How to Run Locally

1. **Clone this repo:**
   ```bash
   git clone https://github.com/your-username/bank-churn-predictor.git
   cd bank-churn-predictor
````

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**

   ```bash
   python app.py
   ```

4. **Visit in browser:**

   ```
   http://127.0.0.1:5000
   ```

---

## 🧪 Sample Inputs to Test

### 🔴 Will Churn 💔

```txt
Gender: Male (1)
Age: 45
Tenure: 1
CreditScore: 350
Balance: ₹75000
NumOfProducts: 1
IsActiveMember: No (0)
```

### 🟢 Will Not Churn ✅

```txt
Gender: Female (0)
Age: 35
Tenure: 7
CreditScore: 720
Balance: ₹30000
NumOfProducts: 2
IsActiveMember: Yes (1)
```

---

## 📈 Future Scope

* Add more features like location, salary, education, etc.
* Train model using Deep Learning (e.g. MLP, XGBoost)
* Deploy to Render / Railway / Heroku
* Add login/auth for business use
* Store results in database (e.g. MongoDB)

---

## 🙌 Author

**Abu Vakar**
Final-Year CSE Student | AI-ML Enthusiast
[🔗 GitHub](https://github.com/AbuVakar) | [🔗 LinkedIn](https://linkedin.com/in/AbuVakar)

---

## 📌 License

This project is licensed under the [MIT License](LICENSE).

```
