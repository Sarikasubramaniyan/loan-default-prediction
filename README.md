# 🏦 Loan Default Prediction ML App

This is a machine learning-powered web application that predicts whether a loan applicant is likely to **default or not** based on personal, financial, and employment data. Built with **Python**, **Logistic Regression**, and deployed using **Streamlit**, this project is beginner-friendly and highly practical for finance-related risk assessment.

---

## 🚀 Features

- Predicts loan default risk in real-time
- User-friendly web interface using Streamlit
- Encodes categorical features automatically
- Saves trained model using Pickle
- Easy to deploy and demo

---

## 🧠 Machine Learning Details

- **Algorithm**: Logistic Regression  
- **Libraries**: pandas, scikit-learn, pickle, streamlit  
- **Model Inputs**:
  - Age, Income, Loan Amount, Credit Score
  - Employment Details, Loan Purpose, Marital Status, etc.

---

## 📂 Files in this Repo

| File               | Description                              |
|--------------------|------------------------------------------|
| `train_model.py`   | Trains logistic regression model         |
| `app.py`           | Streamlit frontend code                  |
| `loan_model.pkl`   | Trained ML model saved with Pickle       |
| `model_columns.pkl`| Column order for input consistency       |
| `data.csv`         | Sample dataset used for training         |
| `README.md`        | You're reading it now 😉                 |

---

## 💡 How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/loan-default-prediction.git
   cd loan-default-prediction

2. Install required packages:
    pip install -r requirements.txt

3. Train the model:
      python train_model.py
 
4. Run the app:
    streamlit run app.py

👩‍🎓 Created By
Sarika Subramaniyan.
