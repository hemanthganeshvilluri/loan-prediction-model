from flask import Flask, request, render_template
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
class feature(BaseEstimator,TransformerMixin):
  def __init__(self):
    pass
  def fit(self,x,y=None):
    return self
  def transform(self,x):
    h=x.copy()
    if 'Loan_ID' in h.columns:
      h.drop('Loan_ID',axis=1,inplace=True)
    if 'ApplicantIncome' in h.columns and 'LoanAmount' in h.columns:
      h['Balance']=h['ApplicantIncome']-h['LoanAmount']
    if 'Loan_Status' in h.columns:
      h['Loan_status']=h['Loan_Status'].map({'Y':1,'N':0})
    if 'Dependents' in h.columns:
      h['Dependents'] = h['Dependents'].replace('3+', 3).astype(float)
    if 'ApplicantIncome' in h.columns and 'CoapplicantIncome' in h.columns:
      h['Total_Income']=h['ApplicantIncome']+h['CoapplicantIncome']
    return h
# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("loan_model.joblib")

# Define home route
@app.route('/')
def home():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect form data
        input_data = {
            "Gender": request.form['Gender'],
            "Married": request.form['Married'],
            "Dependents": request.form['Dependents'],
            "Education": request.form['Education'],
            "Self_Employed": request.form['Self_Employed'],
            "ApplicantIncome": float(request.form['ApplicantIncome']),
            "CoapplicantIncome": float(request.form['CoapplicantIncome']),
            "LoanAmount": float(request.form['LoanAmount']),
            "Loan_Amount_Term": float(request.form['Loan_Amount_Term']),
            "Credit_History": float(request.form['Credit_History']),
            "Property_Area": request.form['Property_Area']
        }

        # Convert into dataframe (to match training format)
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = model.predict(input_df)[0]

        result = "✅ Loan Approved" if prediction == 'Y' else "❌ Loan Rejected"

        return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0',port=port)


