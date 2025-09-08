from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

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
    app.run(debug=True)
