# app.py

from flask import Flask, render_template, request
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier

app = Flask(__name__)

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Load the model from the correct path
model_path = os.path.join(script_dir, 'diabetes_model.pkl')
model = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the form data
        Pregnancies = float(request.form['Pregnancies'])
        Glucose = float(request.form['Glucose'])
        BloodPressure = float(request.form['BloodPressure'])
        SkinThickness = float(request.form['SkinThickness'])
        Insulin = float(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = float(request.form['Age'])

        # Make prediction
        prediction = model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        # Adjust the prediction message
        if prediction[0] == 1:
            prediction_msg = "Patient has diabetes! Hurry, consult a doctor."
        else:
            prediction_msg = "No diabetes detected. However, it's recommended to maintain a healthy lifestyle."

        # Display prediction
        return render_template('result.html', prediction=prediction_msg)

if __name__ == '__main__':
    app.run(debug=True)
