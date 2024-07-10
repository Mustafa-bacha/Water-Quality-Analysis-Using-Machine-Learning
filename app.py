from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/get_prediction', methods=['POST'])
def get_prediction():
    try:
        # Get form data for all 9 features with specific names
        features = [
            float(request.form['ph']),
            float(request.form['hardness']),
            float(request.form['solids']),
            float(request.form['chloramines']),
            float(request.form['sulfate']),
            float(request.form['conductivity']),
            float(request.form['organic_carbon']),
            float(request.form['trihalomethanes']),
            float(request.form['turbidity'])
        ]

        # Scale the features using the loaded scaler
        scaled_features = scaler.transform([features])

        # Make the prediction
        prediction = model.predict(scaled_features)

        # Interpret the prediction
        if prediction[0] == 1:
            result = "Safe for drinking"
        else:
            result = "Not safe for drinking"

        # Return the prediction result
        return render_template('predict.html', prediction_text=f'Prediction: {result}')

    except ValueError as ve:
        return render_template('predict.html', prediction_text=f'Error: Invalid input data. {str(ve)}')
    except Exception as e:
        return render_template('predict.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
