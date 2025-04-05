from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Load model and scaler safely
model_path = "svm_model.pkl"
scaler_path = "scaler.pkl"

if os.path.exists(model_path) and os.path.exists(scaler_path):
    svm_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
else:
    raise FileNotFoundError("Model or Scaler file is missing!")

# Define the feature names used during training
training_features = [
    'age', 'job',
    'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day_of_week', 'month',
    'duration', 'campaign', 'pdays', 'previous', 'poutcome',
    'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate input
        if not data:
            return jsonify({'error': 'Invalid input, JSON data is required'}), 400

        # Extract values from JSON input
        age = data.get('age', 0)
        job = data.get('job', 'unknown')
        marital = data.get('marital', 'unknown')
        education = data.get('education', 'unknown')
        default = data.get('default', 'no')
        housing = data.get('housing', 'no')
        loan = data.get('loan', 'no')

        # Default economic and campaign values
        duration = 700
        campaign = 1
        pdays = 999
        previous = 0
        poutcome = 'nonexistent'
        emp_var_rate = 1.4
        cons_price_idx = 93.918
        cons_conf_idx = -42.7
        euribor3m = 4.961
        nr_employed = 5228.1

        # Input dictionary
        input_data = {
            'age': [int(age)],
            'job': [job],
            'marital': [marital],
            'education': [education],
            'default': [default],
            'housing': [housing],
            'loan': [loan],
            'duration': [duration],
            'campaign': [campaign],
            'pdays': [pdays],
            'previous': [previous],
            'poutcome': [poutcome],
            'emp.var.rate': [emp_var_rate],
            'cons.price.idx': [cons_price_idx],
            'cons.conf.idx': [cons_conf_idx],
            'euribor3m': [euribor3m],
            'nr.employed': [nr_employed],
            'contact': [0],  # Default placeholders
            'month': [5],
            'day_of_week': [1]
        }

        # Convert to DataFrame
        df_input = pd.DataFrame(input_data)

        # One-Hot Encoding
        categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome']
        df_input_encoded = pd.get_dummies(df_input, columns=categorical_cols)

        # Align with training features
        df_input_encoded = df_input_encoded.reindex(columns=training_features, fill_value=0)

        # Ensure column order and type match the scaler
        df_input_encoded = df_input_encoded[scaler.feature_names_in_].astype(np.float64)

        # Scale input
        df_input_scaled = scaler.transform(df_input_encoded)

        # Make prediction
        prediction = svm_model.predict(df_input_scaled)

        # Convert prediction result
        predicted_subscription = 'yes' if prediction[0] == 1 else 'no'

        return jsonify({'result': predicted_subscription})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)