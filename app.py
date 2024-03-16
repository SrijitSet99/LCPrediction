import json
import pickle

from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load Lung Cancer model and scaler
lung_cancer_model = pickle.load(open('best_mlp.pkl', 'rb'))
lung_cancer_scaler = pickle.load(open('scaling.pkl', 'rb'))

# Load COPD model and scaler
copd_model = pickle.load(open('best_mlp1.pkl', 'rb'))
copd_scaler = pickle.load(open('scaling1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    condition = data['condition']

    if condition.lower() == 'lung cancer':
        model = lung_cancer_model
        scaler = lung_cancer_scaler
    elif condition.lower() == 'copd':
        model = copd_model
        scaler = copd_scaler
    else:
        return jsonify({'error': 'Invalid condition provided'})

    # Extract features from JSON data
    features = np.array(list(data.values())[1:]).reshape(1, -1)
    new_data = scaler.transform(features)
    output = model.predict(new_data)
    
    return jsonify({'prediction': output[0]})

@app.route('/predict', methods=['POST'])
def predict():
    condition = request.form['condition']
    data = {key: float(value) for key, value in request.form.items() if key != 'condition'}

    if condition.lower() == 'lung cancer':
        model = lung_cancer_model
        scaler = lung_cancer_scaler
    elif condition.lower() == 'copd':
        model = copd_model
        scaler = copd_scaler
    else:
        return render_template("home.html", prediction_text="Invalid condition provided")

    final_input = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = model.predict(final_input)[0]
    
    return render_template("home.html", prediction_text=f"{condition}{output}")


if __name__ == "__main__":
    app.run(debug=True)
