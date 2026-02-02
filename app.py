from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
model_path = 'customer_default_prediction_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None
    print(f"Warning: Model file '{model_path}' not found. Please train the model first.")

# Define label mappings (based on the notebook preprocessing)
HOME_OWNERSHIP_MAP = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
LOAN_INTENT_MAP = {
    'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 
    'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5
}
LOAN_GRADE_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
HISTORICAL_DEFAULT_MAP = {'N': 0, 'Y': 1}

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please train the model first.'
            }), 500
        
        # Get data from request
        data = request.json
        
        # Extract and validate features
        features = {
            'customer_age': float(data.get('customer_age', 0)),
            'customer_income': float(data.get('customer_income', 0)),
            'home_ownership': HOME_OWNERSHIP_MAP.get(data.get('home_ownership', 'RENT'), 0),
            'employment_duration': float(data.get('employment_duration', 0)),
            'loan_intent': LOAN_INTENT_MAP.get(data.get('loan_intent', 'PERSONAL'), 0),
            'loan_grade': LOAN_GRADE_MAP.get(data.get('loan_grade', 'A'), 0),
            'loan_amnt': float(data.get('loan_amnt', 0)),
            'loan_int_rate': float(data.get('loan_int_rate', 0)),
            'loan_percent_income': float(data.get('loan_percent_income', 0)),
            'historical_default': HISTORICAL_DEFAULT_MAP.get(data.get('historical_default', 'N'), 0),
            'credit_history_length': float(data.get('credit_history_length', 0))
        }
        
        # Create DataFrame with proper column order (as per training data)
        feature_array = np.array([[
            features['customer_age'],
            features['customer_income'],
            features['home_ownership'],
            features['employment_duration'],
            features['loan_intent'],
            features['loan_grade'],
            features['loan_amnt'],
            features['loan_int_rate'],
            features['loan_percent_income'],
            features['historical_default'],
            features['credit_history_length']
        ]])
        
        # Make prediction
        prediction = model.predict(feature_array)[0]
        prediction_proba = model.predict_proba(feature_array)[0]
        
        # Prepare response
        result = {
            'success': True,
            'prediction': int(prediction),
            'prediction_label': 'Default' if prediction == 1 else 'No Default',
            'confidence': {
                'no_default': float(prediction_proba[0] * 100),
                'default': float(prediction_proba[1] * 100)
            },
            'risk_level': get_risk_level(prediction_proba[1])
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

def get_risk_level(default_probability):
    """Determine risk level based on default probability"""
    if default_probability < 0.3:
        return 'Low Risk'
    elif default_probability < 0.6:
        return 'Medium Risk'
    elif default_probability < 0.8:
        return 'High Risk'
    else:
        return 'Very High Risk'

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


