from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)


model = joblib.load("model/hybrid_model.pkl")

scaler = model["scaler"]
rf = model["random_forest"]
gb = model["gradient_boosting"]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
      
        data = request.get_json()

        input_data = np.array([[
            data['age'], data['sex'], data['cp'], 
            data['trestbps'], data['chol'], data['fbs'],
            data['restecg'], data['thalach'], data['exang'], 
            data['oldpeak'], data['slope'], data['ca'], data['thal']
        ]])

  
        input_scaled = scaler.transform(input_data)

  
        rf_probs = rf.predict_proba(input_scaled)[:, 1].reshape(-1, 1)

       
        input_combined = np.hstack((input_scaled, rf_probs))

      
        prediction = gb.predict(input_combined)[0]
        
        result = "Heart Disease" if prediction == 1 else "No Heart Disease"
        
       
        return jsonify({'prediction': result})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Failed to predict'}), 500

@app.route('/positive')
def positive():
    return render_template('positive.html')

if __name__ == '__main__':
    app.run(debug=True)
    
