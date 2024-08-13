from flask import Flask, render_template, request, jsonify
import requests
import json

app = Flask(__name__)

# Azure ML endpoint and API key
endpoint = 'http://7fe86a46-b7e0-46b4-bda2-62f00f53c05f.westeurope.azurecontainer.io/score'
api_key = 'kDeWysyETFuABvYhBmZNfcaggVk8CdUz'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        "Inputs": {
            "input1": [
                {
                    "gender": request.form['gender'],
                    "age": float(request.form['age']),
                    "hypertension": int(request.form['hypertension']),
                    "heart_disease": int(request.form['heart_disease']),
                    "ever_married": request.form['ever_married'],
                    "work_type": request.form['work_type'],
                    "Residence_type": request.form['Residence_type'],
                    "avg_glucose_level": float(request.form['avg_glucose_level']),
                    "bmi": float(request.form['bmi']),
                    "smoking_status": request.form['smoking_status']
                }
            ]
        },
        "GlobalParameters": {}
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    response = requests.post(endpoint, headers=headers, data=json.dumps(data))
    result = response.json()
    
    print(json.dumps(result, indent=4))

    scored_labels = result['Results']['WebServiceOutput0'][0]['Scored Labels']
    scored_probabilities = result['Results']['WebServiceOutput0'][0]['Scored Probabilities']

    prediction_text = "Stroke" if scored_labels == 1 else "No Stroke"
    
    return render_template('result.html', prediction=prediction_text, probability=scored_probabilities)


if __name__ == '__main__':
    app.run(debug=True)
