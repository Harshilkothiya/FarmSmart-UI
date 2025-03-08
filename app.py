from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

model = joblib.load(open("./model.pkl", "rb"))

@app.route('/', methods=['GET'])
def hello():
    return jsonify({'message': 'Welcome to the Crop Prediction API!'})


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the request
        data = request.get_json()
        N = float(data.get('nitrogen', 0))
        P = float(data.get('phosphorus', 0))
        K = float(data.get('potassium', 0))
        temp = float(data.get('temperature', 0))
        humidity = float(data.get('humidity', 0))
        ph = float(data.get('ph', 0))
        rainfall = float(data.get('rainfall', 0))

        # Prepare data for prediction
        features = np.array([[N, P, K, temp, humidity, ph, rainfall]])

        # Perform prediction
        prediction = model.predict(features)[0]

        # Send response
        # print("this is ", prediction, )
        return jsonify({'prediction': prediction})

    except Exception as e:
        print("error")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
