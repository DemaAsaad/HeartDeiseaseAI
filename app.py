import joblib
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the KNN model and scaler

knn_model = joblib.load('Knn_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    # Collect form data
    features = [
        float(request.form['age']),
        float(request.form['sex']),
        float(request.form['chest_pain_type']),
        float(request.form['bp']),
        float(request.form['cholesterol']),
        float(request.form['fbs_over_120']),
        float(request.form['ekg_results']),
        float(request.form['max_hr']),
        float(request.form['exercise_angina']),
        float(request.form['st_depression']),
        float(request.form['slope_of_st']),
        float(request.form['number_of_vessels_fluro']),
        float(request.form['thallium'])
    ]

    # Convert features to numpy array and reshape for the scaler
    features_array = np.array(features).reshape(1, -1)

    # Scale the features
    features_scaled = scaler.transform(features_array)

    # Make a prediction
    prediction = knn_model.predict(features_scaled)

    # Return result
    result = 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'
    return render_template('index.html', prediction_text=f'النتيجة المتوقعة: {result}')


if __name__ == '__main__':
    app.run(debug=True)
