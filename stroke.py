# Import necessary libraries
import numpy as np
import pickle
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler

# Load ML model
model = pickle.load(open('model.pkl', 'rb'))

# Create application
app = Flask(__name__, static_folder='static')

# Bind home function to URL
@app.route('/')
def home():
    return render_template('index.html')

# Bind form function to URL
@app.route('/form')
def form():
    return render_template('form.html')

# Bind predict function to URL
@app.route('/predict', methods=['POST'])
def predict():
    # Define mean and standard deviation for manual scaling
    mean_gender = 0.395887
    mean_age = 55.124832
    mean_hypertension = 0.2651410801383032
    mean_heart_disease = 0.08296424503251194

    std_gender = 0.487896
    std_age = 23.008714
    std_hypertension = 0.4375712297977711
    std_heart_disease = 0.26597857941210684

    # Extract form values and perform manual scaling
    features = [float(i) for i in request.form.values()]
    gender = (features[0] - mean_gender) / std_gender
    age = (features[1] - mean_age) / std_age
    heart_disease = (features[2] - mean_heart_disease) / std_heart_disease
    hypertension = (features[3] - mean_hypertension) / std_hypertension

    features_scaling = [gender, age, hypertension, heart_disease]

    # Reshape features for scaler
    features = np.array(features_scaling).reshape(-1, 1)
    features_2d = features.reshape(1, 4)

    # Fit the scaler
    scaler = StandardScaler()
    scaler.fit(features_2d)

    # Predict features using the model
    prediction = model.predict(features_2d)
    output = prediction

    # Check the output values and retrieve the result with HTML tags
    if output == 1:
        return render_template('negative.html', result='Pasien memiliki kemungkinan terkena stroke!')
    else:
        return render_template('positive.html', result='Pasien tidak memiliki kemungkinan terkena stroke!')


# Run the application
if __name__ == '__main__':
    app.run()
