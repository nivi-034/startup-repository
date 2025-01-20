import pickle
import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load the pre-trained model and scaler
with open('best_model_rf.pkl', 'rb') as rf_file:
    best_model_rf = pickle.load(rf_file)

scaler = StandardScaler()

# Initialize label encoder
label_encoder = LabelEncoder()

@app.route('/')
def home():
    return render_template('index.html')  # Show the input form for data

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        state_code = int(request.form['state_code'])
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        zip_code = request.form['zip_code']  # If required, ensure correct handling
        city = request.form['city']  # If needed, convert to numerical value or handle as category
        name = request.form['name']  # If needed, convert to numerical value or handle as category
        founded_at = int(request.form['founded_at'])
        closed_at = int(request.form['closed_at'])
        first_funding_at = int(request.form['first_funding_at'])
        last_funding_at = int(request.form['last_funding_at'])
        age_first_funding_year = int(request.form['age_first_funding_year'])
        age_last_funding_year = int(request.form['age_last_funding_year'])
        relationships = int(request.form['relationships'])
        funding_rounds = int(request.form['funding_rounds'])
        funding_total_usd = float(request.form['funding_total_usd'])
        milestones = int(request.form['milestones'])
        is_CA = int(request.form['is_CA'])
        is_NY = int(request.form['is_NY'])
        is_MA = int(request.form['is_MA'])
        is_TX = int(request.form['is_TX'])
        is_otherstate = int(request.form['is_otherstate'])
        category_code = request.form['category_code']  # Convert to appropriate format if needed
        avg_participants = float(request.form['avg_participants'])
        is_top500 = int(request.form['is_top500'])
        
        # Encoding city and name
        city_encoded = label_encoder.fit_transform([city])[0]  # Encoding city
        name_encoded = label_encoder.fit_transform([name])[0]  # Encoding name

        # Ensure that all 42 features are defined based on your model's training
        sample_data = np.array([state_code, latitude, longitude, zip_code, city_encoded, name_encoded, founded_at,
                                closed_at, first_funding_at, last_funding_at, age_first_funding_year,
                                age_last_funding_year, relationships, funding_rounds, funding_total_usd, milestones,
                                is_CA, is_NY, is_MA, is_TX, is_otherstate, category_code, avg_participants, is_top500,
                                # Placeholder values for features that might be missing
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Adding placeholders

    except ValueError:
        return "Invalid input. Please enter valid data."

    # Reshape data for prediction
    sample_data = sample_data.reshape(1, -1)

    # Apply the same scaling (ensure the scaler is fitted on all the features used in training)
    sample_scaled = scaler.fit_transform(sample_data)

    # Make prediction
    prediction = best_model_rf.predict(sample_scaled)

    # Convert prediction to human-readable result
    result = "Successful" if prediction == 1 else "Unsuccessful"

    # Return the prediction result to the user
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
