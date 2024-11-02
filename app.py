from flask import Flask, request, jsonify, render_template, send_from_directory
import pickle
import pandas as pd
import os

app = Flask(__name__)

with open('crop_recommendation_model.pkl', 'rb') as f:
    model = pickle.load(f)
print("Model loaded successfully.")

IMAGE_DIR = 'static/images'

label_to_image = {
    "rice": "rice.jpg",
    "maize": "maize.jpeg",
    "chickpea": "chickpea.jpeg",
    "kidneybeans": "kidney_bean.jpg",
    "pigeonpeas": "pigeon_peas.jpg",
    "mothbeans": "mothbeans.jpg",
    "mungbean": "mungbean.jpg",
    "blackgram": "blackgram.jpg",
    "lentil": "lentil.jpg",
    "muskmelon": "muskmelon.jpg",
    "watermelon": "watermelon.jpg",
    "cotton": "cotton.jpg",
    "jute": "jute.jpg"
}

def prepare_input(temperature, humidity, ph, water_availability, season):
    new_data = {
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'water availability': water_availability,
        'season': season
    }
    new_data_df = pd.DataFrame([new_data])
    new_data_df = pd.get_dummies(new_data_df)
    new_data_df = new_data_df.reindex(columns=model.feature_names_in_, fill_value=0)
    return new_data_df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        water_availability = float(data['water_availability'])
        season = data['season']

        new_data_df = prepare_input(temperature, humidity, ph, water_availability, season)
        prediction = model.predict(new_data_df)[0]

        image_filename = label_to_image.get(prediction, "default.jpg")
        image_path = f'/images/{image_filename}'

        print(f"Response to client: {{'prediction': '{prediction}', 'image_path': '{image_path}'}}")

        return jsonify({'prediction': prediction, 'image_path': image_path})

    except ValueError as ve:
        print(f"Value Error: {ve}")
        return jsonify({'error': 'Invalid input data. Please check the values you provided.'}), 400
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True)
