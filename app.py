from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model_path = 'E:\KARTHICK\project\karthick_project\my_model.h5'
model = tf.keras.models.load_model(model_path)

# Preprocess image
def preprocess_image(image):
    resized_image = cv2.resize(image, (150, 150))
    normalized_image = resized_image / 255.0
    preprocessed_image = np.expand_dims(normalized_image, axis=0)
    return preprocessed_image

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Read image file
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        # Preprocess image
        preprocessed_image = preprocess_image(image)
        # Make prediction
        prediction = model.predict(preprocessed_image)
        # Interpret prediction
        result = "Fake" if prediction[0][0] > 0.5 else "Real"
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
