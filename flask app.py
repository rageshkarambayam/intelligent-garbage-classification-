from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

# Initialize the Flask app
app = Flask(_name_)

# Load the trained model
model = load_model('garbage_classification_model.h5')

# Define categories (update with your specific classes)
categories = ['Category1', 'Category2', 'Category3', 'Category4', 'Category5', 'Category6', 'Category7']

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join('static', 'uploaded_image.jpg')
        file.save(file_path)

        # Preprocess the image
        img = load_img(file_path, target_size=(150, 150))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = categories[np.argmax(prediction[0])]

        return render_template('index.html', prediction=predicted_class, image_path=file_path)

if _name_ == '_main_':
    app.run(debug=True)
