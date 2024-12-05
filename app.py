from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

# Load your saved model
model = tf.keras.models.load_model('model.h5')

# Define class labels based on the tumor types your model predicts
class_labels = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]  # Update according to your model's output

# Define a route for home page
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Check if an image file was uploaded
        if 'file' not in request.files:
            return "No file uploaded!"
        
        file = request.files['file']
        
        if file.filename == '':
            return "No selected file"
        
        # Save the file temporarily
        file_path = os.path.join("static", "uploaded_image.jpg")
        file.save(file_path)
        
        # Preprocess the image to fit model requirements
        image = load_img(file_path, target_size=(256, 256))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image /= 255.0  # Normalize

        # Make prediction
        predictions = model.predict(image)
        tumor_type = class_labels[np.argmax(predictions)]  # Get the highest-probability tumor type
        
        return render_template("index.html", result=tumor_type, img_path=file_path)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
