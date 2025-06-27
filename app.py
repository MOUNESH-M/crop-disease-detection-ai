from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load trained model
model = load_model('plant_disease_cnn_model.keras')

# Class labels (update with your trained dataset labels)
class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file selected.')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='Please select a file.')

        # Save file temporarily
        file_path = 'static/temp.jpg'
        file.save(file_path)

        # Preprocess image
        img = Image.open(file_path)
        img = img.resize((128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100

        return render_template('index.html', 
                               prediction=predicted_class, 
                               confidence=f"{confidence:.2f}%",
                               image_url=file_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
