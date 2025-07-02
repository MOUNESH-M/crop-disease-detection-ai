from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown

app = Flask(__name__)

MODEL_PATH = 'crop_disease_detector_model.keras'
GOOGLE_DRIVE_URL = 'https://drive.google.com/uc?id=1THCcQs9f-lBuZSiFtusSACyrO9z7LXBo'

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print('Downloading model from Google Drive...')
    gdown.download(GOOGLE_DRIVE_URL, MODEL_PATH, quiet=False, fuzzy=True)
    print('Model downloaded successfully!')

# Load the model
model = load_model(MODEL_PATH)

# Class names based on your dataset
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
    prediction = None
    confidence = None
    image_url = None
    error = None

    if request.method == 'POST':
        try:
            file = request.files['file']
            if file:
                file_path = 'static/uploads/' + file.filename
                os.makedirs('static/uploads/', exist_ok=True)
                file.save(file_path)
                image_url = '/' + file_path

                img = image.load_img(file_path, target_size=(128, 128))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                prediction_probs = model.predict(img_array)[0]
                predicted_index = np.argmax(prediction_probs)
                predicted_class = class_names[predicted_index]
                prediction = predicted_class
                confidence = f"{prediction_probs[predicted_index] * 100:.2f}%"

        except Exception as e:
            error = str(e)

    return render_template('index.html', prediction=prediction, confidence=confidence, image_url=image_url, error=error)


if __name__ == '__main__':
    app.run(debug=True)
