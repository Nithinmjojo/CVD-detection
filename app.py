# Flask app code (app.py)
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from flask import Flask, render_template, request
import numpy as np
import base64
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras_preprocessing.image import load_img



app = Flask(__name__,template_folder='templates')
model = load_model('model_vgg19_final.h5')  # Load your trained model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['image']
    if uploaded_file.filename != '':
      
        img_path = "static/"+ uploaded_file.filename
        uploaded_file.save(img_path)
        img = image.load_img(img_path, target_size=(192,128))  # Assuming input size of your model
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img/255.0  # Normalize the image

        result = model.predict(img)

        class_idx = np.argmax(result)
    
        index=['Left Bundle Branch block',
       'Normal',
       'Premature Atrial Contraction',
       'Premature Ventricular Contraction',
       'Right Bundle Branch Block',
       'Ventricular Fibrillation']
        result = str(index[class_idx])
    
        prediction = result

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
