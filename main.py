from flask import *  
from tensorflow.keras.utils import to_categorical
import numpy as np
from glob import glob
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import cv2
import json

# Load dog names
dog_names = []
with open('data/dog_names.json') as json_file:
    dog_names = json.load(json_file)

# Load ResNet50 models
ResNet50_model_for_dog_breed = ResNet50(weights='imagenet')
Res_model_for_adjusting_shape = ResNet50(weights='imagenet', include_top=False)

# Load bottleneck features
bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
train_Resnet = bottleneck_features['train']
valid_Resnet = bottleneck_features['valid']
test_Resnet = bottleneck_features['test']

# Build and load ResNet model
Resnet_Model = Sequential()
Resnet_Model.add(GlobalAveragePooling2D(input_shape=train_Resnet.shape[1:]))
Resnet_Model.add(Dense(133, activation='softmax'))
Resnet_Model.load_weights('saved_models/weights.best.Resnet.hdf5')

# Function to extract ResNet50 features
def extract_Resnet50(tensor):
    return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

# Function for pre-processing images into 4D tensor for CNN
def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

# Predict dog breed based on pretrained ResNet50 models
def ResNet50_predict_labels(img_path):
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model_for_dog_breed.predict(img))

# Returns "True" if a dog is detected in the image
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

# Returns "True" if face is detected in the image
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def Resnet_predict_breed(img_path):
    y = path_to_tensor(img_path)
    y = preprocess_input(y)
    x = Res_model_for_adjusting_shape.predict(y)
    predicted_vector = Resnet_Model.predict(x)
    return dog_names[np.argmax(predicted_vector)]

def get_correct_prenom(word, vowels):
    return "an" if word[0].lower() in vowels else "a"

def predict_image(img_path):
    vowels = ["a", "e", "i", "o", "u"]
    if dog_detector(img_path):
        predicted_breed = Resnet_predict_breed(img_path).rsplit('.', 1)[1].replace("_", " ")
        prenom = get_correct_prenom(predicted_breed, vowels)
        return f"The predicted dog breed is {prenom} {predicted_breed}."
    if face_detector(img_path):
        predicted_breed = Resnet_predict_breed(img_path).rsplit('.', 1)[1].replace("_", " ")
        prenom = get_correct_prenom(predicted_breed, vowels)
        return f"This photo looks like {prenom} {predicted_breed}."
    return "No human or dog could be detected, please provide another picture."

# Initialize Flask app
IMAGE_FOLDER = 'static/'
app = Flask(__name__)  
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

@app.route('/')  
def upload():
    return render_template("file_upload_form.html")  

@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        img_path = full_filename
        txt = predict_image(img_path)
        final_text = 'Results after Detecting Dog Breed in Input Image'
        return render_template("success.html", name=final_text, img=full_filename, out_1=txt)

@app.route('/info', methods=['POST'])  
def info():
    return render_template("info.html")  

if __name__ == '__main__':  
    app.run(host="127.0.0.1", port=8080, debug=True)  
