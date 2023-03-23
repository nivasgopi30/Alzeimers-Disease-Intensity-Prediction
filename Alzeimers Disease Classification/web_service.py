import numpy as np
import tensorflow.lite as tflite
import streamlit as st
from PIL import Image

interpreter = tflite.Interpreter(model_path="alzeimer_disease_prediction.tflite")
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def load_img():
    image_file_uploaded = st.file_uploader("Upload the image", type=["png","jpg","jpeg"])
    with Image.open(image_file_uploaded) as image_file:
        image_file = image_file.resize((299, 299), Image.NEAREST)
    return image_file

def preprocess_input(x):
    x /= 127.5
    x -= 1
    return x

x = np.array(img, dtype=float32)
X = np.array([x])
X = preprocess_input(X)

interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)

classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

with final_prediction_of_alzeimer_prediction:
    st.write(f'Output is: {dict(zip(classes, preds[0]))}')