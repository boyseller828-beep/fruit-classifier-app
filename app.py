import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

#load model
model = tf.keras.models.load_model("model.h5")

classes = ["apple","banana","orange"]

st.title("🌶️Fruit Image Classifier")

#upload image
file = st.file_uploader("Upload an image ",type=["jpg","png","jpeg"])


if file is not None:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    #preprocess image
    img = image.resize((128,128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    #predict
    predictions = model.predict(img_array)
    result = classes[np.argmax(predictions)]
    st.write(f"Prediction: {result}")






































