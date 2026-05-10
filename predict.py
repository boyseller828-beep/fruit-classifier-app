import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
model = tf.keras.models.load_model("model.h5")
img = image.load_img("data/banana/download.jpg", target_size=(128,128))
img = image.load_img("data/apple/images (1).jpg", target_size=(128,128))
img = image.load_img("data/orange/images (2).jpg", target_size=(128,128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
predictions = model.predict(img_array)

class_names = ["apple", "banana", "orange"]
print("Prediction:", class_names[np.argmax(predictions)])