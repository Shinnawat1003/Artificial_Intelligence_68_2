import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

MODEL_PATH = "object_classifier.h5"
IMG_SIZE = (128, 128)

model = tf.keras.models.load_model(MODEL_PATH)
class_names = ['vaseline', 'black_box']

img_path = "D:/3/2/k/Artificial_Intelligence_68_2/ai/2.jpg"

img = image.load_img(img_path, target_size=IMG_SIZE, color_mode='grayscale')
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
pred_class = class_names[np.argmax(pred)]
confidence = np.max(pred) 

print(f"Prediction: {pred_class} ({confidence:.2f})")
