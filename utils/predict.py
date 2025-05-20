import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model("model/garbage_model.keras", compile=False)

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = img_array[np.newaxis, ...]
    return img_array.astype(np.float32)

def predict_image(image: Image.Image) -> dict:
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)[0]  # first batch result
    max_index = np.argmax(predictions)
    predicted_class = class_names[max_index]
    confidence = float(predictions[max_index])
    return {"predicted_class": predicted_class, "confidence": confidence}
