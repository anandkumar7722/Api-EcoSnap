import tensorflow as tf
import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

try:
    model = tf.keras.models.load_model("model/garbage_model.keras", compile=False)
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    raise RuntimeError("Model loading failed.")

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = img_array[np.newaxis, ...]
    return img_array.astype(np.float32)

def predict_image(image: Image.Image, top_k: int = 3) -> dict:
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)[0]
    logger.info(f"Raw model predictions: {predictions}")
    # Apply softmax to ensure probabilities
    predictions = tf.nn.softmax(predictions).numpy()
    logger.info(f"Softmaxed predictions: {predictions}")
    if np.any(np.isnan(predictions)):
        logger.error("NaN detected in predictions!")
    top_indices = predictions.argsort()[-top_k:][::-1]
    top_predictions = [
        {"predicted_class": class_names[i], "confidence": float(predictions[i])}
        for i in top_indices
    ]
    return {
        "top_predictions": top_predictions,
        "predicted_class": top_predictions[0]["predicted_class"],
        "confidence": top_predictions[0]["confidence"]
    }
