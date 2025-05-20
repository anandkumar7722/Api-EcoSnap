from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf

app = FastAPI()

# Load model once
model = tf.keras.models.load_model("model/garbage_model.keras", compile=False)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = img_array[np.newaxis, ...]
    return img_array.astype(np.float32)

def predict_image(image: Image.Image):
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)[0]
    top_index = np.argmax(predictions)
    confidence = float(predictions[top_index])
    predicted_class = class_names[top_index]
    return {"predicted_class": predicted_class, "confidence": confidence}

@app.get("/api/health")
def health_check():
    return {"message": "Garbage Classifier API is running"}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image type")
    image_data = await file.read()
    image = Image.open(BytesIO(image_data)).convert("RGB")
    result = predict_image(image)
    return JSONResponse(content=result)

# Mount static files under /static
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
