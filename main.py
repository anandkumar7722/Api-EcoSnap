from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
from PIL import Image
from io import BytesIO
from utils.predict import predict_image  # Import prediction logic
import os

app = FastAPI()

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

# Optional: Serve frontend if you have one
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

port = int(os.environ.get("PORT", 10000))
