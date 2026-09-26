from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

app = FastAPI(title="AI Image Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "AI Image Detector API is running"
    }


@app.post("/verify")
async def verify_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        width, height = image.size

        return {
            "filename": file.filename,
            "prediction": "Image Received",
            "confidence": 100.0,
            "human_score": 0.0,
            "artificial_score": 0.0,
            "message": "Image uploaded successfully",
            "width": width,
            "height": height
        }

    except Exception as e:
        return {
            "error": str(e)
        }
