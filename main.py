from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import pipeline
import io

app = FastAPI(title="AI Image Detector")

# Allow Flutter app to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading AI image detector model...")

detector = pipeline(
    "image-classification",
    model="umm-maybe/AI-image-detector"
)

print("AI image detector model loaded successfully!")


@app.get("/")
def home():
    return {
        "message": "AI Image Detector API is running"
    }


@app.post("/verify")
async def verify_image(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        image_bytes = await file.read()

        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Run AI detection
        results = detector(image)

        # Get scores
        human_score = 0.0
        artificial_score = 0.0

        for result in results:
            label = result["label"].lower()
            score = result["score"]

            if label == "human":
                human_score = score

            elif label == "artificial":
                artificial_score = score

        # Decide result
        if artificial_score > human_score:
            prediction = "AI Generated"
            confidence = artificial_score
        else:
            prediction = "Likely Real"
            confidence = human_score

        return {
            "filename": file.filename,
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "human_score": round(human_score * 100, 2),
            "artificial_score": round(artificial_score * 100, 2),
            "message": "Image analyzed successfully"
        }

    except Exception as e:
        return {
            "error": str(e)
        }