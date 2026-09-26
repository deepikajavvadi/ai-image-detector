from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import pipeline
import io

app = FastAPI(title="AI Image Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = None


def get_detector():
    global detector

    if detector is None:
        print("Loading lightweight AI image detector...")

        detector = pipeline(
            "image-classification",
            model="onnx-community/ai-image-detect-distilled-ONNX",
            device=-1
        )

        print("Lightweight AI image detector loaded!")

    return detector


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

        detector_model = get_detector()

        results = detector_model(image)

        real_score = 0.0
        fake_score = 0.0

        for result in results:
            label = result["label"].lower()
            score = float(result["score"])

            if label == "real":
                real_score = score

            elif label == "fake":
                fake_score = score

        if fake_score > real_score:
            prediction = "AI Generated"
            confidence = fake_score
        else:
            prediction = "Likely Real"
            confidence = real_score

        return {
            "filename": file.filename,
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "human_score": round(real_score * 100, 2),
            "artificial_score": round(fake_score * 100, 2),
            "message": "Image analyzed successfully"
        }

    except Exception as e:
        return {
            "error": str(e)
        }
