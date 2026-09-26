from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import json
import urllib.request
import numpy as np
import onnxruntime as ort

app = FastAPI(title="AI Image Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_URL = (
    "https://huggingface.co/onnx-community/"
    "ai-image-detect-distilled-ONNX/resolve/main/"
    "onnx/model_int8.onnx"
)

MODEL_PATH = "/tmp/model_int8.onnx"

session = None


def get_session():
    global session

    if session is None:
        if not os.path.exists(MODEL_PATH):
            print("Downloading lightweight AI detector model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        print("Loading ONNX model...")
        session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )
        print("ONNX model loaded!")

    return session


def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0

    image = (image - 0.5) / 0.5

    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return image.astype(np.float32)


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

        model = get_session()

        input_data = preprocess_image(image)

        input_name = model.get_inputs()[0].name

        outputs = model.run(
            None,
            {
                input_name: input_data
            }
        )

        logits = outputs[0][0]

        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        fake_score = float(probabilities[0])
        real_score = float(probabilities[1])

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
