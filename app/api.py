from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np

from app.safety import analyse_detections


app = FastAPI(
    title="PPE Safety Detection API",
    description="Detects PPE and safety violations using a custom YOLO model.",
    version="1.0.0"
)

model = YOLO("models/ppe_yolo26n.pt")


@app.get("/")
def root():
    return {
        "message": "PPE Safety Detection API is running"
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image_array = np.frombuffer(image_bytes, np.uint8)

    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    results = model(image)

    result = results[0]

    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class": class_name,
            "confidence": round(confidence, 2),
            "box": [
                round(x1),
                round(y1),
                round(x2),
                round(y2)
            ]
        })

    analysis = analyse_detections(detections)

    return {
        "filename": file.filename,
        "detections": detections,
        "ppe_items_detected": len(analysis["ppe_detected"]),
        "violations_detected": len(analysis["violations"]),
        "safe": analysis["safe"],
        "violations": analysis["violations"]
    }