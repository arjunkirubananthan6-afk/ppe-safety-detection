from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from ultralytics import YOLO
import cv2
import numpy as np

from app.safety import analyse_detections


app = FastAPI(
    title="PPE Safety Detection API",
    description="Detects PPE and safety violations using a custom YOLO model.",
    version="1.0.0"
)

# Load the trained model once when the server starts
model = YOLO("models/ppe_yolo26n.pt")


def read_image(image_bytes):
    image_array = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def run_detection(image):
    result = model(image)[0]

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

    return analyse_detections(detections)


def draw_detections(image, detections):
    annotated_image = image.copy()

    for detection in detections:
        class_name = detection["class"]
        confidence = detection["confidence"]
        x1, y1, x2, y2 = detection["box"]

        # Red for PPE violations, green for everything else
        if class_name.startswith("NO-"):
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        label = f"{class_name} {confidence:.2f}"

        # Draw bounding box
        cv2.rectangle(
            annotated_image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # Draw label
        cv2.putText(
            annotated_image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return annotated_image


async def validate_upload(file):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    image = read_image(image_bytes)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image"
        )

    return image


@app.get("/")
def root():
    return {
        "message": "PPE Safety Detection API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "ppe_yolo26n"
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image = await validate_upload(file)

    analysis = run_detection(image)

    return {
        "filename": file.filename,
        "detections": analysis["detections"],
        "ppe_items_detected": len(analysis["ppe_detected"]),
        "violations_detected": len(analysis["violations"]),
        "safe": analysis["safe"],
        "violations": analysis["violations"]
    }


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    image = await validate_upload(file)

    analysis = run_detection(image)

    annotated_image = draw_detections(
        image,
        analysis["detections"]
    )

    success, encoded_image = cv2.imencode(
        ".jpg",
        annotated_image
    )

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to create annotated image"
        )

    return Response(
        content=encoded_image.tobytes(),
        media_type="image/jpeg"
    )