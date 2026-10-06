from ultralytics import YOLO
from safety import analyse_detections


# Load our trained PPE detection model
model = YOLO("models/ppe_yolo26n.pt")


# Run detection on the test image
results = model("data/construction_test.jpg")
result = results[0]


# Convert YOLO predictions into a simpler format
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


# Analyse PPE detections and resolve conflicting predictions
analysis = analyse_detections(detections)


# Print safety report
print("\n--- PPE SAFETY REPORT ---")

print("\nDetections:")

for detection in analysis["detections"]:
    print(
        f"{detection['class']} | "
        f"Confidence: {detection['confidence']:.2f} | "
        f"Box: {detection['box']}"
    )


print()
print(f"PPE items detected: {len(analysis['ppe_detected'])}")
print(f"Violations detected: {len(analysis['violations'])}")


if analysis["safe"]:
    print("Status: SAFE")
else:
    print("Status: PPE VIOLATION")


if analysis["violations"]:
    print("\nViolations:")

    for violation in analysis["violations"]:
        print(
            f"- {violation['class']} "
            f"({violation['confidence']:.2f})"
        )


# Save image with YOLO bounding boxes
result.save(filename="data/detection_result.jpg")

print("\nDetection image saved to data/detection_result.jpg")