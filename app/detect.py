from ultralytics import YOLO
from safety import analyse_detections


model = YOLO("models/ppe_yolo26n.pt")

results = model("data/construction_test.jpg")

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

print("\n--- PPE SAFETY REPORT ---")

print(f"PPE items detected: {len(analysis['ppe_detected'])}")
print(f"Violations detected: {len(analysis['violations'])}")

if analysis["safe"]:
    print("Status: SAFE")
else:
    print("Status: PPE VIOLATION")

for violation in analysis["violations"]:
    print(
        f"- {violation['class']} "
        f"({violation['confidence']:.2f})"
    )


result.save(filename="data/detection_result.jpg")