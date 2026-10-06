PPE_PRESENT = {
    "Hardhat",
    "Safety Vest",
    "Gloves",
    "Goggles",
    "Mask"
}

PPE_VIOLATIONS = {
    "NO-Hardhat",
    "NO-Safety Vest",
    "NO-Gloves",
    "NO-Goggles",
    "NO-Mask"
}

CONFLICTING_CLASSES = {
    "Hardhat": "NO-Hardhat",
    "NO-Hardhat": "Hardhat",
    "Safety Vest": "NO-Safety Vest",
    "NO-Safety Vest": "Safety Vest",
    "Gloves": "NO-Gloves",
    "NO-Gloves": "Gloves",
    "Goggles": "NO-Goggles",
    "NO-Goggles": "Goggles",
    "Mask": "NO-Mask",
    "NO-Mask": "Mask"
}


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection = intersection_width * intersection_height

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union


def resolve_conflicts(detections, iou_threshold=0.5):
    removed = set()

    for i, detection1 in enumerate(detections):
        if i in removed:
            continue

        conflicting_class = CONFLICTING_CLASSES.get(detection1["class"])

        if conflicting_class is None:
            continue

        for j in range(i + 1, len(detections)):
            if j in removed:
                continue

            detection2 = detections[j]

            if detection2["class"] != conflicting_class:
                continue

            iou = calculate_iou(
                detection1["box"],
                detection2["box"]
            )

            if iou >= iou_threshold:
                if detection1["confidence"] >= detection2["confidence"]:
                    removed.add(j)
                else:
                    removed.add(i)
                    break

    return [
        detection
        for index, detection in enumerate(detections)
        if index not in removed
    ]


def analyse_detections(detections):
    detections = resolve_conflicts(detections)

    ppe_detected = []
    violations = []

    for detection in detections:
        class_name = detection["class"]

        if class_name in PPE_PRESENT:
            ppe_detected.append(detection)

        elif class_name in PPE_VIOLATIONS:
            violations.append(detection)

    return {
        "detections": detections,
        "ppe_detected": ppe_detected,
        "violations": violations,
        "safe": len(violations) == 0
    }