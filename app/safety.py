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


def analyse_detections(detections):
    ppe_detected = []
    violations = []

    for detection in detections:
        class_name = detection["class"]

        if class_name in PPE_PRESENT:
            ppe_detected.append(detection)

        elif class_name in PPE_VIOLATIONS:
            violations.append(detection)

    return {
        "ppe_detected": ppe_detected,
        "violations": violations,
        "safe": len(violations) == 0
    }