from app.safety import analyse_detections


def test_ppe_detection_is_safe():
    detections = [
        {
            "class": "Hardhat",
            "confidence": 0.90,
            "box": [10, 10, 100, 100]
        },
        {
            "class": "Safety Vest",
            "confidence": 0.85,
            "box": [20, 20, 120, 150]
        }
    ]

    result = analyse_detections(detections)

    assert result["safe"] is True
    assert len(result["ppe_detected"]) == 2
    assert len(result["violations"]) == 0


def test_missing_ppe_is_violation():
    detections = [
        {
            "class": "NO-Hardhat",
            "confidence": 0.80,
            "box": [10, 10, 100, 100]
        }
    ]

    result = analyse_detections(detections)

    assert result["safe"] is False
    assert len(result["ppe_detected"]) == 0
    assert len(result["violations"]) == 1
    assert result["violations"][0]["class"] == "NO-Hardhat"


def test_conflicting_detection_keeps_higher_confidence():
    detections = [
        {
            "class": "NO-Hardhat",
            "confidence": 0.42,
            "box": [86, 68, 167, 145]
        },
        {
            "class": "Hardhat",
            "confidence": 0.41,
            "box": [87, 69, 166, 144]
        }
    ]

    result = analyse_detections(detections)

    assert len(result["detections"]) == 1
    assert result["detections"][0]["class"] == "NO-Hardhat"
    assert result["safe"] is False