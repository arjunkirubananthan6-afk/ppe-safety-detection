from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "PPE Safety Detection API is running"
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model"] == "ppe_yolo26n"


def test_reject_non_image_file():
    response = client.post(
        "/detect",
        files={
            "file": (
                "test.txt",
                b"this is not an image",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "File must be an image"