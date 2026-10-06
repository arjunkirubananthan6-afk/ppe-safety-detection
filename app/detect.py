from ultralytics import YOLO


# Load a pretrained YOLO model
model = YOLO("models/ppe_yolo26n.pt")


# Run object detection on our test image
results = model("data/test.jpg")


# Display the image with detected objects
results[0].show()