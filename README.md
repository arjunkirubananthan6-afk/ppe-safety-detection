# PPE Safety Detection System

A computer vision project that detects PPE and safety violations in construction site images.

I trained a YOLO model to detect things like hardhats, safety vests, gloves and missing PPE. I then connected the model to a FastAPI backend where an image can be uploaded and analysed.

The API can return either the detection results as JSON or an annotated image with bounding boxes.

## What it does

The model detects 14 classes:

`Fall-Detected`, `Gloves`, `Goggles`, `Hardhat`, `Ladder`, `Mask`, `NO-Gloves`, `NO-Goggles`, `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`, `Person`, `Safety Cone`, `Safety Vest`

PPE detections are shown in green and violations such as `NO-Hardhat` are shown in red.

The general flow is:

```text
Image
  ↓
FastAPI
  ↓
YOLO model
  ↓
PPE detections
  ↓
Safety analysis
  ↓
JSON / Annotated image
```

## Model Training

I fine-tuned YOLO26n for 30 epochs using a PPE dataset containing construction site images.

- Training images: 30,765
- Validation images: 8,813
- Image size: 640
- Batch size: 8
- GPU: NVIDIA RTX 4050 Laptop GPU

Final validation results were approximately:

| Metric | Result |
| --- | --- |
| Precision | 0.71 |
| Recall | 0.82 |
| mAP@50 | 0.77 |
| mAP@50-95 | 0.49 |

The dataset is the Personal Protective Equipment Combined Model dataset from Roboflow Universe (Version 4, CC BY 4.0).

## Handling conflicting detections

One problem I ran into was the model sometimes predicting opposite classes in almost the same location.

For example:

```text
NO-Hardhat  0.42
Hardhat     0.41
```

To handle this, I added a simple post-processing step using IoU (Intersection over Union).

If two opposing classes have heavily overlapping bounding boxes, the detection with the higher confidence is kept.

## API

Start the server with:

```bash
uvicorn app.api:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

FastAPI's Swagger UI can be used to upload and test images.

The main endpoints are:

- `GET /health` - checks that the API is running
- `POST /detect` - returns detections and safety results as JSON
- `POST /detect/image` - returns the image with detection boxes drawn on it

Example `/detect` response:

```json
{
  "filename": "construction_test.jpg",
  "detections": [
    {
      "class": "NO-Hardhat",
      "confidence": 0.42,
      "box": [86, 68, 167, 145]
    }
  ],
  "ppe_items_detected": 0,
  "violations_detected": 1,
  "safe": false
}
```

## Running locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

The trained model should be placed at:

```text
models/ppe_yolo26n.pt
```

To run the API:

```bash
uvicorn app.api:app --reload
```

You can also run detection directly from the terminal:

```bash
python app/detect.py
```

## Tests

I added automated tests for the safety logic and API using pytest.

```bash
python -m pytest -v
```

Current result:

```text
6 passed
```

The tests cover PPE/violation classification, conflicting detections, API health checks and invalid file uploads.

## Project Structure

```text
app/
├── api.py
├── detect.py
├── safety.py
└── train.py

tests/
├── test_api.py
└── test_safety.py

models/
└── ppe_yolo26n.pt
```

The dataset, training runs and model weights are excluded from Git.

## Limitations

The model still struggles with some small or heavily occluded workers and can produce false positives or false negatives.

The current safety check is also image-level. It detects explicit missing-PPE classes, but it does not yet match every individual worker with all of their required PPE.

Some improvements I'd make in the future are better per-worker PPE matching, more training data for difficult examples and real-time video detection.

## Tech Used

Python, YOLO, PyTorch, OpenCV, FastAPI, Uvicorn and pytest.