from ultralytics import YOLO


def main():
    model = YOLO("yolo26n.pt")

    model.train(
        data="data/data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        workers=2,
        project="runs/ppe",
        name="yolo26n"
    )


if __name__ == "__main__":
    main()