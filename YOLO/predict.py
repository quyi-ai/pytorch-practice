from ultralytics import YOLO


def main():
    model = YOLO("/home/quyi/practice/pytorch/YOLO/runs/detect/character_yolo-2/weights/best.pt")

    model.predict(
        source="yolo_onnx/lllj_017.png",
        classes=[0],
        conf=0.5,
        save=True,
        name="character_predict"
    )


if __name__ == "__main__":
    main()