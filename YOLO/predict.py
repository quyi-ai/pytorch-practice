from ultralytics import YOLO


def main():
    model = YOLO("/home/quyi/practice/pytorch/YOLO/runs/detect/character_yolo-2/weights/best.pt")

    model.predict(
        source="bili_opus_images",
        conf=0.4,
        save=True,
        name="character_predict"
    )


if __name__ == "__main__":
    main()