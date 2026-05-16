from ultralytics import YOLO


def main():


    model = YOLO("runs/detect/character_yolo-3/weights/best.pt")
    model.train(
            data="data.yaml",
            epochs=40,
            imgsz=640,
            batch=4,
            name="character_yolo"
        )


if __name__ == "__main__":
    main()