from ultralytics import YOLO


def main():
    for i in range(1,4):

        model = YOLO(f"runs/detect/character_yolo-{i}/weights/best.pt")

        metrics = model.val(
            data="data.yaml",
            imgsz=640
        )

        print("mAP50-95:", metrics.box.map)
        print("mAP50:", metrics.box.map50)
        print("mAP75:", metrics.box.map75)


if __name__ == "__main__":
    main()