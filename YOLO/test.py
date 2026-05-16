from ultralytics import YOLO


def main():
    for i in range(1,4):


        model = YOLO(f"runs/detect/character_yolo-{i}/weights/best.pt")

        metrics = model.val(
            data="data.yaml",
            split="test",
            imgsz=640,
            name="final_test"
        )

        print("Precision:", metrics.box.mp)
        print("Recall:", metrics.box.mr)
        print("mAP50:", metrics.box.map50)
        print("mAP50-95:", metrics.box.map)


if __name__ == "__main__":
    main()

