from ultralytics import YOLO

# 1. 加载预训练 YOLO 检测模型
model = YOLO("yolo11n.pt")

# 2. 对图片做预测
# classes=[0] 表示只检测 person
results = model.predict(
    source="pytorch/YOLO/eva08_a.jpg",
    classes=[0],
    conf=0.25,
    save=True
)

# 3. 打印检测结果
for result in results:
    boxes = result.boxes

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()

        print("class id:", cls_id)
        print("confidence:", conf)
        print("box xyxy:", xyxy)
        print("-" * 30)