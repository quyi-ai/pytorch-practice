from pathlib import Path
import csv
from ultralytics import YOLO


MODEL_NAMES = [
    "character_yolo",
    "character_yolo-2",
    "character_yolo-3",
    "character_yolo-4",
]

DATA_YAML = "data.yaml"
IMG_SIZE = 640
SPLIT = "val"   # 验证集用 "val"，最终测试集才改成 "test"


def evaluate_model(model_name):
    weight_path = Path("runs") / "detect" / model_name / "weights" / "best.pt"

    if not weight_path.exists():
        print(f"[跳过] 找不到模型权重: {weight_path}")
        return None

    print("=" * 60)
    print(f"正在验证模型: {model_name}")
    print(f"权重路径: {weight_path}")

    model = YOLO(str(weight_path))

    metrics = model.val(
        data=DATA_YAML,
        split=SPLIT,
        imgsz=IMG_SIZE,
        project="runs/detect",
        name=f"val_{model_name}",
        exist_ok=True,
    )

    result = {
        "model": model_name,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
        "map50": metrics.box.map50,
        "map50_95": metrics.box.map,
    }

    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print(f"mAP50:     {result['map50']:.4f}")
    print(f"mAP50-95:  {result['map50_95']:.4f}")

    return result


def main():
    results = []

    for model_name in MODEL_NAMES:
        result = evaluate_model(model_name)

        if result is not None:
            results.append(result)

    print("\n模型验证结果汇总")
    print("-" * 75)
    print(f"{'model':<22} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>10}")
    print("-" * 75)

    for r in results:
        print(
            f"{r['model']:<22} "
            f"{r['precision']:>8.4f} "
            f"{r['recall']:>8.4f} "
            f"{r['map50']:>8.4f} "
            f"{r['map50_95']:>10.4f}"
        )

    with open("val_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "precision", "recall", "map50", "map50_95"]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n验证结果已保存到 val_results.csv")


if __name__ == "__main__":
    main()
