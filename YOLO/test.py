from pathlib import Path
from ultralytics import YOLO


MODEL_DIRS = [
    "character_yolo",
    "character_yolo-2",
    "character_yolo-3",
    #"character_yolo-5",
]


def evaluate_model(run_name):
    weight_path = Path("runs") / "detect" / run_name / "weights" / "best.pt"

    if not weight_path.exists():
        print(f"[跳过] 找不到模型: {weight_path}")
        return None

    print("=" * 60)
    print(f"正在测试模型: {run_name}")
    print(f"权重路径: {weight_path}")

    model = YOLO(str(weight_path))

    metrics = model.val(
        data="data.yaml",
        split="test",
        imgsz=640,
        project="runs/detect",
        name=f"test_{run_name}",
        exist_ok=True,
    )

    result = {
        "model": run_name,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
    }

    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print(f"mAP50:     {result['mAP50']:.4f}")
    print(f"mAP50-95:  {result['mAP50-95']:.4f}")

    return result


def main():
    results = []

    for run_name in MODEL_DIRS:
        result = evaluate_model(run_name)
        if result is not None:
            results.append(result)

    print("\n最终对比结果")
    print("-" * 80)
    print(f"{'model':<20} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>10}")
    print("-" * 80)

    for r in results:
        print(
            f"{r['model']:<20} "
            f"{r['precision']:>8.4f} "
            f"{r['recall']:>8.4f} "
            f"{r['mAP50']:>8.4f} "
            f"{r['mAP50-95']:>10.4f}"
        )


if __name__ == "__main__":
    main()