这个项目实现了 YOLO ONNXRuntime 手写推理流程。
包括推理前处理和后处理。
核心流程包括 letterbox、preprocess、输出解析、置信度筛选、IoU、NMS、坐标还原和画框。

runs/detect/character_yolo-2是普通训练，yolo3是加了一些背景训练的结果