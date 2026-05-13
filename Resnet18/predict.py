from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "checkpoints" / "best_resnet18.pth"

class_names = ["ants", "bees"]
num_classes = len(class_names)

weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

img_path = BASE_DIR / "data" / "hymenoptera_data" / "val" / "ants" / "10308379_1b6c72e180.jpg"

image = Image.open(img_path).convert("RGB")
image = transform(image)
image = image.unsqueeze(0)

with torch.no_grad():
    outputs = model(image)
    probs = torch.softmax(outputs, dim=1)
    confidence, pred_idx = torch.max(probs, 1)

print("预测类别:", class_names[pred_idx.item()])
print("置信度:", confidence.item())































