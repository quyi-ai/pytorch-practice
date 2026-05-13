from pathlib import Path
import copy
import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "hymenoptera_data"
SAVE_DIR = BASE_DIR / "checkpoints"
SAVE_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 4
num_epochs = 5

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),
}

image_datasets = {
    phase: datasets.ImageFolder(
        DATA_DIR / phase,
        data_transforms[phase]
    )
    for phase in ["train", "val"]
}

dataloaders = {
    phase: torch.utils.data.DataLoader(
        image_datasets[phase],
        batch_size=BATCH_SIZE,
        shuffle=True if phase == "train" else False,
        num_workers=0
    )
    for phase in ["train", "val"]
}

class_names = image_datasets["train"].classes
num_classes = len(class_names)

print("train classes:", class_names)

weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# 冻结前面的预训练层，只训练最后分类层
for param in model.parameters():
    param.requires_grad = False

# 替换最后一层：1000 类 -> ants / bees 2 类
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.fc.parameters(),
    lr=0.001,
    momentum=0.9
)

best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("-" * 30)

    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for images, labels in dataloaders[phase]:
            optimizer.zero_grad()

            if phase == "train":
                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        print(f"{phase} loss: {epoch_loss:.4f}")
        print(f"{phase} acc: {epoch_acc.item():.4f}")

        if phase == "val" and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), SAVE_DIR / "best_resnet18.pth")
            print("saved best model")

    print()

model.load_state_dict(best_model_wts)
print(f"best val acc: {best_acc:.4f}")