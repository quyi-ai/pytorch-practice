from pathlib import Path

import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

DATA_DIR = Path(__file__).resolve().parent / "data" / "hymenoptera_data"
BATCH_SIZE = 4
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
# train_dataset = datasets.ImageFolder(DATA_DIR / "train", data_transforms["train"])
# val_dataset = datasets.ImageFolder(DATA_DIR / "val", data_transforms["val"])
dataloaders = {
    phase: torch.utils.data.DataLoader(
        image_datasets[phase],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    for phase in ["train", "val"]
}
class_names = image_datasets["train"].classes

print("train classes:", class_names)


num_classes = 2
weights=ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
criterion=nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.fc.parameters(),
    lr=0.001,
    momentum=0.9
)
num_epochs = 5

for epoch in range(num_epochs):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for images, labels in dataloaders["train"]:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(image_datasets["train"])
    epoch_acc = running_corrects.double() / len(image_datasets["train"])

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"train loss: {epoch_loss:.4f}")
    print(f"train acc: {epoch_acc.item():.4f}")
    print("-" * 30)  