from pathlib import Path

import torch
from torchvision import datasets, transforms
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

images, labels = next(iter(dataloaders["train"]))

print("images shape:", images.shape)
print("labels shape:", labels.shape)
print("labels:", labels)