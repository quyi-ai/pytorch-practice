import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
MODEL_PATH = Path(__file__).resolve().parent / "cifar10_cnn.pth"
DATA_DIR = Path(__file__).resolve().parent / "data"
BATCH_SIZE = 64
def load_data(data_dir=DATA_DIR, batch_size=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=False,
        transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=False,
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    classes = (
        "plane", "car", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    )

    return trainloader, testloader, classes
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x

def train_one_epoch(net, trainloader, criterion, optimizer, epoch):
    running_loss = 0.0

    net.train()

    for i, data in enumerate(trainloader):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:
            print(
                f"epoch {epoch + 1}, batch {i + 1}, "
                f"loss: {running_loss / 100:.3f}"
            )
            running_loss = 0.0
def evaluate(net, testloader):
    correct = 0
    total = 0

    net.eval()

    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy
def load_model(model_path=MODEL_PATH):
    net = Net()

    state_dict = torch.load(model_path, weights_only=True)

    net.load_state_dict(state_dict)

    return net
def main():
    trainloader, testloader, classes = load_data()

    net = Net()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        net.parameters(),
        lr=0.001,
        momentum=0.9
    )

    for epoch in range(2):
        train_one_epoch(net, trainloader, criterion, optimizer, epoch)

    print("Finished Training")

    accuracy = evaluate(net, testloader)

    print(f"Accuracy on test images: {accuracy:.2f}%")
    torch.save(net.state_dict(), MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")
    loaded_net = load_model()

    loaded_accuracy = evaluate(loaded_net, testloader)

    print(f"Loaded model accuracy: {loaded_accuracy:.2f}%")
if __name__ == "__main__":
    main()