# -*- encoding: utf8 -*-
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch import nn
import os
import torch

PROJECT_PATH = './'

DATA_TRAIN = os.path.join(PROJECT_PATH, "Abnormal_Leaves_Database/train")
DATA_TEST = os.path.join(PROJECT_PATH, "Abnormal_Leaves_Database/test")
# DATA_MODEL = os.path.join(PROJECT_PATH, "CNN_database/model")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layer
        self.fc1 = nn.Linear(50 * 5 * 5, 500)
        self.relu3 = nn.ReLU()

        # Output Layer
        self.fc2 = nn.Linear(500, 4)  # Assuming 4 classes

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def get_transform():
    return transforms.Compose([
        # transforms.Resize((32, 32)),  # Resize to 32x32
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
    ])

def get_dataset(batch_size=10, num_workers=1):
    data_transform = get_transform()
    # Load train_dataset
    train_dataset = ImageFolder(root=DATA_TRAIN, transform=data_transform)
    # Load test_dataset
    test_dataset = ImageFolder(root=DATA_TEST, transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, test_loader

if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = Net().to(DEVICE)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_dataloader, test_dataloader = get_dataset(batch_size=10)

    # Training loop
    num_epochs = 50  # Adjust as needed
    for epoch in range(num_epochs):
        print(f"Epoch:{epoch + 1}/{num_epochs}")
        model.train()  # Set the model to training mode
        # running_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        #     running_loss += loss.item() * inputs.size(0)
        
        # epoch_loss = running_loss / len(train_dataloader.dataset)
        # print(f"Training Loss: {epoch_loss:.4f}")

    model_path = os.path.join(PROJECT_PATH, "Detect_abnormal_leaves.pth")
    torch.save(model.state_dict(), model_path)
    print('CNN model saved as Detect_abnormal_leaves.pth!')

    # Evaluate the model
    model.eval() # 评估模式
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # print(f"predicted:{predicted}\n")
            # print(f"labels:{labels}\n")
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f'Final Classification Accuracy: {accuracy * 100:.2f}%')