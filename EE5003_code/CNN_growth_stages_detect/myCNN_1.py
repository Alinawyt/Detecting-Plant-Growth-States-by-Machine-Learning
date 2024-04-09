# -*- encoding: utf8 -*-
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch import nn
import os
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


# PROJECT_PATH='/Users/alina./CODE/EE5003/program_final/CNN'
PROJECT_PATH='./'
DATA_TRAIN = os.path.join(PROJECT_PATH, "Lettuce_Growth_Stages_Database/train_filter")
DATA_TEST = os.path.join(PROJECT_PATH, "Lettuce_Growth_Stages_Database/test_filter")
# DATA_TRAIN = os.path.join(PROJECT_PATH, "CNN_database/train")
# DATA_TEST = os.path.join(PROJECT_PATH, "CNN_database/test")
# DATA_MODEL = os.path.join(PROJECT_PATH, "CNN_database/model")

class Net(nn.Module):
    # number of nodes: 20-50-500-26
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
        self.fc2 = nn.Linear(500, 3) # 26

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def get_transform():
    return transforms.Compose([
            # 32 x 32
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor() ,
            # Normalize
            transforms.Normalize(mean=[0.4, 0.4, 0.4],
                                 std=[0.2, 0.2, 0.2])
        ])

def get_dataset(batch_size=10, num_workers=1):
    data_transform = get_transform()
    # load train_dataset
    train_dataset = ImageFolder(root=DATA_TRAIN, transform=data_transform)
    # load test_dataset
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
    num_epochs = 50  # Adjust as needed %10
    for epoch in range(num_epochs):
        print(f"Epoch:{epoch + 1}/{num_epochs}")
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    # print(f"epoch:{num_epochs}\n")
    # save model
    model_path = os.path.join(PROJECT_PATH, "Detect_growth_stages.pth")
    torch.save(model.state_dict(), model_path)
    print('CNN model saved as Detect_growth_stages.pth!')