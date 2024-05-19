import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.optim import RMSprop
from torch.nn import CrossEntropyLoss

import os


base_dir = 'C:/images/OX_images/'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# training dataset Path
train_o_dir = os.path.join(train_dir, 'O')
train_x_dir = os.path.join(train_dir, 'X')
train_n_dir = os.path.join(train_dir, 'None')

# validation dataset Path
validation_o_dir = os.path.join(validation_dir, 'O')
validation_x_dir = os.path.join(validation_dir, 'X')
validation_n_dir = os.path.join(validation_dir, 'None')

# test dataset Path
test_o_dir = os.path.join(test_dir, 'O')
test_x_dir = os.path.join(test_dir, 'X')
test_n_dir = os.path.join(test_dir, 'None')


train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((150,150)),
    transforms.RandomRotation(25),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(150, scale=(0.8, 1.2), ratio=(0.75, 1.33)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

validation_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((150,150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

test_transforms = validation_transforms

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
validation_dataset = datasets.ImageFolder(validation_dir, transform=validation_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


class_indices = train_dataset.class_to_idx
print(class_indices)


class OX_Model_CNN(nn.Module):
    def __init__(self):
        super(OX_Model_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        
        self.fc1 = nn.Linear(32*18*18, 512)
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32*18*18)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
model = OX_Model_CNN()
print(model)


optimizer = RMSprop(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

epochs = 30
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        
    model.eval()
    validation_loss = 0.0
    accuracy = 0.0
    best_loss = 100.0
    
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            accuracy += (predicted == labels).float().mean().item()
            if validation_loss < best_loss:
                torch.save(model.state_dict(), 'OX_class_model.pth')
                best_loss = validation_loss
            
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}, "
          f"Validation Loss: {validation_loss / len(validation_loader)}, "
          f"Validation Accuracy: {accuracy / len(validation_loader)}")
    
    
torch.save(model.state_dict(), 'OX_class_model.pth')