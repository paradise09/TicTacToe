import torch.nn as nn
import torch.nn.functional as F

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