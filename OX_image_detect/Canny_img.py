import cv2
import torch
import torchvision.transforms as transforms
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

def empty(a):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 150, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 255, 255, empty)

# Model
class OX_Model_CNN(nn.Module):
    def __init__(self):
        super(OX_Model_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        
        self.fc1 = nn.Linear(32*18*18, 512)
        self.fc2 = nn.Linear(512, 1)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32*18*18)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        
        return x
    
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
    
base_dir = 'C:/Users/Hong/source/repos/TicTacToe/OX_image_detect/'
predict_dir = os.path.join(base_dir, 'predict')
class_names = {0:'O', 1:'X'}

model = OX_Model_CNN()
model.load_state_dict(torch.load('OX_class_model.pth'))
model.eval()

while True:
    success, img = cap.read()
    
    imgBlur = cv2.GaussianBlur(img, (7,7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    
    cv2.imshow("Result", imgCanny)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        saved_img_path = os.path.join(predict_dir, 'saved_img.jpg')
        cv2.imwrite(saved_img_path, imgCanny)
        print("Screen shot!")
        
        saved_img = Image.open(saved_img_path).convert('L')
        saved_img = transform(saved_img)
        saved_img = saved_img.unsqueeze(0)
        
        with torch.no_grad():
            output = model(saved_img)
            _, predicted = torch.max(output, 1)
            predicted_label = class_names[predicted.item()]
            
        print("Predicted Label : ", predicted_label)