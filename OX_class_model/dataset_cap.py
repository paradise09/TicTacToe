import cv2
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import transforms, datasets

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

def empty(a):
    pass

n=0

success, img = cap.read()
cv2.imshow("cam", img)

'''while True:
    success, img = cap.read()
    
    imgBlur = cv2.GaussianBlur(img, (7,7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("Result", img)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        print("Screen shot!")'''