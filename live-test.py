"""
Use a live feed from the webcam to crop and classify faces.
"""


import sys, os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import cv2
import PIL

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    print(frame.shape)

    

    live_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(pretrained=False)

    num_in = model.fc.in_features
    model.fc = nn.Linear(num_in, 2)

    model.load_state_dict(torch.load("weights/21epochs-0.8312val-0509-1841.pt"))
    model = model.to(device)
    model.eval()
    

    x,y,w,h = (0,0,480,480)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x,y,w,h = faces[0]


        gray_crop = gray[y:y+h, x:x+w]
        gray3 = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)
        graypil = PIL.Image.fromarray(gray3)


        cv2.rectangle(gray3, (x,y), (x+w,y+h), (255,0,0), 2)
        gray3 = cv2.resize(gray3, (224,224))

        
        image = live_transforms(graypil)
        image = image.cuda()
        image = image.unsqueeze(0)

        pred = model(image)
        print(pred)
        val, ind = torch.max(pred,1)
        
        if ind == 0:
            cv2.putText(gray3, 'bad', (10, 214), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        else:
            cv2.putText(gray3, 'good', (10, 214), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        cv2.imshow('frame', gray3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    cap.release()
    cv2.destroyAllWindows()