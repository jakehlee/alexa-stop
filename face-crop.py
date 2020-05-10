"""
Given a dataset, crop all of the faces and overwrite.
"""


import sys, os
import csv
import numpy as np
import cv2


target = 'OURdata'
subdirs = ['bad', 'good']



for d in subdirs:
    root = os.path.join(target, d)
    imgs = os.listdir(root)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    x,y,w,h = (0,0,224,224)
    for i in imgs:
        frame = cv2.imread(os.path.join(target, d, i))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x,y,w,h = faces[0]

        gray_crop = gray[y:y+h, x:x+w]
        gray3 = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(os.path.join(target, d, i), gray3)