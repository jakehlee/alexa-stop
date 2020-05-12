import pandas as pd
import numpy as np
import cv2 as cv
import torch
from torchvision import datasets, transforms
import os
from sklearn.model_selection import train_test_split
import sys

train_file = pd.read_csv("train.csv")
allx = []
emotions = []
for index, row in train_file.iterrows():
    if index % 1000 == 0:
        print(index)
    x = np.array(list(map(int, row["pixels"].split(" ")))).reshape(48, 48, 1)
    allx.append(x)
    if row["emotion"] == 3 or row["emotion"] == 6:
        emotions.append(("1", index))
    else:
        emotions.append(("0", index))

train, intm, train_labels, intm_labels = train_test_split(allx, emotions, test_size=0.3, random_state=42)
dev, test, dev_labels, test_labels = train_test_split(intm, intm_labels, test_size=0.66, random_state=42)

# https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html
for i in range(len(train)):
    cv.imwrite("images/train/" + train_labels[i][0] + "/" + str(train_labels[i][1]) + ".png", train[i])
    img = cv.imread("images/train/" + train_labels[i][0] + "/" + str(train_labels[i][1]) + ".png", 1)
    cv.imwrite("images/train/" + train_labels[i][0] + "/" + str(train_labels[i][1]) + ".png", img)

for i in range(len(dev)):
    cv.imwrite("images/dev/" + dev_labels[i][0] + "/" + str(dev_labels[i][1]) + ".png", dev[i])
    img = cv.imread("images/dev/" + dev_labels[i][0] + "/" + str(dev_labels[i][1]) + ".png", 1)
    cv.imwrite("images/dev/" + dev_labels[i][0] + "/" + str(dev_labels[i][1]) + ".png", img)

for i in range(len(test)):
    cv.imwrite("images/test/" + test_labels[i][0] + "/" + str(test_labels[i][1]) + ".png", test[i])
    img = cv.imread("images/test/" + test_labels[i][0] + "/" + str(test_labels[i][1]) + ".png", 1)
    cv.imwrite("images/test/" + test_labels[i][0] + "/" + str(test_labels[i][1]) + ".png", img)