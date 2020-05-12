import torchvision.models as models
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
import pandas as pd
import sys
import ast
import cv2 as cv
from google.colab.patches import cv2_imshow

# these commands are for a Colab Notebook

resnet50 = models.resnet50(pretrained=True, progress=True)

# make everything into actual images for the drive
train_file = pd.read_csv("train.csv")
for index, row in train_file.iterrows():
  x = np.array(list(map(int, row["pixels"].split(" ")))).reshape(48, 48, 1)
  cv.imwrite("/content/drive/My Drive/meld/images/" + str(index) + "_" + str(row["emotion"]) + ".png", x)

# we downloaded /images onto our local computer, then ran extract_faces.py
# then we upload the folder with the new splits back onto colab

# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#finetuning-the-convnet
# We use this tutorial below, approved by a TA

# https://pytorch.org/docs/stable/torchvision/models.html
# this is what the model expects, load our images into a dataloader
import os
from sklearn.model_selection import train_test_split
# get the tensors into train, dev, test array for batching later and do the preprocessing
# 70 (20096), 10 (2928), 20 (5685), total 28709

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'dev': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join("/content/images", x), data_transforms[x]) for x in ['train', 'dev', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4) for x in ['train', 'dev', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'dev', 'test']}
class_names = image_datasets['train'].classes


# finetune with our train, dev, test arrays above
# this saves the model with the best validation accuracy
def train_model(model, criterion, optimizer, num_epochs=1):
  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(num_epochs):
    since = time.time()
    print('Epoch {}/{}'.format(epoch, num_epochs-1))
    print('-'*10)

    # batches of 64
    for phase in ['train', 'dev']:
      if phase == 'train':
        model.train()
      else:
        model.eval()

      running_loss = 0.0
      running_corrects = 0

      for inputs, labels in dataloaders[phase]:
        if torch.cuda.is_available():
          inputs = inputs.to('cuda')
          labels = labels.to('cuda')
      
        optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(phase == "train"):
            # uncomment these to freeze the layers in the model for that experiment
          # for param in resnet50.layer1.parameters():
          #   param.requires_grad = False
          # for param in resnet50.layer2.parameters():
          #   param.requires_grad = False
          # for param in resnet50.layer3.parameters():
          #   param.requires_grad = False
          # for param in resnet50.layer4.parameters():
          #   param.requires_grad = False
          outputs = resnet50(inputs)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)

          if phase == "train":
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.double() / dataset_sizes[phase]
      print('Loss: {} Acc: {}'.format(epoch_loss, epoch_acc))
      new_time = time.time() - since
      print("Time: ", new_time)

      if phase == 'dev' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())

  print('Best val acc: {}'.format(best_acc))
  model.load_state_dict(best_model_wts)
  return model

# this is the next cell, we finetune the model here

resnet50 = models.resnet50(pretrained=True, progress=True)
features = resnet50.fc.in_features
resnet50.fc = nn.Linear(features, 2)

if torch.cuda.is_available():
  resnet50.to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(resnet50.parameters(), lr=0.001, momentum=0.9)

new_model = train_model(resnet50, criterion, optimizer_ft, num_epochs=10)

# https://pytorch.org/tutorials/beginner/saving_loading_models.html
torch.save(new_model.state_dict(), "/content/best_model.pt")

# now we work with MOSI, the above was with FER2013
# https://docs.python.org/3/library/urllib.request.html#module-urllib.request

import urllib.request
from urllib.request import urlretrieve

urlretrieve("http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSI.zip", "CMU_MOSI.zip")

# we go through all the videos and get their filenames

import os
import sys

new_file = open("files.txt", "w")

for root, dir, files in os.walk("/content/Raw/Video/Full"):
  for filex in files:
    new_file.write(filex + "\n")
new_file.close()

# then we run mosei.py for labels_file.csv
# https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html

import numpy as np
import cv2 as cv
import pandas as pd
import os
import sys

train = pd.read_csv("/content/labels_file.csv")
for index, row in train.iterrows():
  if index % 100 == 0:
    print(index)
  cap = cv.VideoCapture("/content/Raw/Video/Segmented/" + row["segment"] + ".mp4")
  count = 0
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
    else:
      if count % 30 == 0:
      # every 1 second, pick one frame
        cv.imwrite("/content/images/" + row["segment"] + "_" + str(count//30) + ".png", frame)
      count += 1
  cap.release()

# put images into the structure that dataloader needs
import pandas as pd
import os
import shutil
import glob
from mtcnn import MTCNN
import cv2 as cv
import matplotlib.pyplot as plt

labels_file = pd.read_csv("labels_file.csv")
allx = []
emotions = []
for index, row in labels_file.iterrows():
  allx.append(row["segment"])
  emotions.append(row["label"])

# they noted these as the splits in the dataset
train = ['2iD-tVS8NPw', '8d-gEyoeBzc', 'Qr1Ca94K55A', 'Ci-AH39fi3Y', '8qrpnFRGt2A', 'Bfr499ggo-0', 'QN9ZIUWUXsY', '9T9Hf74oK10', '7JsX8y1ysxY', '1iG0909rllw', 'Oz06ZWiO20M', 'BioHAh1qJAQ', '9c67fiY0wGQ', 'Iu2PFX3z_1s', 'Nzq88NnDkEk', 'Clx4VXItLTE', '9J25DZhivz8', 'Af8D0E4ZXaw', 'TvyZBvOMOTc', 'W8NXH0Djyww', '8OtFthrtaJM', '0h-zjBukYpk', 'Vj1wYRQjB-o', 'GWuJjcEuzt8', 'BI97DNYfe5I', 'PZ-lDQFboO8', '1DmNV9C1hbY', 'OQvJTdtJ2H4', 'I5y0__X72p0', '9qR7uwkblbs', 'G6GlGvlkxAQ', '6_0THN4chvY', 'Njd1F0vZSm4', 'BvYR0L6f2Ig', '03bSnISJMiM', 'Dg_0XKD0Mf4', '5W7Z1C_fDaE', 'VbQk4H8hgr0', 'G-xst2euQUc', 'MLal-t_vJPM', 'BXuRRbG0Ugk', 'LSi-o-IrDMs', 'Jkswaaud0hk', '2WGyTLYerpo', '6Egk_28TtTM', 'Sqr0AcuoNnk', 'POKffnXeBds', '73jzhE8R1TQ', 'OtBXNcAL_lE', 'HEsqda8_d0Q', 'VCslbP0mgZI', 'IumbAb8q2dM']
dev = ['WKA5OygbEKI', 'c5xsKMxpXnc', 'atnd_PF-Lbs', 'bvLlb-M3UXU', 'bOL9jKpeJRs', '_dI--eQ6qVU', 'ZAIRrfG22O0', 'X3j2zQgwYgE', 'aiEXnCPZubE', 'ZUXBRvtny7o']
test = ['tmZoasNr4rU', 'zhpQhgha_KU', 'lXPQBPVc5Cw', 'iiK8YX8oH1E', 'tStelxIAHjw', 'nzpVDcQ0ywM', 'etzxEpPuc6I', 'cW1FSBF59ik', 'd6hH302o4v8', 'k5Y_838nuGo', 'pLTX3ipuDJI', 'jUzDDGyPkXU', 'f_pcplsH_V0', 'yvsjCA6Y5Fc', 'nbWiPyCm4g0', 'rnaNMUZpvvg', 'wMbj6ajWbic', 'cM3Yna7AavY', 'yDtzw_Y-7RU', 'vyB00TXsimI', 'dq3Nf_lMPnE', 'phBUpBr1hSo', 'd3_k5Xpfmik', 'v0zCBqDeKcE', 'tIrG4oNLFzE', 'fvVhgmXxadc', 'ob23OKe5a9Q', 'cXypl4FnoZo', 'vvZ4IcEtiZc', 'f9O3YtZ2VfI', 'c7UH_rxdZv4']

# https://pypi.org/project/mtcnn/
# This is an open source implementation of the MTCNN face detector, under the MIT license
detector = MTCNN()

no_box = []

for i in range(len(train)):
  for filex in glob.glob("/content/images/" + train[i] + "_*.png"):
    new_file = filex.split("/")[-1]
    # use the bounding boxes here
    image = cv.imread(filex)
    bounding_boxes = detector.detect_faces(image)
    bounding_boxes = sorted(bounding_boxes, key=lambda x:x["confidence"])
    try:
      crop_place = bounding_boxes[0]["box"]
      image = image[crop_place[1]:(crop_place[1] + crop_place[3]), crop_place[0]:(crop_place[0] + crop_place[2])]
      index = allx.index("_".join(new_file.split(".")[0].split("_")[:-1]))
      if len(image) != 0:
        cv.imwrite("/content/mosi_images/train/" + str(emotions[index]) + "/" + new_file, image)
    except IndexError:
      no_box.append(new_file)

print('finished train')

for i in range(len(dev)):
  for filex in glob.glob("/content/images/" + dev[i] + "_*.png"):
    new_file = filex.split("/")[-1]
    # use the bounding boxes here
    image = cv.imread(filex)
    bounding_boxes = detector.detect_faces(image)
    try:
      crop_place = bounding_boxes[0]["box"]
      image = image[crop_place[1]:(crop_place[1] + crop_place[3]), crop_place[0]:(crop_place[0] + crop_place[2])]
      index = allx.index("_".join(new_file.split(".")[0].split("_")[:-1]))
      if len(image) != 0:
        cv.imwrite("/content/mosi_images/dev/" + str(emotions[index]) + "/" + new_file, image)
    except IndexError:
      no_box.append(new_file)

print('finished dev')

for i in range(len(test)):
  for filex in glob.glob("/content/images/" + test[i] + "_*.png"):
    new_file = filex.split("/")[-1]
    # use the bounding boxes here
    image = cv.imread(filex)
    bounding_boxes = detector.detect_faces(image)
    bounding_boxes = sorted(bounding_boxes, key=lambda x:x["confidence"])
    try:
      crop_place = bounding_boxes[0]["box"]
      image = image[crop_place[1]:(crop_place[1] + crop_place[3]), crop_place[0]:(crop_place[0] + crop_place[2])]
      index = allx.index("_".join(new_file.split(".")[0].split("_")[:-1]))
      if len(image) != 0:
        cv.imwrite("/content/mosi_images/test/" + str(emotions[index]) + "/" + new_file, image)
    except IndexError:
      no_box.append(new_file)

# then data transform it
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'dev': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join("mosi_images", x), data_transforms[x]) for x in ['train', 'dev', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4) for x in ['train', 'dev', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'dev', 'test']}
class_names = image_datasets['train'].classes


# to extract embeddings, we set certain layers to identity
finetuned_model = models.resnet50(pretrained=True, progress=True)
features = resnet50.fc.in_features
finetuned_model.fc = nn.Linear(features, 2)
finetuned_model.load_state_dict(torch.load("/content/best_model.pt", map_location=torch.device('cpu')))

# finetuned_model.fc = nn.Identity()
# finetuned_model.layer4 = nn.Identity()
# finetuned_model.layer3 = nn.Identity()

if torch.cuda.is_available():
  finetuned_model.to('cuda')

finetuned_model.eval()

train = np.array([])
train_labels = None
dev = np.array([])
dev_labels = None
test = np.array([])
test_labels = None

with torch.no_grad():
  for phase in ['train', 'dev', 'test']:
    for inputs, labels in dataloaders[phase]:
      # we have to do this because of a Colab issue where '.ipynb_checkpoints' becomes another folder
      # and that messes up labels, so we have to subtract 1 from the labels in order to match
      labels = labels.sub(1)
      if torch.cuda.is_available():
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')

      output = finetuned_model(inputs).cpu().numpy()
      labels = labels.cpu().numpy()
      if phase == 'train':
        if train.size == 0:
          train = output
          train_labels = labels
        else:
          train = np.concatenate((train, output))
          train_labels = np.concatenate((train_labels, labels))
      elif phase == 'dev':
        if dev.size == 0:
          dev = output
          dev_labels = labels
        else:
          dev = np.concatenate((dev, output))
          dev_labels = np.concatenate((dev_labels, labels))
      else:
        if test.size == 0:
          test = output
          test_labels = labels
        else:
          test = np.concatenate((test, output))
          test_labels = np.concatenate((test_labels, labels))

# this is the PCA/LDA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

clf = LinearDiscriminantAnalysis()

pca = PCA(n_components=200)
pca.fit(train)
clf.fit(pca.transform(train), train_labels)
print(clf.score(pca.transform(dev), dev_labels))
print(clf.score(pca.transform(test), test_labels))

# this is to finetune the model on MOSI after finetuning it on FER2013

finetuned_model = models.resnet50()
features = resnet50.fc.in_features
finetuned_model.fc = nn.Linear(features, 2)
finetuned_model.load_state_dict(torch.load("/content/best_model.pt", map_location=torch.device('cpu')))

if torch.cuda.is_available():
  finetuned_model.to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(finetuned_model.parameters(), lr=0.001, momentum=0.9)

new_model = train_model(finetuned_model, criterion, optimizer_ft, num_epochs=10)

torch.save(new_model.state_dict(), "/content/best_model_mosi_from_old_lr001.pt")

# finetuning on MOSI
resnet50 = models.resnet50(pretrained=True, progress=True)
features = resnet50.fc.in_features
resnet50.fc = nn.Linear(features, 2)

if torch.cuda.is_available():
  resnet50.to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(resnet50.parameters(), lr=0.001, momentum=0.9)

new_model = train_model(resnet50, criterion, optimizer_ft, num_epochs=10)

torch.save(new_model.state_dict(), "/content/best_model_mosi.pt")


# testing/doing inference
finetuned_model = models.resnet50()
features = resnet50.fc.in_features
finetuned_model.fc = nn.Linear(features, 2)
finetuned_model.load_state_dict(torch.load("/content/best_model.pt", map_location=torch.device('cpu')))

if torch.cuda.is_available():
  finetuned_model.to('cuda')

finetuned_model.eval()

test = np.array([])
test_labels = None

with torch.no_grad():
  for phase in ['test']:
    for inputs, labels in dataloaders[phase]:
      if torch.cuda.is_available():
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')

      outputs = finetuned_model(inputs)
      _, preds = torch.max(outputs, 1)

      preds = preds.cpu().numpy()
      labels = labels.cpu().numpy()

      if test.size == 0:
        test = preds
        test_labels = labels
      else:
        test = np.concatenate((test, preds))
        test_labels = np.concatenate((test_labels, labels))

correct = 0
for i in range(len(test)):
  if test[i] == test_labels[i]:
    correct += 1

print(correct/len(test))


# testing on real images
# put images into the structure that dataloader needs
import pandas as pd
import os
import shutil
import glob
from mtcnn import MTCNN
import cv2 as cv
import matplotlib.pyplot as plt

detector = MTCNN()

no_box = []

for filex in glob.glob("/content/test_real_images/test/0/"+ "*.jpg"):
  new_file = filex.split("/")[-1]
  # use the bounding boxes here
  image = cv.imread(filex)
  bounding_boxes = detector.detect_faces(image)
  bounding_boxes = sorted(bounding_boxes, key=lambda x:x["confidence"])
  try:
    crop_place = bounding_boxes[0]["box"]
    image = image[crop_place[1]:(crop_place[1] + crop_place[3]), crop_place[0]:(crop_place[0] + crop_place[2])]
    if len(image) != 0:
      cv.imwrite("/content/testing/test/0/" + new_file, image)
  except IndexError:
    no_box.append(new_file)

for filex in glob.glob("/content/test_real_images/test/1/"+ "*.jpg"):
  new_file = filex.split("/")[-1]
  # use the bounding boxes here
  image = cv.imread(filex)
  bounding_boxes = detector.detect_faces(image)
  bounding_boxes = sorted(bounding_boxes, key=lambda x:x["confidence"])
  try:
    crop_place = bounding_boxes[0]["box"]
    image = image[crop_place[1]:(crop_place[1] + crop_place[3]), crop_place[0]:(crop_place[0] + crop_place[2])]
    if len(image) != 0:
      cv.imwrite("/content/testing/test/1/" + new_file, image)
  except IndexError:
    no_box.append(new_file)


# inference on our faces
import os

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join("testing", x), data_transforms[x]) for x in ['test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
class_names = image_datasets['test'].classes

finetuned_model = models.resnet50()
features = resnet50.fc.in_features
finetuned_model.fc = nn.Linear(features, 2)
finetuned_model.load_state_dict(torch.load("/content/best_model_freeze3.pt", map_location=torch.device('cpu')))

if torch.cuda.is_available():
  finetuned_model.to('cuda')

finetuned_model.eval()

test = np.array([])
test_labels = None

with torch.no_grad():
  for inputs, labels in dataloaders['test']:
    labels = labels.sub(1)
    if torch.cuda.is_available():
      inputs = inputs.to('cuda')
      labels = labels.to('cuda')

    outputs = finetuned_model(inputs)
    _, preds = torch.max(outputs, 1)

    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    if test.size == 0:
      test = preds
      test_labels = labels
    else:
      test = np.concatenate((test, preds))
      test_labels = np.concatenate((test_labels, labels))

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

print("Precision", precision_score(test_labels, test, pos_label=0))
print("Recall", recall_score(test_labels, test, pos_label=0))
print("Accuracy", accuracy_score(test_labels, test))