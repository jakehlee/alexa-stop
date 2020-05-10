"""
Generates classification statistics for a test set on the provided model weights
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

from sklearn.metrics import classification_report

WEIGHTS = "weights/21epochs-0.8312val-0509-1841.pt"

def usage():
    print("Usage: python eval-perf.py OURdata/")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        usage()
    else:
        test_dir = sys.argv[1]

    # define data transforms

    test_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_dir, test_transforms)

    print("test set", len(test_dataset))

    test_loader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=64,
                        shuffle=True,
                        num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(torch.cuda.current_device())

    criterion = nn.CrossEntropyLoss()

    # import model
    model = models.resnet50(pretrained=False)

    num_in = model.fc.in_features
    model.fc = nn.Linear(num_in, 2)

    model.load_state_dict(torch.load(WEIGHTS))
    model = model.to(device)
    model.eval()

    test_loss = 0.0
    test_corrects = 0

    all_labels = []
    all_preds = []

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)


        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        all_labels += list(labels.to('cpu'))
        all_preds += list(preds.to('cpu'))

        test_loss += loss.item() * inputs.size(0)
        test_corrects += torch.sum(preds == labels.data)
        
    epoch_test_loss = test_loss / len(test_dataset)
    epoch_test_acc = test_corrects.double() / len(test_dataset)

    print("Test Loss: {:.4f} Acc: {:.4f}".format(epoch_test_loss, 
        epoch_test_acc))

    print(classification_report(all_labels, all_preds, target_names=['bad', 'good']))