"""
Train binary classification model on FERPlus
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

if __name__ == "__main__":

    # define data transforms

    train_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # define data import

    train_dir = 'FERdata/FER2013Train'
    valid_dir = 'FERdata/FER2013Valid'

    train_dataset = datasets.ImageFolder(train_dir, train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, valid_transforms)

    print("training set", len(train_dataset))
    print("training set", len(valid_dataset))

    train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=64,
                        shuffle=True,
                        num_workers=4)

    valid_loader = torch.utils.data.DataLoader(
                        valid_dataset,
                        batch_size=64,
                        shuffle=True,
                        num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(torch.cuda.current_device())


    # import model
    model_ft = models.resnet50(pretrained=True)

    # freeze layers
    """
    for param in model_ft.parameters():
        param.requires_grad = False
    """
    num_in = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_in, 2)


    model_ft = model_ft.to(device)

    # define optimizer
    optimizer = optim.SGD(model_ft.parameters(),
                            lr=0.0001,
                            momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # train model
    train_start = time.time()
    epochs = 21

    for epoch in range(epochs):
        epoch_start = time.time()

        print('Epoch {}/{}'.format(epoch, epochs-1))

        # TRAIN
        model_ft.train()

        train_loss = 0.0
        train_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model_ft(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)

        scheduler.step()

        epoch_train_loss = train_loss / len(train_dataset)
        epoch_train_acc = train_corrects.double() / len(train_dataset)

        print("Train Loss: {:.4f} Acc: {:.4f}".format(epoch_train_loss, 
            epoch_train_acc))

        # VAL

        model_ft.eval()

        val_loss = 0.0
        val_corrects = 0

        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(False):
                outputs = model_ft(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
        
        epoch_val_loss = val_loss / len(valid_dataset)
        epoch_val_acc = val_corrects.double() / len(valid_dataset)

        print("Val Loss: {:.4f} Acc: {:.4f}".format(epoch_val_loss, 
            epoch_val_acc))

        print("Epoch Time: {:.0f}".format(time.time() - epoch_start))

        datetime_obj = datetime.now()
        timestamp = datetime_obj.strftime("%m%d-%H%M")

        outputname = "weights/{}epochs-{:.4f}val-{}.pt".format(epochs, epoch_val_acc, timestamp)

        torch.save(model_ft.state_dict(), outputname)