"""
Main system code for Alexa Stop Project
"""
# System packages
import sys, os
import PIL
import time
import serial
import serial.tools.list_ports
# Science packages
import numpy as np
import matplotlib.pyplot as plt
# ML/DL packages
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import cv2

# filepath for trained model weights
WEIGHTS = "weights/21epochs-0.8312val-0509-1841.pt"
# voltage threshold for when Alexa is responding

THRESHOLD = 930


def detect_emotion(frame, tf, model, fc):
    """ Detect face emotion from webcam capture.

    Arguments:
    frame:  image frame taken with webcam
    tf:     input transformation definition via transforms.Compose
    model:  pytorch model to use for emotion detection
    fc:     face cascade model to use for face detection

    Returns:
    True if bad emotion, False if good emotion.

    """
    # convert to black and white for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # perform face detection
    #faces = fc.detectMultiScale(gray, 1.3, 5)
    #x,y,w,h = (0,0,480,480)
    x,y,w,h = (241,161,185,185)
    #if len(faces) > 0:
    #    x,y,w,h = faces[0]
    #else:
        # no face found, can't make decision
    #    return False

    # crop for face detection and preprocess for network
    gray_crop = gray[y:y+h, x:x+w]
    gray3 = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)
    graypil = PIL.Image.fromarray(gray3)

    # resize for user display
    #gray3 = cv2.resize(gray3, (224,224))
    
    # preprocess for network
    image = tf(graypil)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.unsqueeze(0)

    # get model prediction
    pred = model(image)
    val, ind = torch.max(pred,1)

    if ind == 0:
        return True
    else:
        return False



if __name__ == "__main__":

    ### SERIAL COMMUNICATION
    ports = serial.tools.list_ports.comports(include_links=False)

    # query user to select arduino's port
    print("[INFO] Currently available ports:")

    i = 0
    for p in ports:
        print("[{}] {}: {}".format(i, p.device, p.description))
        i += 1

    pid = input("[INPUT] Select the Arduino Uno port:")

    # initialize serial 
    port = ports[int(pid)].device
    ser = serial.Serial(port, 9600)
    if not ser.isOpen():
        ser.open()


    ### VIDEO CAPTURE
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    print("[INFO] Video initialized at resolution:", frame.shape)


    ### NEURAL NETWORK

    # input preprocessing
    live_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # CUDA if available, CPU if not
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load ResNet50 Architecture
    model = models.resnet50(pretrained=False)

    # Modify last layer to have two output classes
    num_in = model.fc.in_features
    model.fc = nn.Linear(num_in, 2)

    # Load weights (different for CUDA and CPU)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(WEIGHTS))
    else:
        model.load_state_dict(torch.load(WEIGHTS, map_location=torch.device('cpu')))

    # Set model to eval mode
    model = model.to(device)
    model.eval()

    ### OTHER

    # face detection model
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


    ### MAIN LOGIC

    print("Running... Press 'q' at any point to quit gracefully.")

    resp_state = False
    resp_last = time.time()
    int_last = time.time()
    while True:
        try:
            ret, frame = cap.read()
            # we expect this at around 10HZ, varying
            raw = ser.readline()
            raw = raw.decode()
            try:
                voltage = float(raw.rstrip("\r\n"))
            except ValueError:
                continue

            print("[DEBUG] voltage={}".format(voltage))

            if voltage <= THRESHOLD:
                # Alexa is now responding to a user query
                # Change state and record timestamp.
                resp_state = True
                resp_last = time.time()
            else:
                # If it's been more than 2 seconds since the last sensor reading
                # indicating Alexa response, then change state back
                if time.time() - resp_last > 2:
                    resp_state = False

            if resp_state:
                # Alexa is now responding, and we need to determine if we should
                # interrupt it. Begin deep learning procedure here.

                interrupt = detect_emotion(frame, live_transforms, 
                                            model, face_cascade)

                if interrupt and time.time() - int_last > 2:
                    ser.write("int".encode())
                    int_last = time.time()

                if interrupt:
                    cv2.putText(frame, 'interrupt', (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                else:
                    cv2.putText(frame, 'normal', (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            else:
                # Alexa is not responding, nothing to do...
                pass
            x,y,w,h = (241,161,185,185)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            break


    print("Terminating...")
    ser.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


