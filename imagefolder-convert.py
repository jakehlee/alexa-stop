"""
Converts dataset structure generated by FERPlus to one expected by ImageFolder
"""

import sys, os
import csv
import numpy as np

# train
good = ['neutral', 'happiness']
bad = ['surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
labels = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 
            'fear', 'contempt', 'unknown', 'NF']
dirs = ['good', 'bad']


roots = ['data/FER2013Train/', 'data/FER2013Test/', 'data/FER2013Valid/']

for root in roots:
    for d in dirs:
        if not os.path.isdir(os.path.join(root, d)):
            os.mkdir(os.path.join(root, d))


    with open(os.path.join(root, 'label.csv')) as f:
        r = csv.reader(f, delimiter=',')

        for row in r:
            votes = np.array(row[2:10]).astype(int)
            majvote = labels[votes.argmax()]

            if majvote in ['unknown', 'NF']:
                continue
            elif majvote in good:
                filename = row[0]
                os.rename(os.path.join(root, filename), 
                            os.path.join(root, 'good', filename))
                print('moved to', os.path.join(root, 'good', filename))
            elif majvote in bad:
                filename = row[0]
                os.rename(os.path.join(root, filename), 
                            os.path.join(root, 'bad', filename))
                print('moved to', os.path.join(root, 'bad', filename))