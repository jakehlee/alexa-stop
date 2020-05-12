# y_test has sentiment from strongly negative -3 to strongly positive +#

import mmsdk
from mmsdk import mmdatasdk
import sys
import csv

# https://github.com/A2Zadeh/CMU-MultimodalSDK/issues/51
mydict={'myfeatures': 'CMU_MOSI_Opinion_Labels.csd'}
mydataset=mmdatasdk.mmdataset(mydict)

texts = open("../Downloads/files.txt")
label_file = open("labels_file.csv", "w")
writer = csv.writer(label_file, delimiter=',')
writer.writerow(["segment", "start", "end", "label"])

# https://github.com/A2Zadeh/CMU-MultimodalSDK/issues/54
for row in texts:
    labels = mydataset.computational_sequences['myfeatures'].data[row.split(".")[0]]['features'][:]
    intervals = mydataset.computational_sequences['myfeatures'].data[row.split(".")[0]]['intervals'][:]
    for i in range(len(labels)):
        if float(labels[i][0]) < 0:
            writer.writerow([row.split(".")[0] + "_" + str(i+1), intervals[i][0], intervals[i][1], 0])
        else:
            writer.writerow([row.split(".")[0] + "_" + str(i+1), intervals[i][0], intervals[i][1], 1])