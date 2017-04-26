from keras.layers import Conv2D, Dense, Flatten, ELU, Input, concatenate, Dropout, MaxPool2D, add
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical
import numpy as np
import sys
import random
import os

def load_data():
    datapath = '/home/hongxwing/Workspace/cali/data/transfer/data_grid6_200/'
    valid_label = np.load(os.path.join(datapath, 'valid_label.npy'))
    opms = np.load(os.path.join(datapath, 'opms.npy'))
    print(valid_label.shape)
    print(opms.shape)

if __name__ == "__main__":
    load_data()
