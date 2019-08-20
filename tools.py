import numpy as np
import sys
import os
from array import array

from struct import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

## change location folder for your setting.( I used idx_type datas)
folder = "c://python/slime_mold/data/"

#read train data
def load_train_data():
    train_img = open(os.path.join(folder,'train-images.idx3-ubyte'),'rb' )
    train_lab = open(os.path.join(folder,'train-labels.idx1-ubyte'),'rb' )
    return train_img, train_lab

#read test dats
def load_test_data():
    test_img = open(os.path.join(folder,'t10k-images.idx3-ubyte'),'rb' )
    test_lab = open(os.path.join(folder,'t10k-labels.idx1-ubyte'),'rb' )
    return test_img, test_lab

#re formulating data shape!! 
def load_mnist():
    # return in order, (train_img_set, train_lab, test_img_set, test_lab )
    # load_datas
    train_img_file, train_lab_file = load_train_data()
    test_img_file, test_lab_file = load_test_data()

    img = np.zeros((28, 28))
    train_img_set = []
    train_cat_set = []
    test_img_set = []
    test_cat_set = []

    # load train set
    head_rem = train_img_file.read(16)
    head_rem_l = train_lab_file.read(8)

    while True:
        img = train_img_file.read(784)
        lab = train_lab_file.read(1)

        if not img:
            break;
        if not lab:
            break;

        train_cat_set.append(lab[0])

        # unpack
        img = np.reshape(unpack(len(img) * 'B', img), (784))
        train_img_set.append(img)

    # load_test_set
    head_rem = test_img_file.read(16)
    head_rem_l = test_lab_file.read(8)

    while True:
        img = test_img_file.read(784)
        lab = test_lab_file.read(1)


        if not img:
            break;
        if not lab:
            break;

        test_cat_set.append(lab[0])

        # unpack
        img = np.reshape(unpack(len(img) * 'B', img), (784))
        test_img_set.append(img)

    train_img = np.array(train_img_set)
    test_img = np.array(test_img_set)
    train_lab = np.array(train_cat_set)
    test_lab = np.array(test_cat_set)

    return train_img, train_lab, test_img, test_lab