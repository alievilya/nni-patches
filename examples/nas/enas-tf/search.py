# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from tensorflow.keras.losses import Reduction, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD

from nni.algorithms.nas.tensorflow import enas

import datasets
from macro import GeneralNetwork
from micro import MicroNetwork
from utils import accuracy, accuracy_metrics

import numpy as np
import os
import cv2
import json
from os.path import isfile, join
from sklearn.model_selection import train_test_split


# TODO: argparse

def load_images(file_path, size=120, is_train=True):
    file_path='C:/Users/aliev/Documents/GitHub/nas-fedot/Generated_dataset'
    with open('C:/Users/aliev/Documents/GitHub/nas-fedot/dataset_files/labels.json', 'r') as fp:
        labels_dict = json.load(fp)
    with open('C:/Users/aliev/Documents/GitHub/nas-fedot/dataset_files/encoded_labels.json', 'r') as fp:
        encoded_labels = json.load(fp)

    Xarr = []
    Yarr = []
    number_of_classes = 3
    files = [f for f in os.listdir(file_path) if isfile(join(file_path, f))]
    files.sort()
    for filename in files:
        image = cv2.imread(join(file_path, filename))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (size, size))
        Xarr.append(image)
        label_names = labels_dict[filename[:-4]]
        each_file_labels = [0 for _ in range(number_of_classes)]
        for name in label_names:
            num_label = encoded_labels[name]
            # each_file_labels.append(num_label)
            each_file_labels[num_label] = 1
        Yarr.append(each_file_labels)
    Xarr = np.array(Xarr)
    Yarr = np.array(Yarr)
    # Xarr = Xarr.reshape(-1, size, size, 1)

    return Xarr, Yarr

def load_patches(file_path=''):

    Xtrain, Ytrain = load_images(file_path, size=120, is_train=True)
    new_Ytrain = []
    for y in Ytrain:
        y_a = []
        ys = np.argmax(y)
        y_a.append(ys)
        new_Ytrain.append(y_a)
    new_Ytrain = np.array(new_Ytrain)
    Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, new_Ytrain, random_state=1, train_size=0.8)

    return Xtrain, Ytrain, Xval, Yval


# dataset_train, dataset_valid = datasets.get_dataset()
Xtrain, Ytrain, Xval, Yval = load_patches()
Xtrain, Xval = Xtrain / 255.0, Xval / 255.0
dataset_train, dataset_valid = (Xtrain, Ytrain), (Xval, Yval)

# model = GeneralNetwork()
model = MicroNetwork()

loss = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)
optimizer = SGD(learning_rate=0.05, momentum=0.9)

trainer = enas.EnasTrainer(model,
                           loss=loss,
                           metrics=accuracy_metrics,
                           reward_function=accuracy,
                           optimizer=optimizer,
                           batch_size=64,
                           num_epochs=30,
                           dataset_train=dataset_train,
                           dataset_valid=dataset_valid)
trainer.train()
