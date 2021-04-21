# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (AveragePooling2D, BatchNormalization, Conv2D, Dense, MaxPool2D)
from tensorflow.keras.losses import Reduction, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD

from nni.nas.tensorflow.mutables import LayerChoice, InputChoice
from nni.algorithms.nas.tensorflow.enas import EnasTrainer

import numpy as np
import os
import cv2
import json
from os.path import isfile, join
from sklearn.model_selection import train_test_split


def load_images(file_path, size=120, is_train=True):
    with open('/nfshome/ialiev/Ilya-files/nni-patches/dataset_files/tenlabels.json', 'r') as fp:
        labels_dict = json.load(fp)
    with open('/nfshome/ialiev/Ilya-files/nni-patches/dataset_files/tenencoded_labels.json', 'r') as fp:
        encoded_labels = json.load(fp)
    Xarr = []
    Yarr = []
    number_of_classes = 10
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


def load_patches(file_path='/nfshome/ialiev/Ilya-files/nni-patches/10cls_Generated_dataset'):
    Xtrain, Ytrain = load_images(file_path, size=120, is_train=True)
    new_Ytrain = []
    for y in Ytrain:
        ys = np.argmax(y)
        new_Ytrain.append(ys)
    new_Ytrain = np.array(new_Ytrain)
    Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, new_Ytrain, random_state=1, train_size=0.8)

    return (Xtrain, Ytrain), (Xval, Yval)



class Net(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = LayerChoice([
            Conv2D(6, 3, padding='same', activation='relu'),
            Conv2D(6, 5, padding='same', activation='relu'),
        ])
        self.pool = MaxPool2D(2)
        self.conv2 = LayerChoice([
            Conv2D(16, 3, padding='same', activation='relu'),
            Conv2D(16, 5, padding='same', activation='relu'),
        ])
        self.conv3 = Conv2D(16, 1)

        self.skipconnect = InputChoice(n_candidates=1)
        self.bn = BatchNormalization()

        self.gap = AveragePooling2D(2)
        self.fc1 = Dense(120, activation='relu')
        self.fc2 = Dense(84, activation='relu')
        self.fc3 = Dense(10)

    def call(self, x):
        bs = x.shape[0]

        t = self.conv1(x)
        x = self.pool(t)
        x0 = self.conv2(x)
        x1 = self.conv3(x0)

        x0 = self.skipconnect([x0])
        if x0 is not None:
            x1 += x0
        x = self.pool(self.bn(x1))

        x = self.gap(x)
        x = tf.reshape(x, [bs, -1])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def accuracy(truth, logits):
    truth = tf.reshape(truth, (-1, ))
    predicted = tf.cast(tf.math.argmax(logits, axis=1), truth.dtype)
    equal = tf.cast(predicted == truth, tf.int32)
    return tf.math.reduce_sum(equal).numpy() / equal.shape[0]

def accuracy_metrics(truth, logits):
    acc = accuracy(truth, logits)
    return {'accuracy': acc}


if __name__ == '__main__':
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_valid, y_valid) = load_patches()
    x_train, x_valid = x_train / 255.0, x_valid / 255.0
    train_set = (x_train, y_train)
    valid_set = (x_valid, y_valid)

    net = Net()

    trainer = EnasTrainer(
        net,
        loss=SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE),
        metrics=accuracy_metrics,
        reward_function=accuracy,
        optimizer=SGD(learning_rate=0.001, momentum=0.9),
        batch_size=64,
        num_epochs=100,
        dataset_train=train_set,
        dataset_valid=valid_set
    )

    trainer.train()
