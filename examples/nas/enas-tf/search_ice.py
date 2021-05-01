# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#!/usr/bin/python
# -*- coding: utf-8 -*-
import os, sys


import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (AveragePooling2D, BatchNormalization, Conv2D, Dense, MaxPool2D)
from tensorflow.keras.losses import Reduction, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import roc_auc_score as roc_auc
import statistics

from nni.nas.tensorflow.mutables import LayerChoice, InputChoice
from nni.algorithms.nas.tensorflow.enas import EnasTrainer

import numpy as np
import os
import cv2
import json
from os.path import isfile, join
from sklearn.model_selection import train_test_split
import pandas as pd



def from_json(file_path):
    df_train = pd.read_json(file_path)
    Xtrain = get_scaled_imgs(df_train)
    Ytrain = np.array(df_train['is_iceberg'])
    df_train.inc_angle = df_train.inc_angle.replace('na', 0)
    idx_tr = np.where(df_train.inc_angle > 0)
    Ytrain = Ytrain[idx_tr[0]]
    Xtrain = Xtrain[idx_tr[0], ...]
    Ytrain_new = []
    for y in Ytrain:
        new_Y = []
        new_Y.append(y)
        Ytrain_new.append(new_Y)
    Ytrain = np.array(Ytrain_new)


    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xtrain, Ytrain, random_state=1, train_size=0.75)
    Xtr_more = get_more_images(Xtrain)
    Ytr_more = np.concatenate((Ytrain, Ytrain, Ytrain))

    return Xtr_more, Ytr_more, Xtest, Ytest
    # return Xtrain, Ytrain, Xtest, Ytest

def get_scaled_imgs(df):
    imgs = []

    for i, row in df.iterrows():
        # make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2  # plus since log(x*y) = log(x) + log(y)

        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())
        im = np.dstack((a, b, c))
        im = cv2.resize(im, (72, 72), interpolation = cv2.INTER_AREA)
        imgs.append(im)

    return np.array(imgs)


def get_more_images(imgs):
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []

    for i in range(0, imgs.shape[0]):
        a = imgs[i, :, :, 0]
        b = imgs[i, :, :, 1]
        c = imgs[i, :, :, 2]

        av = cv2.flip(a, 1)
        ah = cv2.flip(a, 0)
        bv = cv2.flip(b, 1)
        bh = cv2.flip(b, 0)
        cv = cv2.flip(c, 1)
        ch = cv2.flip(c, 0)

        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))

    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)

    more_images = np.concatenate((imgs, v, h))

    return more_images



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
        self.fc3 = Dense(2)

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


def loss_f(truth, prediction):
    test_loss = tf.keras.metrics.Mean()
    test_loss(prediction, truth)
    return test_loss.result()


def auc_f(truth, prediction):
    roc_auc_values = []
    for predict, true in zip(prediction, truth):
        y_true = [0 for _ in range(2)]
        y_true[true[0]] = 1
        roc_auc_score = roc_auc(y_true=y_true,
                                y_score=predict)
        roc_auc_values.append(roc_auc_score)
    roc_auc_value = statistics.mean(roc_auc_values)
    return roc_auc_value

def accuracy_metrics(truth, logits):
    acc = accuracy(truth, logits)

    loss = loss_f(truth, logits)
    auc = auc_f(truth, logits)
    return {'accuracy': acc, 'loss': loss, 'ROC AUC': auc}


if __name__ == '__main__':


    file_path = 'C:/Users/aliev/Documents/GitHub/nas-fedot/IcebergsDataset/train.json'
    Xtrain, Ytrain, Xval, Yval = from_json(file_path=file_path)
    train_set = (Xtrain, Ytrain)
    valid_set = (Xval, Yval)

    net = Net()

    trainer = EnasTrainer(
        net,
        loss=SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE),
        metrics=accuracy_metrics,
        reward_function=accuracy,
        optimizer=SGD(learning_rate=0.001, momentum=0.9),
        batch_size=64,
        num_epochs=1,
        dataset_train=train_set,
        dataset_valid=valid_set
    )

    trainer.train()
