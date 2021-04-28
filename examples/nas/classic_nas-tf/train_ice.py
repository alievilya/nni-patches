import argparse
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (AveragePooling2D, BatchNormalization, Conv2D, Dense, MaxPool2D)
from tensorflow.keras.losses import Reduction, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD

import nni
from nni.nas.tensorflow.mutables import LayerChoice, InputChoice
from nni.algorithms.nas.tensorflow.classic_nas import get_and_apply_next_architecture

import numpy as np
import os
import cv2
import json
from os.path import isfile, join
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.model_selection import train_test_split
import cv2

tf.get_logger().setLevel('ERROR')

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

        self.skipconnect = InputChoice(n_candidates=2, n_chosen=1)
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

        x0 = self.skipconnect([x0, None])
        if x0 is not None:
            x1 += x0
        x = self.pool(self.bn(x1))

        x = self.gap(x)
        x = tf.reshape(x, [bs, -1])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def loss(model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)

    return loss_object(y_true=y, y_pred=y_)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def train(net, train_dataset, optimizer, num_epochs):
    train_loss_results = []
    train_accuracy_results = []

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        for x, y in train_dataset:
            loss_value, grads = grad(net, x, y)
            optimizer.apply_gradients(zip(grads, net.trainable_variables))
            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(y, net(x, training=True))

        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 1 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

def test(model, test_dataset):
    test_accuracy = tf.keras.metrics.Accuracy()

    for (x, y) in test_dataset:
        # training=False is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        logits = model(x, training=False)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)

    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
    return test_accuracy.result()



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
    # Xtr_more = get_more_images(Xtrain)
    # Ytr_more = np.concatenate((Ytrain, Ytrain, Ytrain))

    # return Xtr_more, Ytr_more, Xtest, Ytest
    return Xtrain, Ytrain, Xtest, Ytest

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



if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 10)')
    args, _ = parser.parse_known_args()

    file_path = 'C:/Users/aliev/Documents/GitHub/nas-fedot/IcebergsDataset/train.json'
    x_train, y_train, x_test, y_test = from_json(file_path=file_path)
    split = int(len(x_train) * 0.9)
    dataset_train = tf.data.Dataset.from_tensor_slices((x_train[:split], y_train[:split])).batch(64)
    dataset_valid = tf.data.Dataset.from_tensor_slices((x_train[split:], y_train[split:])).batch(64)
    dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)

    net = Net()
    
    get_and_apply_next_architecture(net)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    train(net, dataset_train, optimizer, args.epochs)

    acc = test(net, dataset_test)

    nni.report_final_result(acc.numpy())
