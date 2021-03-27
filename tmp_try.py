import cv2
import json
import os
from os.path import isfile, join

import keras
import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

H, W = 28, 28
NUM_CLASSES = 10


def load_mnist_data():
    '''
    Load MNIST dataset
    '''

    num_train = 60000
    num_test = 10000
    # mnist_path = os.path.join(os.environ.get('NNI_OUTPUT_DIR'), 'mnist.npz')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # os.remove(mnist_path)

    x_train = (np.expand_dims(x_train, -1).astype(np.float) / 255.)[:num_train]
    x_test = (np.expand_dims(x_test, -1).astype(np.float) / 255.)[:num_test]
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)[:num_train]
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)[:num_test]


    return x_train, y_train, x_test, y_test

def load_images(file_path, size=120, is_train=True):
    with open('dataset_files/labels.json', 'r') as fp:
        labels_dict = json.load(fp)
    with open('dataset_files/encoded_labels.json', 'r') as fp:
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


def load_patches(file_path = 'Generated_dataset'):
    Xtrain, Ytrain = load_images(file_path, size=120, is_train=True)
    Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, random_state=1, train_size=0.8)
    return Xtrain, Xval, Ytrain, Yval


x_train, y_train, x_test, y_test = load_patches()
print(type(x_train))
