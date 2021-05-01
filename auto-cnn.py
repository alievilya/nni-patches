import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Removes Tensorflow debuggin ouputs

import tensorflow as tf

tf.get_logger().setLevel('INFO') # Removes Tensorflow debugging ouputs

from auto_cnn.gan import AutoCNN

import random
import numpy as np
import os
import cv2
import json
from os.path import isfile, join
from sklearn.model_selection import train_test_split


# Sets the random seeds to make testing more consisent
random.seed(42)
tf.random.set_seed(42)


def load_images(size=120, is_train=True):
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

def load_patches():

    Xtrain, Ytrain = load_images(size=120, is_train=True)
    new_Ytrain = []
    for y in Ytrain:
        ys = np.argmax(y)
        new_Ytrain.append(ys)
    new_Ytrain = np.array(new_Ytrain)
    Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, new_Ytrain, random_state=1, train_size=0.8)

    return (Xtrain, Ytrain), (Xval, Yval)

def mnist_test():
    # Loads the data as test and train
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    (Xtrain, y_train), (Xval, y_test) = load_patches()
    x_train, x_test = Xtrain / 255.0, Xval / 255.0
    # dataset_train, dataset_valid = (Xtrain, Ytrain), (Xval, Yval)


    # Puts the data in a dictionary for the algorithm to use
    data = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}

    # Sets the wanted parameters
    a = AutoCNN(population_size=5, maximal_generation_number=4, dataset=data, epoch_number=5)

    # Runs the algorithm until the maximal_generation_number has been reached
    best_cnn = a.run()
    print(best_cnn)

if __name__ == '__main__':
    mnist_test()