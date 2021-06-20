import argparse
import json
import os
import statistics
from os.path import isfile, join

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import (AveragePooling2D, BatchNormalization, Conv2D, Dense, MaxPool2D)

from nni.algorithms.nas.tensorflow.classic_nas import get_and_apply_next_architecture
from nni.nas.tensorflow.mutables import LayerChoice, InputChoice

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
        activations = [tf.nn.relu, tf.nn.softmax, tf.nn.leaky_relu, tf.nn.gelu, tf.nn.elu]
        ind_act = np.random.randint(0, len(activations) - 1)
        self.fc1 = Dense(np.random.randint(20, 200), activation=activations[ind_act])
        self.fc2 = Dense(np.random.randint(20, 200), activation=activations[ind_act])
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


def load_images(size=120, is_train=True):
    file_path = 'C:/Users/aliev/Documents/GitHub/nas-fedot/10cls_Generated_dataset'
    with open('C:/Users/aliev/Documents/GitHub/nas-fedot/dataset_files/labels_10.json', 'r') as fp:
        labels_dict = json.load(fp)
    with open('C:/Users/aliev/Documents/GitHub/nas-fedot/dataset_files/encoded_labels_10.json', 'r') as fp:
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


def load_patches():
    Xtrain, Ytrain = load_images(size=120, is_train=True)
    new_Ytrain = []
    for y in Ytrain:
        ys = np.argmax(y)
        new_Ytrain.append(ys)
    new_Ytrain = np.array(new_Ytrain)
    Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, new_Ytrain, random_state=1, train_size=0.8)

    return Xtrain, Ytrain, Xval, Yval


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
    test_loss = tf.keras.metrics.Mean()
    # test_AUC = tf.keras.metrics.AUC(num_thresholds=3)
    roc_auc_values = []
    for (x, y) in test_dataset:
        # training=False is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        logits = model(x, training=False)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)
        test_loss(prediction, y)

        for predict, true in zip(logits, y):
            y_true = [0 for _ in range(10)]
            y_true[true] = 1
            roc_auc_score = roc_auc(y_true=y_true,
                                    y_score=predict)
            roc_auc_values.append(roc_auc_score)

    for el in model.variables:
        print(el.name, el.shape)

    roc_auc_value = statistics.mean(roc_auc_values)

    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
    print("Test set loss: {:.3%}".format(test_loss.result()))
    print("ROC AUC: {:.3%}".format(roc_auc_value))
    return test_accuracy.result(), test_loss.result(), roc_auc_value


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 10)')
    args, _ = parser.parse_known_args()

    # cifar10 = tf.keras.datasets.cifar10
    # x_train, y_train, x_test, y_test = cifar10.load_data()
    x_train, y_train, x_test, y_test = load_patches()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    split = int(len(x_train) * 0.9)
    dataset_train = tf.data.Dataset.from_tensor_slices((x_train[:split], y_train[:split])).batch(64)
    dataset_valid = tf.data.Dataset.from_tensor_slices((x_train[split:], y_train[split:])).batch(64)
    dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)

    net = Net()

    get_and_apply_next_architecture(net)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    train(net, dataset_train, optimizer, args.epochs)
    summary_net = net.summary()
    # for el in net.trainable_variables:
    #     print(el.name, el.shape)

    acc, loss, auc = test(net, dataset_test)

    # nni.report_final_result(acc.numpy())
    # nni.report_final_result(loss.numpy())
    # nni.report_final_result(auc)
