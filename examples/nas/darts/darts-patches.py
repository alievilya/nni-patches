from nni.algorithms.nas.pytorch.darts.trainer import DartsTrainer, DartsMutator
import json
import logging
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn

import datasets
from model import CNN
from nni.nas.pytorch.callbacks import ArchitectureCheckpoint, LRSchedulerCallback
from utils import accuracy


import numpy as np
import os
import cv2
import json
from os.path import isfile, join
from sklearn.model_selection import train_test_split

def load_images(file_path, size=120, is_train=True):
    with open('X:/code/Maga_Nir/frameworks_for_paper/nni-patches/nni/dataset_files/labels.json', 'r') as fp:
        labels_dict = json.load(fp)
    with open('X:/code/Maga_Nir/frameworks_for_paper/nni-patches/nni/dataset_files/encoded_labels.json', 'r') as fp:
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


def load_patches(file_path='X:/code/Maga_Nir/frameworks_for_paper/nni-patches/nni/Generated_dataset'):
    Xtrain, Ytrain = load_images(file_path, size=120, is_train=True)
    new_Ytrain = []
    for y in Ytrain:
        ys = np.argmax(y)
        new_Ytrain.append(ys)
    new_Ytrain = np.array(new_Ytrain)
    Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, new_Ytrain, random_state=1, train_size=0.8)

    return (Xtrain, Ytrain), (Xval, Yval)

class MyTrainer(DartsTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.status_writer = open("log", "w")

    def _logits_and_loss(self, X, y):
        self.mutator.reset()
        logits = self.model(X)
        loss = self.loss(logits, y)
        print(json.dumps(self.mutator.status()), file=self.status_writer)
        self.status_writer.flush()
        return logits, loss


if __name__ == "__main__":

    model = CNN(120, 3, 16, 3, 3)
    model.cuda()
    mutator = DartsMutator(model)
    vis_graph = mutator.graph(torch.randn((1, 3, 120, 120)).cuda())
    with open("graph.json", "w") as f:
        json.dump(vis_graph, f)

    # dataset_train, dataset_valid = datasets.get_dataset("cifar10")
    # dataset_train, dataset_valid = load_patches()
    # Xarr = Xarr.reshape(-1, size, size, 1)

    (x_train, y_train), (x_valid, y_valid) = load_patches()
    x_train, x_valid = x_train / 255.0, x_valid / 255.0
    dataset_train = (x_train, y_train)
    dataset_valid = (x_valid, y_valid)


    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
    trainer = MyTrainer(model=model,
                        mutator=mutator,
                        loss=criterion,
                        metrics=lambda output, target: accuracy(output, target, topk=(1,)),
                        optimizer=optim,
                        num_epochs=1,
                        dataset_train=dataset_train,
                        dataset_valid=dataset_valid,
                        batch_size=2,
                        log_frequency=10,
                        arc_learning_rate=0.1,
                        unrolled=False)
    trainer.train()