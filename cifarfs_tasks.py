"""
This file was modified from: Giacomo Spigler. Meta-learnt priors slow down catastrophic forgetting in neural networks. arXiv preprint arXiv:1909.04170, 2019.

Utility functions to create Tasks from the Mini-ImageNet dataset.

The created tasks will be derived from CB_OCCTask, and can be aggregated in a TaskDistribution object.
"""

import numpy as np
import pickle

from task import CB_OCCTask, OCCTask, ClassificationTask
from task_distribution import TaskDistribution


# All data is pre-loaded in memory. This takes ~5GB if I recall correctly.
cifarfs_trainX = []
cifarfs_trainY = []

cifarfs_valX = []
cifarfs_valY = []

cifarfs_testX = []
cifarfs_testY = []


def create_cifarfs_data_split(
    path_to_pkl_tr,
    path_to_pkl_val,
    path_to_pkl_test,

):
    """
    Returns a split of meta-training, meta-validation and meta-testing classes.

    Arguments:
    path_to_pkl_tr: string
        Path to the pkl wrapped CIFAR-FS meta-training dataset.
    path_to_pkl_val: string
        Path to the pkl wrapped CIFAR-FS meta-validation dataset.
    path_to_pkl_test: string
        Path to the pkl wrapped CIFAR-FS meta-testing dataset.

    Returns:
    cifarfs_trainX : array
        features of the meta-training examples
    cifarfs_trainY : array
        labels of the meta-training examples
    cifarfs_valX : array
        features of the meta-validation examples
    cifarfs_valY : array
        labels of the meta-validation examples
    cifarfs_testX : array
        features of the meta-testing examples
    cifarfs_testY : array
        labels of the meta-testing examples
    """

    global cifarfs_trainX
    global cifarfs_trainY

    global cifarfs_valX
    global cifarfs_valY

    global cifarfs_testX
    global cifarfs_testY

    with open(path_to_pkl_tr, "rb") as f:
        d = pickle.load(f, encoding='bytes')
        key_label, key_data = d.keys()
        cifarfs_trainX, cifarfs_trainY = d[key_data], d[key_label]

    with open(path_to_pkl_val, "rb") as f:
        d = pickle.load(f, encoding='bytes')
        key_label, key_data = d.keys()

        cifarfs_valX, cifarfs_valY = d[key_data], d[key_label]

    with open(path_to_pkl_test, "rb") as f:
        d = pickle.load(f, encoding='bytes')
        key_label, key_data = d.keys()

        cifarfs_testX, cifarfs_testY = d[key_data], d[key_label]

    cifarfs_trainX = cifarfs_trainX.astype(np.float32) / 255.0
    cifarfs_valX = cifarfs_valX.astype(np.float32) / 255.0
    cifarfs_testX = cifarfs_testX.astype(np.float32) / 255.0

    cifarfs_trainY, cifarfs_valY, cifarfs_testY = np.array(
        cifarfs_trainY), np.array(cifarfs_valY), np.array(cifarfs_testY)

    del d

    return cifarfs_trainX, cifarfs_trainY, cifarfs_valX, cifarfs_valY, cifarfs_testX, cifarfs_testY
