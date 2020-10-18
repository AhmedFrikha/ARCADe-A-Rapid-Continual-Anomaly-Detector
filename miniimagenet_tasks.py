"""
This file was modified from: Giacomo Spigler. Meta-learnt priors slow down catastrophic forgetting in neural networks. arXiv preprint arXiv:1909.04170, 2019.

Utility functions to create Tasks from the Mini-ImageNet dataset.

The created tasks will be derived from CB_OCCTask, and can be aggregated in a TaskDistribution object.
"""

import numpy as np
import pickle

from task import CB_OCCTask, OCCTask
from task_distribution import TaskDistribution


# All data is pre-loaded in memory. This takes ~5GB if I recall correctly.
miniimagenet_trainX = []
miniimagenet_trainY = []

miniimagenet_valX = []
miniimagenet_valY = []

miniimagenet_testX = []
miniimagenet_testY = []


def create_miniimagenet_data_split(
    path_to_pkl,

):
    """
    Returns a TaskDistribution that, on each reset, samples a different set of Mini-ImageNet classes.

    Arguments:
    path_to_pkl: string
        Path to the pkl wrapped Mini-ImageNet dataset.

    Returns:
    miniimagenet_trainX : array
        features of the meta-training examples
    miniimagenet_trainY : array
        labels of the meta-training examples
    miniimagenet_valX : array
        features of the meta-validation examples
    miniimagenet_valY : array
        labels of the meta-validation examples
    miniimagenet_testX : array
        features of the meta-testing examples
    miniimagenet_testY : array
        labels of the meta-testing examples
    """

    global miniimagenet_trainX
    global miniimagenet_trainY

    global miniimagenet_valX
    global miniimagenet_valY

    global miniimagenet_testX
    global miniimagenet_testY

    with open(path_to_pkl, "rb") as f:
        d = pickle.load(f)
        miniimagenet_trainX, miniimagenet_trainY = d["train"]
        miniimagenet_valX, miniimagenet_valY = d["val"]
        miniimagenet_testX, miniimagenet_testY = d["test"]

    miniimagenet_trainX = miniimagenet_trainX.astype(np.float32) / 255.0
    miniimagenet_valX = miniimagenet_valX.astype(np.float32) / 255.0
    miniimagenet_testX = miniimagenet_testX.astype(np.float32) / 255.0

    del d

    return miniimagenet_trainX, miniimagenet_trainY, miniimagenet_valX, miniimagenet_valY, miniimagenet_testX, miniimagenet_testY
