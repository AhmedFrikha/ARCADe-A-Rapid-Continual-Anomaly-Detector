"""
This file was modified from: Giacomo Spigler. Meta-learnt priors slow down catastrophic forgetting in neural networks. arXiv preprint arXiv:1909.04170, 2019.

Utility functions to create Tasks from the Omniglot dataset.

The created tasks will be derived from CB_OCCTask, and can be aggregated in a TaskDistribution object.
"""

import numpy as np
import pickle

from task import CB_OCCTask, OCCTask
from task_distribution import TaskDistribution

charomniglot_trainX = []
charomniglot_trainY = []

charomniglot_valX = []
charomniglot_valY = []

charomniglot_testX = []
charomniglot_testY = []


def get_omniglot_allcharacters_data_split(
    path_to_pkl,

):
    """
    Returns a TaskDistribution that, on each reset, samples a different set of Omniglot characters.

    Arguments:
    path_to_pkl: string
        Path to the pkl wrapped Omniglot dataset.

    Returns:
    trX : array
        features of the meta-training examples
    trY : array
        labels of the meta-training examples
    valX : array
        features of the meta-validation examples
    valY : array
        labels of the meta-validation examples
    teX : array
        features of the meta-testing examples
    teY : array
        labels of the meta-testing examples
    """

    with open(path_to_pkl, "rb") as f:
        d = pickle.load(f)
        trainX_ = d["trainX"]
        trainY_ = d["trainY"]
        testX_ = d["testX"]
        testY_ = d["testY"]
    trainX_.extend(testX_)
    trainY_.extend(testY_)

    global charomniglot_trainX
    global charomniglot_trainY

    global charomniglot_valX
    global charomniglot_valY

    global charomniglot_testX
    global charomniglot_testY

    cutoff_tr, cutoff_val = 25, 30
    charomniglot_trainX = trainX_[:cutoff_tr]
    charomniglot_trainY = trainY_[:cutoff_tr]

    charomniglot_valX = trainX_[cutoff_tr:cutoff_val]
    charomniglot_valY = trainY_[cutoff_tr:cutoff_val]

    charomniglot_testX = trainX_[cutoff_val:]
    charomniglot_testY = trainY_[cutoff_val:]

    # Create a single large dataset with all characters, each for train and
    # test, and rename the targets appropriately
    trX = []
    trY = []

    valX = []
    valY = []

    teX = []
    teY = []

    cur_label_start = 0
    for alphabet_i in range(len(charomniglot_trainY)):
        charomniglot_trainY[alphabet_i] += cur_label_start
        trX.extend(charomniglot_trainX[alphabet_i])
        trY.extend(charomniglot_trainY[alphabet_i])
        cur_label_start += len(set(charomniglot_trainY[alphabet_i]))

    cur_label_start = 0
    for alphabet_i in range(len(charomniglot_valY)):
        charomniglot_valY[alphabet_i] += cur_label_start
        valX.extend(charomniglot_valX[alphabet_i])
        valY.extend(charomniglot_valY[alphabet_i])
        cur_label_start += len(set(charomniglot_valY[alphabet_i]))

    cur_label_start = 0
    for alphabet_i in range(len(charomniglot_testY)):
        charomniglot_testY[alphabet_i] += cur_label_start
        teX.extend(charomniglot_testX[alphabet_i])
        teY.extend(charomniglot_testY[alphabet_i])
        cur_label_start += len(set(charomniglot_testY[alphabet_i]))

    trX = np.asarray(trX, dtype=np.float32) / 255.0
    trY = np.asarray(trY, dtype=np.float32)
    valX = np.asarray(valX, dtype=np.float32) / 255.0
    valY = np.asarray(valY, dtype=np.float32)
    teX = np.asarray(teX, dtype=np.float32) / 255.0
    teY = np.asarray(teY, dtype=np.float32)

    return trX, trY, valX, valY, teX, teY
