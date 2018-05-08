# -*- coding: utf-8 -*-
# Project: ml_more_algorithm
# Author: chaoxu create this file
# Time: 2018/4/19
# Company : Maxent
# Email: chao.xu@maxent-inc.com
import tensorflow as tf
import keras.backend as K

import numpy as np


def binary_auc(y_true, y_pred):
    """
    AUC for a binary classifier
    :param y_true:
    :param y_pred:
    :return:
    """
    ptas = tf.stack([binary_TPA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.stack([binary_FPA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
    binSizes = -(pfas[1:] - pfas[:-1])
    s = ptas * binSizes
    return K.sum(s, axis=0)


def binary_FPA(y_true, y_pred, threshold=K.variable(value=0.5)):
    """
    FPR alert for binary classifier
    :param y_true:
    :param y_pred:
    :param threshold:
    :return:
    """
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP / N


def binary_TPA(y_true, y_pred, threshold=K.variable(value=0.5)):
    """
    TPR alerts for binary classifier
    :param y_true:
    :param y_pred:
    :param threshold:
    :return:
    """
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP / P
