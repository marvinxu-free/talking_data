# -*- coding: utf-8 -*-
# Project: local-spark
# Author: chaoxu create this file
# Time: 2018/2/6
# Company : Maxent
# Email: chao.xu@maxent-inc.com
import keras.backend as K


def pos_precision(y_true, y_pred):
    # Calculates the precision
    y_true_mask = K.cast(K.equal(y_true, 1), dtype=K.floatx())
    true_positives = K.sum(K.round(K.clip(y_true * y_pred * y_true_mask, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred * y_true_mask, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def pos_recall(y_true, y_pred):
    # Calculates the recall
    y_true_mask = K.cast(K.equal(y_true, 1), dtype=K.floatx())
    true_positives = K.sum(K.round(K.clip(y_true * y_pred * y_true_mask, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true * y_true_mask, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def pos_fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = pos_precision(y_true, y_pred)
    r = pos_recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def pos_fmeasure(y_true, y_pred):
    '''Calculates the f-measure, the harmonic mean of precision and recall.
    '''
    return pos_fbeta_score(y_true, y_pred, beta=0.1)


def f_beta(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    fbeta = 1.01 ** 2 * precision * recall / (0.01 * precision + recall)
    return fbeta


def pos_precision_2d(y_true, y_pred):
    # Calculates the precision
    y_true = y_true[:, 1]
    y_pred = y_pred[:, 1]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def pos_recall_2d(y_true, y_pred):
    # Calculates the recall
    y_true = y_true[:, 1]
    y_pred = y_pred[:, 1]
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# def fmeasure(y_true, y_pred):
#     # Calculates the f-measure, the harmonic mean of precision and recall.
#     return fbeta_score(y_true, y_pred, beta=1)
