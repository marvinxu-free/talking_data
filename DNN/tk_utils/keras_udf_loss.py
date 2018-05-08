# -*- coding: utf-8 -*-
# Project: local-spark
# Author: chaoxu create this file
# Time: 2018/2/9
# Company : Maxent
# Email: chao.xu@maxent-inc.com
import keras.backend as K
from functools import partial, update_wrapper


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def bin_categorical_crossentropy(y_true, y_pred, e1=1.2, e2=0.1):
    """
    this function used to define user defined loss funtion,
    which will pay more attention on error match
    :param y_true:
    :param y_pred:
    :param e1:
    :param e2:
    :return:
    """
    y_pred_label = K.round(K.clip(y_pred, K.epsilon(), 1 - K.epsilon()))
    y_match = K.equal(y_true, y_pred_label)
    y_not_match = K.not_equal(y_true, y_pred_label)
    y_cross = K.cast(y_match, dtype=K.floatx()) * e1 + K.cast(y_not_match, dtype=K.floatx()) * e2 + K.epsilon()
    return K.binary_crossentropy(y_true, y_pred) * y_cross
