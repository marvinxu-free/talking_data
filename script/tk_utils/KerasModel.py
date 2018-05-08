# -*- coding: utf-8 -*-
# Project: ml_more_algorithm
# Author: chaoxu create this file
# Time: 2018/4/16
# Company : Maxent
# Email: chao.xu@maxent-inc.com

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten
from keras.layers import Input, concatenate, LSTM
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from collections import OrderedDict
from .keras_udf_loss import bin_categorical_crossentropy, wrapped_partial
from .metric_ros import *
from keras.metrics import binary_accuracy


class KerasModel:
    def __init__(self, col_max, sample_features):
        self.col_max = col_max
        self.sample_features = sample_features

        self.embedding_size = {
            'app': 50,
            'channel': 20,
            'device': 50,
            'ip': 100,
            'os': 50,
            'day': 5,
            'dayofweek': 3,
            'hour': 5
        }

    def __call__(self, *args, **kwargs):
        models = []
        inputs = []

        # 需要做embedding的输入
        for i in self.col_max:
            # print(i, self.col_max[i])
            input_layer = Input([1], name="{0}_embedding_input_layer".format(i))
            emb_size = self.embedding_size.get(i, 50)
            if emb_size is None:
                print('embedding {0} size not defined'.format(i))
            embed = Embedding(self.col_max[i], emb_size, name="{0}_embedding_layer".format(i))(input_layer)
            flat = Flatten(name="{0}_flatten_layer".format(i))(embed)
            inputs.append(input_layer)
            models.append(flat)

        # 无需embedding 的layer
        for i in self.sample_features:
            input_layer = Input([1], name="{0}_num_layer".format(i))
            inputs.append(input_layer)
            dense_layer = Dense(1, name="{0}_num_dense".format(i))(input_layer)
            models.append(dense_layer)

        main_1 = concatenate(models)
        main_2 = Dropout(0.2)(Dense(256)(main_1))
        main_3 = Dropout(0.2)(Dense(128)(main_2))
        output = Dense(1, activation='sigmoid')(main_3)

        model = Model(inputs=inputs,
                      outputs=output)

        udf_loss = wrapped_partial(bin_categorical_crossentropy, e1=1.2, e2=0.8)
        model.compile(loss=udf_loss,
                      optimizer='adam',
                      metrics=[binary_accuracy, binary_FPA, binary_TPA, binary_auc])
        return model

