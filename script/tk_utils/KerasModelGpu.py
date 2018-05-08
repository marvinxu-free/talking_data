# -*- coding: utf-8 -*-
# Project: ml_more_algorithm
# Author: chaoxu create this file
# Time: 2018/4/16
# Company : Maxent
# Email: chao.xu@maxent-inc.com
"""
本模型主要根据public kernel 里面的DNN做一些改进和尝试
主要观点：
1. 其实代码复杂度比我的还低
2. 将所有的特征都转为int， 做embedding(后续可以考虑一下做为离散值)
3. 训练数据选择子集
4. dense的activity选择两种激活函数， 让模型自己去选择哪一个激活函数

5. 其它待提升的点：
    1. 加入attend layer

"""

from keras.models import Model
from keras.utils import multi_gpu_model
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten, SpatialDropout1D
from keras.layers import Input, concatenate, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from collections import OrderedDict
from .keras_udf_loss import bin_categorical_crossentropy, wrapped_partial
from .metric_ros import *
from keras.metrics import binary_accuracy


class KerasModel:
    def __init__(self, decay_steps, col_max, sample_features):
        self.col_max = col_max
        self.sample_features = sample_features
        self.decay_steps = decay_steps

        self.embedding_size = {
            'app': 50,
            'channel': 50,
            'device': 50,
            'ip': 50,
            'os': 50,
            'day': 50,
            'wday': 50,
            'hour': 50
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
            # flat = Flatten(name="{0}_flatten_layer".format(i))(embed)
            inputs.append(input_layer)
            models.append((embed))

        # 无需embedding 的layer
        print(f'sample cols is {self.sample_features}')
        for i in self.sample_features:
            input_layer = Input([1], name="{0}_num_layer".format(i))
            inputs.append(input_layer)
            dense_layer = Dense(1, name="{0}_num_dense".format(i))(input_layer)
            models.append(dense_layer)
        main_1 = concatenate(models)
        s_dout = SpatialDropout1D(0.1)(main_1)
        x = Flatten()(s_dout)
        x = Dropout(0.2)(Dense(1000, activation='relu')(x))
        x = Dropout(0.2)(Dense(1000, activation='relu')(x))
        output = Dense(1, activation='sigmoid')(x)

        exp_decay = lambda init, fin, steps: (init / fin) ** (1 / (steps - 1)) - 1
        lr_init, lr_fin = 0.001, 0.0001
        lr_decay = exp_decay(lr_init, lr_fin, self.decay_steps)
        optimizer_adam = Adam(lr=0.001, decay=lr_decay)
        # udf_loss = wrapped_partial(bin_categorical_crossentropy, e1=1.2, e2=0.8)

        model = Model(inputs=inputs,
                      outputs=output)
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(
            loss='binary_crossentropy',
            # loss=udf_loss,
            optimizer=optimizer_adam,
            metrics=[binary_accuracy, binary_FPA, binary_TPA, binary_auc])
        return parallel_model
