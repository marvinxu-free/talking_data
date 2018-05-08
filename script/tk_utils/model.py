# -*- coding: utf-8 -*-
# Project: ml_more_algorithm
# Author: chaoxu create this file
# Time: 2018/4/16
# Company : Maxent
"""
本文件主要用于搭建模型
"""

import numpy as np
import gc

seed = 7
np.random.seed(seed)

# from .Tdata import TalkingData
# from .TXGdata import TalkingData

from .AllEmbedding import TalkingData
# from .KerasModel import KerasModel
# from .KerasModelV1 import KerasModel
# from .KerasModelGpu import KerasModel

from .roc_call_backs import ROCCallback, ROCCallbackVld
from .time_call_back import TimeHistory
from sklearn.metrics import classification_report
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
from .metric_ros import *
from .func_timer import timeit
from .pic_keras import pic_history
import ast


class Model():
    def __init__(self, section):
        self.model = None
        self.data_source = TalkingData(section)
        self.predict_report_file = section.get('predict_report_file')
        self.best_weights_file = section.get('best_weights_file')
        self.Data_path = section.get('Data_path')
        self.epoch_num = int(section.get('epoch_num'))
        self.batch_size = int(section.get('batch_size'))
        self.pick_hours = ast.literal_eval(section.get('pick_hours'))
        self.model_file = section.get('model_file', 'KerasModelV1')

        self.metric_rpt = [
            ('binary_accuracy', u'训练集准确率'),
            ('binary_FPA', u'训练集FPR'),
            ('binary_TPA', u'训练集TPR'),
            ('binary_auc', u'训练集AUC'),
            ('val_binary_accuracy', u'验证集准确率'),
            ('val_binary_FPA', u'验证集FPR'),
            ('val_binary_TPA', u'验证集TPR'),
            ('val_binary_auc', u'验证集AUC'),
        ]

    @property
    def train_size(self):
        return self.data_source.len_train

    @property
    def vld_size(self):
        return self.data_source.len_vld

    @property
    def test_size(self):
        return self.data_source.len_test

    @timeit
    def build_model(self):
        decay_steps = int(self.train_size[0] / self.batch_size) * self.epoch_num
        print(
            f'build model {self.model_file} for keras:\n\tepoch: {self.epoch_num}\n\tbatch size: '
            f'{self.batch_size}\n\t decay steps: {decay_steps}')
        exec(f'from .{self.model_file} import KerasModel')
        self.model = eval(
            'KerasModel(decay_steps=decay_steps, '
            'col_max=self.data_source.col_max,sample_features=self.data_source.sample_cols)()')
        self.modul_summary()
        # self.plot_model()

    @timeit
    def modul_summary(self):
        if self.model is not None:
            print(self.model.summary())
        else:
            print('model is None')

    @timeit
    def plot_model(self):
        if self.model is not None:
            plot_model(self.model, to_file="{0}/model_arch.png".format(self.Data_path))
        else:
            print('model is None')

    @timeit
    def fit(self):
        X_train, X_vld, y_train, y_vld = self.data_source(mode='train')
        # X_train, y_train = self.data_source(mode='train')
        X_train_list = []
        X_vld_list = []
        # y_weight = np.where(y_train == 1, 1.0, 0.2)
        # weights = y_weight * X_train['hour'].apply(lambda x: 1.0 if x in self.pick_hours else 0.5)
        self.order_list = self.data_source.embedding_cols + self.data_source.sample_cols
        for col in self.order_list:
            X_train_list.append(X_train[col].values.reshape((-1, 1)))
            X_vld_list.append(X_vld[col].values.reshape((-1, 1)))
        del (X_train, X_vld)
        # del (X_train)
        gc.collect()
        self.build_model()
        # model_chk = ModelCheckpoint(self.best_weights_file, monitor='binary_accuracy', verbose=1,
        #                             save_best_only=True,
        #                             mode='max')
        # roc_rpt = ROCCallback(training_data=(X_train_list, y_train))
        # roc_rpt = ROCCallbackVld(training_data=(X_train_list, y_train), validation_data=(X_vld_list, y_vld))
        auc_stop = EarlyStopping(monitor='vld_binary_auc', min_delta=0.0001, patience=1, verbose=1, mode='max')
        time_rpts = TimeHistory()

        call_backs = [auc_stop, time_rpts]
        self.model.fit(x=X_train_list,
                       y=y_train,
                       batch_size=self.batch_size,
                       epochs=self.epoch_num,
                       verbose=1,
                       validation_data=(X_vld_list, y_vld),
                       shuffle=True,
                       # class_weight=weights,
                       callbacks=call_backs
                       )
        epoch_times = time_rpts.times
        print(f"all epoch used time is {sum(epoch_times)} s")
        self.model.save(f'{self.best_weights_file}')
        # pic_history(history=history.history, metrics_reports=self.metric_rpt, title='Metric Report',
        #             img_file="{0}/metrics.png".format(Data_path))
        # del (X_train_list)
        # gc.collect()
        # y_pred_prob = self.model.predict(X_vld_list)
        # del (X_vld_list)
        # gc.collect()
        # y_pred_prob = y_pred_prob.reshape((y_pred_prob.shape[0],))
        # y_pred = np.round(y_pred_prob)
        # print(classification_report(y_vld, y_pred, target_names=['0', '1']))

    @timeit
    def predict(self):
        X_test = self.data_source(mode='test')
        X_test_list = []
        self.order_list = self.data_source.embedding_cols + self.data_source.sample_cols
        for col in self.order_list:
            X_test_list.append(X_test[col].values.reshape((-1, 1)))
        if self.model is None:
            print('load model from {0}'.format(self.best_weights_file))
            self.build_model()
            self.model.load_weights(self.best_weights_file)

        y_pred_test_prob = self.model.predict(X_test_list, batch_size=self.batch_size, verbose=1)
        # y_pred_test_prob = y_pred_test_prob.reshape((y_pred_test_prob.shape[0],))
        # y_pred_size = y_pred_test_prob.size
        # data_dict = {
        #     'click_id': np.arange(y_pred_size),
        #     'is_attributed': y_pred_test_prob
        # }
        test_df = pd.DataFrame()
        test_df['click_id'] = X_test['click_id'].astype(int)
        test_df['is_attributed'] = y_pred_test_prob
        del X_test, X_test_list
        gc.collect()
        test_df.to_csv(self.predict_report_file, index=False)
        print('save predict file {0}'.format(self.predict_report_file))
