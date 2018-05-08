# -*- coding: utf-8 -*-
# Project: ml_more_algorithm
# Author: chaoxu create this file
# Time: 2018/4/16
# Company : Maxent
# Email: chao.xu@maxent-inc.com
import pandas as pd
from sklearn.model_selection import train_test_split
from .params import *
import os
import gc
import numpy as np
from collections import OrderedDict
import re
from .func_timer import timeit
from .data_params import tk_dtypes

sample = re.compile('^sample.*')
sample_match = np.vectorize(lambda x: bool(sample.match(x)))


class TalkingData:
    def __init__(self):
        self.ratio = test_ratio
        self.train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
        self.test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_tme']
        # self.new_cols_tuple = [
        #     (['ip'], ['app','channel', 'device']),
        #     (['ip', 'app'], ['os']),
        #     (['ip'], ['day', 'hour']),
        #     (['app'], ['channel']),
        # ]
        self.next_click_cols = [
            ['ip'],
            ['ip', 'app'],
            ['ip', 'channel'],
            ['ip', 'os'],
            ['ip', 'app', 'device', 'os', 'channel'],
            ['ip', 'os', 'device'],
            ['ip', 'os', 'device', 'app']
        ]
        self.new_cols_tuple = [
            ['ip', 'day', 'hour'],
            ['ip', 'app'],
            ['ip', 'app', 'os'],
            ['ip', 'device'],
            ['app', 'channel'],
            ['ip', 'channel']
        ]
        self.unique_cols = [
            (['ip', 'day', 'hour'], ['app', 'channel', 'os', 'device']),
        ]
        self.col_max = OrderedDict()
        self.len_train = (0, 0)
        self.len_test = (0, 0)
        self.len_vld = (0, 0)
        self.sample_cols = []
        self.embedding_cols = []
        self.label = 'is_attributed'
        # self._train_size = 0
        # self._vld_size = 0
        # self._test_size = 0

    @timeit
    def prepare_data(self):
        if os.path.exists(train_file_tmp) and os.path.exists(test_file_tmp):
            train_data = pd.read_csv(train_file_tmp)
            test_data = pd.read_csv(test_file_tmp)
        else:
            train_data, test_data = self.feature_more()

        self.len_train = train_data.shape
        self.len_test = test_data.shape
        self.sample_cols = train_data.columns.values[sample_match(train_data.columns.values)].tolist()
        # self.sample_cols.append(self.label)
        self.embedding_cols = train_data.columns.difference(self.sample_cols).difference([self.label]).tolist()

        for col in self.embedding_cols:
            self.col_max[col] = max(train_data[col].max(), test_data[col].max()) + 1
        self.col_max['hour'] = 24
        self.col_max['day'] = 31

        return train_data, test_data

    @timeit
    def feature_more(self):
        print("loading data...")
        df_train = pd.read_csv(train_file,
                               dtype=tk_dtypes,
                               header=0,
                               usecols=self.train_cols,
                               # nrows=1000,
                               parse_dates=["click_time"])  # .sample(1000)
        df_test = pd.read_csv(test_file,
                              dtype=tk_dtypes,
                              header=0,
                              # nrows=1000,
                              usecols=self.test_cols,
                              parse_dates=["click_time"])
        self.len_train = df_train.shape
        self.len_test = df_test.shape
        df_train = df_train.append(df_test)
        del df_test
        gc.collect()
        print("Creating new time features: 'hour' and 'day'...")
        dt = df_train["click_time"].dt
        df_train['hour'] = dt.hour.astype('uint8')
        df_train['day'] = dt.day.astype('uint8')
        del dt
        gc.collect()

        # sample_ranges = ['15Min', '30Min', '1H', '1D']
        sample_ranges = ['30Min']
        # df_train = df_train.set_index('click_time').sort_index()
        # for sample_unit in sample_ranges:
        #     for gcol, fcol in self.new_cols_tuple:
        #         # df_new = self.sample_new_features(df=df_train,
        #         #                                   col=gcol,
        #         #                                   feature_cols=fcol,
        #         #                                   time_range=sample_unit
        #         #                                   )
        df_train = self.get_next_click_time(df=df_train, cols=self.next_click_cols)
        df_train = df_train.drop(['click_time'], axis=1)
        # df_train = df_train.set_index('click_time').sort_index()
        # for gcol, fcol in self.new_cols_tuple:
        #     df_new = self.count_new_features(df=df_train,
        #                                      col=gcol,
        #                                      feature_cols=fcol,
        #                                      )
        #     df_train = pd.concat([df_train, df_new], axis=1)
        #     del df_new
        #     gc.collect()
        # df_train = df_train.reset_index(drop=True)

        for g_col in self.new_cols_tuple:
            df_size = self.size_new_features(df=df_train, col=g_col)
            df_train = df_train.merge(df_size, on=g_col, how='left')

        # a = df_train.apply(pd.Series.max)
        # self.col_max.update(dict(zip(a.index, a.values)))

        data_train = df_train.iloc[:self.len_train[0]]
        data_test = df_train.iloc[self.len_train[0]:]
        del df_train
        gc.collect()
        assert (data_test.shape[0] == self.len_test[0], 'prepare data error, test data size not eqaul')
        data_train.to_csv(train_file_tmp, index=False)
        data_test.to_csv(test_file_tmp, index=False)
        return data_train, data_test

    @classmethod
    @timeit
    def get_next_click_time(cls, df, cols):
        """
        this function used to get next click time
        :param df:
        :param cols:
        :return:
        """
        for col in cols:
            new_feature = "sample_{0}_nextClickTime".format("_".join(col))
            df[new_feature] = df.groupby(col)['click_time'].transform(lambda x: x.diff().shift(-1)).dt.seconds
            df[new_feature] = df[new_feature].fillna(30000000000).apply(lambda x: np.log(x + 1)).astype(int)
        return df

    @classmethod
    @timeit
    def size_new_features(cls, df, col=[]):
        """

        :param df:
        :param col:
        :return:
        """
        cols_str = 'sample_' + "_".join(col) + '_size'
        print(f"GEt group size of {cols_str}")
        df = df.groupby(col).size().reset_index(name=cols_str)
        df[cols_str] = df[cols_str].apply(lambda x: np.log(x + 1))
        return df

    @classmethod
    @timeit
    def count_new_features(cls, df, col=[], feature_cols=[]):
        """

        :param df:
        :param col:
        :param time_range:
        :param feature_cols:
        :return:
        """
        print(col, feature_cols)
        df_g = df.groupby(col)[feature_cols].agg(dict([(x, 'count') for x in feature_cols])) \
            .applymap(lambda x: np.log(x + 1))
        df_g = df_g.reset_index(col, drop=True)
        cols_str = "_".join(col)
        feature_cols_str = "_".join(feature_cols)
        rename_cols = {x: f"sample_{cols_str}_count_{feature_cols_str}" for x in feature_cols}
        df_g = df_g.rename(columns=rename_cols)
        df_g = df_g.astype(int)
        return df_g

    @classmethod
    @timeit
    def sample_new_features(cls, df, col=[], time_range='7D', feature_cols=[]):
        """
        this function usde to get login features
        :param df:
        :param col:
        :param feature_cols:
        :param time_range:
        :return:
        """
        # df = df.sort_index(ascending=False)
        print(col, feature_cols)
        df_g = df.groupby(col)[feature_cols].rolling(time_range) \
            .apply(lambda x: np.unique(x).shape[0]) \
            .apply(lambda x: np.log(x))
        df_g = df_g.reset_index(col, drop=True).sort_index()
        cols_str = "_".join(col)
        feature_cols_str = "_".join(feature_cols)
        rename_cols = {x: "sample_{0}_per_{1}_nuique_{2}".format(cols_str, x, time_range, feature_cols_str) for x in
                       feature_cols}
        df_g = df_g.rename(columns=rename_cols)
        df_g = df_g.astype('uint16')
        return df_g

    def __call__(self, mode, **kwargs):
        data_train, data_test = self.prepare_data()
        if mode == 'train':
            print('get data for training...')
            del data_test
            gc.collect()
            X = data_train[data_train.columns.difference([self.label])]
            y = data_train[self.label]
            X_train, X_vld, y_train, y_vld = train_test_split(
                X,
                y,
                stratify=y,
                test_size=self.ratio,
                random_state=29
            )
            self.len_vld = X_vld.shape
            return X_train, X_vld, y_train, y_vld
        else:
            print('get data for test...')
            del data_train
            gc.collect()
            return data_test


if __name__ == '__main__':
    data_source = TalkingData(0.1)
    X_train, y_train, X_vld, y_vld = data_source(mode='train')
