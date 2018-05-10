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
from .data_params import *
from .xg_group_params import *
from multiprocessing import Pool

sample = re.compile('^sample.*')
sample_match = np.vectorize(lambda x: bool(sample.match(x)))


class TalkingData:
    def __init__(self):
        self.ratio = test_ratio
        self.train_cols = train_cols
        self.test_cols = test_cols
        self.col_max = OrderedDict()
        self.len_train = (0, 0)
        self.len_test = (0, 0)
        self.len_vld = (0, 0)
        self.sample_cols = []
        self.embedding_cols = []
        self.label = 'is_attributed'

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
        df_train = df_train.sort_values(by='click_time')
        print("Creating new time features: 'hour' and 'day'...")
        dt = df_train["click_time"].dt
        df_train['hour'] = dt.hour.astype('uint8')
        df_train['day'] = dt.day.astype('uint8')
        del dt
        gc.collect()

        df_train = df_train.sort_values(by='click_time')
        df_train = self.get_next_click_time(df=df_train, click_groups=GROUP_BY_NEXT_CLICKS)
        df_train = self.get_agg_features(df=df_train, agg_cols=GROUPBY_AGGREGATIONS)
        # df_train = self.get_click_order(df=df_train, click_act_dict=HISTORY_CLICKS)

        data_train = df_train.iloc[:self.len_train[0]]
        data_test = df_train.iloc[self.len_train[0]:]
        del df_train
        gc.collect()
        assert (data_test.shape[0] == self.len_test[0], 'prepare data error, test data size not eqaul')
        data_train.to_csv(train_file_tmp, index=False)
        data_test.to_csv(test_file_tmp, index=False)
        return data_train, data_test

    @classmethod
    def next_click_time(cls, arg):
        df, spec = arg
        # Name of new feature
        new_feature = 'sample_{}_nextClick'.format('_'.join(spec['groupby']))
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']
        df_new = df[all_features].copy(deep=True)
        del df
        gc.collect()
        # Run calculation
        print(f">> Grouping by {spec['groupby']}, and saving time to next click in: {new_feature}")
        df_new[new_feature] = df_new[all_features].groupby(spec['groupby']).click_time. \
            transform(lambda x: x.diff().shift(-1)).dt.seconds
        df_new = df_new.drop(['click_time'], axis=1)
        df_new[new_feature] = df_new[[new_feature]].fillna(30000000000).apply(lambda x: np.log(x + 1))
        return df_new, spec['groupby']

    @classmethod
    @timeit
    def get_next_click_time(cls, df, click_groups):
        """
        this function used to get next click time
        :param df:
        :param click_groups:
        :return:
        """

        # for spec in click_groups:
        pool = Pool()
        res = pool.map(cls.next_click_time, [(df, i) for i in click_groups])
        pool.close()
        pool.join()
        if 'click_time' in df.columns.tolist():
            df = df.drop(['click_time'], axis=1)
        else:
            pass
        for x, y in res:
            df = df.merge(x, on=y, how='left')
        return df

    @classmethod
    def agg_features(cls, args):
        df, spec = args
        agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']
        # Name of new feature
        new_feature = 'sample_{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])
        # Info
        print("Grouping by {}, and aggregating {} with {}".format(
            spec['groupby'], spec['select'], agg_name
        ))
        # Unique list of features to select
        all_features = list(set(spec['groupby'] + [spec['select']]))
        df_new = df[all_features].copy(deep=True)
        del df
        gc.collect()
        # Perform the groupby
        df_new = df_new[all_features]. \
            groupby(spec['groupby'])[spec['select']]. \
            agg(spec['agg']). \
            reset_index(). \
            rename(index=str, columns={spec['select']: new_feature})
        return df_new, spec['groupby']

    @classmethod
    @timeit
    def get_agg_features(cls, df, agg_cols):
        """

        :param df:
        :param agg_cols:
        :return:
        """
        pool = Pool()
        res = pool.map(cls.agg_features, [(df, i) for i in agg_cols])
        pool.close()
        pool.join()
        if 'click_time' in df.columns.tolist():
            df = df.drop(['click_time'], axis=1)
        else:
            pass
        for x, y in res:
            df = df.merge(x, on=y, how='left')
        return df

    @classmethod
    def click_order(cls, args):
        df, fname, fset = args
        # Clicks in the past
        new_feature = f"prev_{fname}"
        df_new = df[fset].copy(deep=True)
        del df
        gc.collect()
        df_g = df_new.groupby(fset)
        df_new['prev_' + fname] = df_g.cumcount()
        # # Clicks in the future
        # df['future_' + fname] = df_g.\
        # cumcount(ascending=False).\
        # rename('future_' + fname).iloc[::-1]
        return df_new, fset

    @classmethod
    @timeit
    def get_click_order(cls, df, click_act_dict):
        """

        :param df:
        :param click_act_dict:
        :return:
        """
        pool = Pool()
        res = pool.map(cls.click_order, [(df, i, j) for i, j in click_act_dict.items()])
        pool.close()
        pool.join()
        if 'click_time' in df.columns.tolist():
            df = df.drop(['click_time'], axis=1)
        else:
            pass
        for x, y in res:
            df = df.merge(x, on=y, how='left')
        return df

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
