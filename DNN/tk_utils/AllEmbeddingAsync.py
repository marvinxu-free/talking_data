# -*- coding: utf-8 -*-
# Project: ml_more_algorithm
# Author: chaoxu create this file
# Time: 2018/4/16
# Company : Maxent
# Email: chao.xu@maxent-inc.com
import pandas as pd
# import ray.dataframe as pd
from sklearn.model_selection import train_test_split
import os
import gc
import numpy as np
from collections import OrderedDict
import re
from .func_timer import timeit
from .xg_group_params import *
import ast
import json
from sklearn.preprocessing import LabelEncoder
import mlcrate as mlc

import multiprocessing as mp

sample = re.compile('^sample.*')
sample_match = np.vectorize(lambda x: bool(sample.match(x)))


class TalkingData:
    def __init__(self, section):
        self.section = section
        self.ratio = float(section.get('test_ratio'))
        # self.train_data_size = int(section.get('train_data_size'))
        # self.train_size = int(section.get('train_size'))
        # self.train_vld_size = int(section.get('train_vld_size'))
        self.train_cols = ast.literal_eval(section.get('train_cols'))
        self.test_cols = ast.literal_eval(section.get('test_cols'))
        self.use_cols = ast.literal_eval(section.get('use_cols'))
        self.label_cols = ast.literal_eval(section.get('label_cols'))
        self.tk_dtypes = json.loads(section.get('dtypes'))
        if section.get('skip_rows') != "":
            self.skip_scope = ast.literal_eval(section.get('skip_rows'))
            self.skip_rows = range(self.skip_scope[0], self.skip_scope[1])
        else:
            self.skip_rows = None

        if section.get('n_rows') != "":
            self.n_rows = int(section.get('n_rows'))
        else:
            self.n_rows = None

        self.col_max = OrderedDict()
        self.len_train = (0, 0)
        self.len_test = (0, 0)
        self.len_vld = (0, 0)
        self.sample_cols = []
        self.embedding_cols = []
        self.label = 'is_attributed'

    @timeit
    def prepare_data(self):
        if os.path.exists(self.section.get('train_file_tmp')) and os.path.exists(self.section.get('test_file_tmp')):
            train_data = mlc.load(self.section.get('train_file_tmp'))
            test_data = mlc.load(self.section.get('test_file_tmp'))
        else:
            train_data, test_data = self.feature_more()

        train_data = train_data.drop('ip', axis=1)
        self.len_train = train_data.shape
        self.len_test = test_data.shape
        self.sample_cols = train_data.columns.values[sample_match(train_data.columns.values)].tolist()
        # self.embedding_cols = train_data.columns.difference(self.sample_cols).difference([self.label]).tolist()
        self.embedding_cols = train_data.columns.difference([self.label, 'click_id']).tolist()

        for col in self.embedding_cols:
            self.col_max[col] = max(train_data[col].max(), test_data[col].max()) + 1
        # self.col_max['hour'] = 24
        # self.col_max['day'] = 31
        # self.col_max['wday'] = 7

        return train_data, test_data

    @timeit
    def feature_more(self):
        print("loading data...")
        df_train = pd.read_csv(self.section.get('train_file'),
                               dtype=self.tk_dtypes,
                               usecols=self.train_cols,
                               skiprows=self.skip_rows,
                               nrows=self.n_rows,
                               parse_dates=["click_time"])
        df_test = pd.read_csv(self.section.get('test_file'),
                              dtype=self.tk_dtypes,
                              nrows=self.n_rows,
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
        df_train['wday'] = dt.dayofweek.astype('uint8')
        del dt
        gc.collect()

        # df_train = df_train.set_index('click_time').sort_index()
        df_train = self.get_agg_features(df=df_train, agg_cols=GROUPBY_AGGREGATIONS)
        df_train = self.get_click_order(df=df_train, click_act_dict=HISTORY_CLICKS)
        # df_train = self.get_next_click_time(df=df_train, click_groups=GROUP_BY_NEXT_CLICKS)
        # df_train = self.get_pre_click_time(df=df_train, click_groups=GROUP_BY_NEXT_CLICKS)
        df_train = self.get_roll_features(df=df_train, roll_cols=ROLLING_BY_TIME)
        print(f'begin to do LabelEncoder for {self.label_cols}')
        df_train[self.label_cols].apply(LabelEncoder().fit_transform)
        df_train = df_train.reset_index('click_time', drop=True)

        print('get train and test dataset')
        data_train = df_train.iloc[:self.len_train[0]]
        data_test = df_train.iloc[self.len_train[0]:]
        del df_train
        gc.collect()
        assert (data_test.shape[0] == self.len_test[0], 'prepare data error, test data size not eqaul')
        # data_train.to_csv(self.section.get('train_file_tmp'), index=False)
        # data_test.to_csv(self.section.get('test_file_tmp'), index=False)
        mlc.save(data_train, self.section.get('train_file_tmp'))
        mlc.save(data_test, self.section.get('test_file_tmp'))
        return data_train, data_test

    @classmethod
    @timeit
    def get_next_click_time(cls, df, click_groups):
        """
        this function used to get next click time
        :param df:
        :param click_groups:
        :return:
        """
        df = df.sort_values(by='click_time')
        for spec in click_groups:
            # Name of new feature
            new_feature = 'sample_{}_nextClick'.format('_'.join(spec['groupby']))

            # Unique list of features to select
            all_features = spec['groupby'] + ['click_time']

            # Run calculation
            print(f">> Grouping by {spec['groupby']}, and saving time to next click in: {new_feature}")
            df[new_feature] = df[all_features].groupby(spec['groupby']).click_time.transform(
                lambda x: x.diff().shift(-1)).dt.seconds
            df[new_feature] = df[[new_feature]].fillna(30000000000).apply(lambda x: np.log(x + 1))
            gc.collect()
        return df

    @classmethod
    @timeit
    def get_pre_click_time(cls, df, click_groups):
        """
        this function used to get next click time
        :param df:
        :param click_groups:
        :return:
        """
        # df = df.sort_values(by='click_time')
        df = df.sort_values(by='click_time')
        for spec in click_groups:
            # Name of new feature
            new_feature = 'sample_{}_preClick'.format('_'.join(spec['groupby']))

            # Unique list of features to select
            all_features = spec['groupby'] + ['click_time']

            # Run calculation
            print(f">> Grouping by {spec['groupby']}, and saving time to pre-click in: {new_feature}")
            df[new_feature] = df[all_features].groupby(spec['groupby']).click_time.transform(
                lambda x: 0 - x.diff().shift(1)).dt.seconds
            df[new_feature] = df[[new_feature]].fillna(30000000000).apply(lambda x: np.log(x + 1))
            gc.collect()
        return df

    @classmethod
    def roll_func(cls, args):
        df, spec = args
        roll_func = spec['apply']
        roll_time = spec['roll']
        new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), roll_time, spec['select'])
        print(f"Grouping by {spec['groupby']}, and rolling {spec['select']} with {roll_time} and apply {roll_func}")
        all_features = list(set(spec['groupby'] + [spec['select']]))
        gp = df[all_features].groupby(spec['groupby'])[spec['select']] \
            .rolling(roll_time).apply(eval(roll_func)) \
            .reset_index(spec['groupby'], drop=True) \
            .sort_index().rename(columns={spec['select']: new_feature}).astype(spec['dtype'])
        return gp, new_feature

    @classmethod
    @timeit
    def get_roll_features(cls, df, roll_cols):
        # df = df.sort_index()
        # for spec in roll_cols:
        with mp.Pool() as pool:
            results = pool.map(cls.roll_func, [(df, i) for i in roll_cols])
        for x, y in results:
            df[y] = x
        return df

    @classmethod
    def agg_func(cls, args):
        # Name of the aggregation we're applying
        df, spec = args
        agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']
        # Name of new feature
        new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])
        # Info
        print("Grouping by {}, and aggregating {} with {}".format(
            spec['groupby'], spec['select'], agg_name
        ))
        # Unique list of features to select
        all_features = list(set(spec['groupby'] + [spec['select']]))
        # Perform the groupby
        gp = df[all_features]. \
            groupby(spec['groupby'])[[spec['select']]]. \
            agg(spec['agg']). \
            reset_index(spec['groupby']). \
            rename(columns={spec['select']: new_feature})
            # drop(spec['groupby'], axis=1)
        gp[new_feature] = gp[new_feature].astype(spec['dtype'])
        return gp, spec['groupby']

    @classmethod
    @timeit
    def get_agg_features(cls, df, agg_cols):
        """
        :param df:
        :param agg_cols:
        :return:
        """
        # df = df.reset_index()

        # for spec in agg_cols:
        with mp.Pool() as pool:
            results = pool.map(cls.agg_func, [(df, i) for i in agg_cols])
        for x, y in results:
            df = df.merge(x, on=y, how='left')
            # df[y] = x[y]
        df = df.set_index('click_time').sort_index()
        return df

    @classmethod
    def order_func(cls, args):
        # Clicks in the past
        df, spec = args
        new_feature = '{}_click_order'.format('_'.join(spec['groupby']))
        all_features = list(set(spec['groupby']))
        gp = df[all_features]. \
            groupby(all_features). \
            cumcount(). \
            rename(new_feature).astype(spec['dtype'])
        return gp, new_feature

    @classmethod
    @timeit
    def get_click_order(cls, df, click_act_dict):
        """
        :param df:
        :param click_act_dict:
        :return:
        """
        df = df.sort_index()
        # for spec in click_act_dict.items():

        with mp.Pool() as pool:
            results = pool.map(cls.order_func, [(df, i) for i in click_act_dict])
        for x, y in results:
            df[y] = x
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
                # stratify=y,
                test_size=self.ratio,
                random_state=42
            )
            self.len_vld = X_vld.shape
            return X_train, X_vld, y_train, y_vld
            # return X, y
        else:
            print('get data for test...')
            del data_train
            gc.collect()
            return data_test


if __name__ == '__main__':
    data_source = TalkingData(0.1)
    X_train, y_train, X_vld, y_vld = data_source(mode='train')
