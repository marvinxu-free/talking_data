# -*- coding: utf-8 -*-
# Project: ml_more_algorithm
# Author: chaoxu create this file
# Time: 2018/4/24
# Company : Maxent
# Email: chao.xu@maxent-inc.com
import numpy as np
from .sys_help import *
import pandas as pd


def df_add_counts(df, cols, tag="_count"):
    arr_slice = df[cols].values
    unq, unqtags, counts = np.unique(np.ravel_multi_index(arr_slice.T, arr_slice.max(0) + 1),
                                     return_inverse=True, return_counts=True)
    df["_".join(cols) + tag] = counts[unqtags]
    return df


def df_add_uniques(df, cols, tag="_unique"):
    gp = df[cols].groupby(by=cols[0:len(cols) - 1])[cols[len(cols) - 1]].nunique().reset_index(). \
        rename(index=str, columns={cols[len(cols) - 1]: "_".join(cols) + tag})
    df = df.merge(gp, on=cols[0:len(cols) - 1], how='left')
    return df


def df2csr(df, pick_hours=None):
    df.reset_index(drop=True, inplace=True)
    with timer("Adding counts"):
        df['click_time'] = pd.to_datetime(df['click_time'])
        dt = df['click_time'].dt
        df['day'] = dt.day.astype('uint8')
        df['hour'] = dt.hour.astype('uint8')
        del (dt)
        df = df_add_counts(df, ['ip', 'day', 'hour'])
        df = df_add_counts(df, ['ip', 'app'])
        df = df_add_counts(df, ['ip', 'app', 'os'])
        df = df_add_counts(df, ['ip', 'device'])
        df = df_add_counts(df, ['app', 'channel'])
        df = df_add_uniques(df, ['ip', 'channel'])

    with timer("Adding next click times"):
        D = 2 ** 26
        df['category'] = (df['ip'].astype(str) + "_" + df['app'].astype(str) + "_" + df['device'].astype(str) \
                          + "_" + df['os'].astype(str)).apply(hash) % D
        click_buffer = np.full(D, 3000000000, dtype=np.uint32)
        df['epochtime'] = df['click_time'].astype(np.int64) // 10 ** 9
        next_clicks = []
        for category, time in zip(reversed(df['category'].values), reversed(df['epochtime'].values)):
            next_clicks.append(click_buffer[category] - time)
            click_buffer[category] = time
        del (click_buffer)
        df['next_click'] = list(reversed(next_clicks))

    with timer("Log-binning features"):
        for fea in ['ip_day_hour_count', 'ip_app_count', 'ip_app_os_count', 'ip_device_count',
                    'app_channel_count', 'next_click', 'ip_channel_unique']:
            df[fea] = np.log2(1 + df[fea].values).astype(int)

    with timer("Generating str_array"):
        str_array = (
                # "I" + df['ip'].astype(str) \
                " A" + df['app'].astype(str) \
                + " D" + df['device'].astype(str) \
                + " O" + df['os'].astype(str) \
                + " C" + df['channel'].astype(str) \
                + " WD" + df['day'].astype(str) \
                + " H" + df['hour'].astype(str) \
                + " AXC" + df['app'].astype(str) + "_" + df['channel'].astype(str) \
                + " OXC" + df['os'].astype(str) + "_" + df['channel'].astype(str) \
                + " AXD" + df['app'].astype(str) + "_" + df['device'].astype(str) \
                + " IXA" + df['ip'].astype(str) + "_" + df['app'].astype(str) \
                + " AXO" + df['app'].astype(str) + "_" + df['os'].astype(str) \
                + " IDHC" + df['ip_day_hour_count'].astype(str) \
                + " IAC" + df['ip_app_count'].astype(str) \
                + " AOC" + df['ip_app_os_count'].astype(str) \
                + " IDC" + df['ip_device_count'].astype(str) \
                + " AC" + df['app_channel_count'].astype(str) \
                + " NC" + df['next_click'].astype(str) \
                + " ICU" + df['ip_channel_unique'].astype(str)
        ).values
    # cpuStats()
    if 'is_attributed' in df.columns:
        labels = df['is_attributed'].values
        weights = np.multiply([1.0 if x == 1 else 0.2 for x in df['is_attributed'].values],
                              df['hour'].apply(lambda x: 1.0 if x in pick_hours else 0.5))
    else:
        labels = []
        weights = []
    return str_array, labels, weights
