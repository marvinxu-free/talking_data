# -*- coding: utf-8 -*-
# Project: maxent-ml
# Author: chaoxu create this file
# Time: 2018/4/16
# Company : Maxent
# Email: chao.xu@maxent-inc.com


def get_time_cat(x):
    """
    get day of month
    :param x:
    :return:
    """
    day = x.day
    weekth = (day - 1) // 7 + 1
    dayofweek = x.dayofweek
    hour = x.hour
    return [weekth, dayofweek, day, hour]