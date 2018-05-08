# -*- coding: utf-8 -*-
# Project: ml_more_algorithm
# Author: chaoxu create this file
# Time: 2018/4/23
# Company : Maxent
# Email: chao.xu@maxent-inc.com
import numpy as np

GROUPBY_AGGREGATIONS = [
    # new
    {'groupby': ['ip', 'app'], 'select': 'channel', 'agg': 'count'},
    {'groupby': ['ip', 'day', 'hour'], 'select': 'channel', 'agg': 'count'},
    {'groupby': ['ip', 'app', 'os'], 'select': 'channel', 'agg': 'count'},
    # V1 - GroupBy Features #
    #########################
    # Variance in day, for ip-app-channel
    # {'groupby': ['ip', 'app', 'channel'], 'select': 'day', 'agg': 'var'},
    # Variance in hour, for ip-app-os
    # {'groupby': ['ip', 'app', 'os'], 'select': 'hour', 'agg': 'var'},
    # Variance in hour, for ip-day-channel
    # {'groupby': ['ip', 'day', 'channel'], 'select': 'hour', 'agg': 'var'},
    # Count, for ip-day-hour
    # {'groupby': ['ip', 'day', 'hour'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app
    # {'groupby': ['ip', 'app'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app-os
    # {'groupby': ['ip', 'app', 'os'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app-day-hour
    # {'groupby': ['ip', 'app', 'day', 'hour'], 'select': 'channel', 'agg': 'count'},
    # Mean hour, for ip-app-channel
    # {'groupby': ['ip', 'app', 'channel'], 'select': 'hour', 'agg': 'mean'},

    # V2 - GroupBy Features #
    #########################
    # Average clicks on app by distinct users; is it an app they return to?
    # {'groupby': ['app'],
    #  'select': 'ip',
    #  'agg': lambda x: float(len(x)) / len(x.unique()),
    #  'agg_name': 'AvgViewPerDistinct'
    #  },
    # How popular is the app or channel?
    # {'groupby': ['app'], 'select': 'channel', 'agg': 'count'},
    # {'groupby': ['channel'], 'select': 'app', 'agg': 'count'},

    # V3 - GroupBy Features                                              #
    # https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977 #
    ######################################################################
    {'groupby': ['ip'], 'select': 'channel', 'agg': 'nunique'},
    {'groupby': ['ip'], 'select': 'app', 'agg': 'nunique'},
    {'groupby': ['ip', 'day'], 'select': 'hour', 'agg': 'nunique'},
    {'groupby': ['ip', 'app'], 'select': 'os', 'agg': 'nunique'},
    {'groupby': ['ip'], 'select': 'device', 'agg': 'nunique'},
    {'groupby': ['app'], 'select': 'channel', 'agg': 'nunique'},
    {'groupby': ['ip', 'device', 'os'], 'select': 'app', 'agg': 'nunique'},
    # {'groupby': ['ip', 'device', 'os'], 'select': 'app', 'agg': 'cumcount'},
    # {'groupby': ['ip'], 'select': 'app', 'agg': 'cumcount'},
    # {'groupby': ['ip'], 'select': 'os', 'agg': 'cumcount'},
    # {'groupby': ['ip', 'day', 'channel'], 'select': 'hour', 'agg': 'var'}
]

GROUP_BY_NEXT_CLICKS = [

    # V1
    {'groupby': ['ip']},
    {'groupby': ['ip', 'app']},
    {'groupby': ['ip', 'channel']},
    {'groupby': ['ip', 'os']},

    # V3
    {'groupby': ['ip', 'app', 'device', 'os', 'channel']},
    {'groupby': ['ip', 'os', 'device']},
    {'groupby': ['ip', 'os', 'device', 'app']}
]

HISTORY_CLICKS = {
    'identical_clicks': ['ip', 'app', 'device', 'os', 'channel'],
    'app_clicks': ['ip', 'app'],
    'device_clicks': ['ip', 'app', 'device'],
}

ROLLING_BY_TIME = [
    {'groupby': ['ip', 'app'], 'select': 'channel', 'roll': '1H', 'apply': lambda x: np.unique(x).shape[0]},
    {'groupby': ['ip', 'app', 'os'], 'select': 'channel', 'roll': '1H', 'apply': lambda x: np.unique(x).shape[0]},
    {'groupby': ['ip', 'app', 'os', 'device'], 'select': 'channel', 'roll': '1H', 'apply': lambda x: np.unique(x).shape[0]},
]
