# -*- coding: utf-8 -*-
# Project: ml_more_algorithm
# Author: chaoxu create this file
# Time: 2018/4/16
# Company : Maxent
# Email: chao.xu@maxent-inc.com

import os

module_path = os.path.split(os.path.realpath(__file__))[0]
Data_path = os.path.realpath("{0}/../../input".format(module_path))
train_file = '{0}/train_new_subsample.csv'.format(Data_path)
train_file_tmp = '{0}/train_subsample_tmp.csv'.format(Data_path)
test_file = '{0}/test.csv'.format(Data_path)
test_file_tmp = '{0}/test_tmp.csv'.format(Data_path)

predict_report_file = '{0}/predict.csv'.format(Data_path)
best_weights_file = '{0}/best.hdf5'.format(Data_path)

batchsize = 10000000
D = 2 ** 20
