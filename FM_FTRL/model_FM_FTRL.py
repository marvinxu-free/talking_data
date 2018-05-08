# -*- coding: utf-8 -*-
# Project: ml_more_algorithm
# Author: chaoxu create this file
# Time: 2018/4/20
# Company : Maxent
# Email: chao.xu@maxent-inc.com

import gc
import wordbatch
from wordbatch.extractors import WordHash
from wordbatch.models import FM_FTRL
# from wordbatch.data_utils import *
import threading
import pandas as pd
from sklearn.metrics import roc_auc_score
from ftrl_utils.params import *
from ftrl_utils.sys_help import *
from ftrl_utils.get_features import df2csr
from ftrl_utils.data_params import *
from ftrl_utils.run_thread import ThreadWithReturnValue

start_time = time.time()
mean_auc = 0


def fit_batch(clf, X, y, w):  clf.partial_fit(X, y, sample_weight=w)


def predict_batch(clf, X):  return clf.predict(X)


def evaluate_batch(clf, X, y, rcount):
    auc = roc_auc_score(y, predict_batch(clf, X))
    global mean_auc
    if mean_auc == 0:
        mean_auc = auc
    else:
        mean_auc = 0.2 * (mean_auc * 4 + auc)
    print(rcount, "ROC AUC:", auc, "Running Mean:", mean_auc)
    return auc


wb = wordbatch.WordBatch(None, extractor=(WordHash, {"ngram_range": (1, 1), "analyzer": "word",
                                                     "lowercase": False, "n_features": D,
                                                     "norm": None, "binary": True})
                         , minibatch_size=batchsize // 80, procs=8, freeze=True, timeout=1800, verbose=0)

clf = FM_FTRL(alpha=0.05, beta=0.1, L1=0.0, L2=0.0, D=D, alpha_fm=0.02, L2_fm=0.0, init_fm=0.01, weight_fm=1.0,
              D_fm=8, e_noise=0.0, iters=2, inv_link="sigmoid", e_clip=1.0, threads=8, use_avx=1, verbose=0)

p = None
rcount = 0
for df_c in pd.read_csv('{0}/train.csv'.format(Data_path), engine='c', chunksize=batchsize,
                        skiprows=range(1, 9308569), sep=",", dtype=dtypes):
    rcount += batchsize
    if rcount == 130000000:
        df_c['click_time'] = pd.to_datetime(df_c['click_time'])
        df_c['day'] = df_c['click_time'].dt.day.astype('uint8')
        df_c = df_c[df_c['day'] == 8]
    str_array, labels, weights = df2csr(df_c, pick_hours={4, 5, 10, 13, 14})
    del df_c
    if p is not None:
        p.join()
        del X
    gc.collect()
    X = wb.transform(str_array)
    del str_array
    if rcount % (2 * batchsize) == 0:
        if p is not None:
            p.join()
        p = threading.Thread(target=evaluate_batch, args=(clf, X, labels, rcount))
        p.start()
    print("Training", rcount, time.time() - start_time)
    cpuStats()
    if p is not None:
        p.join()
    p = threading.Thread(target=fit_batch, args=(clf, X, labels, weights))
    p.start()
    if rcount == 130000000:
        break

if p is not None:
    p.join()

del (X)
p = None
click_ids = []
test_preds = []
rcount = 0
for df_c in pd.read_csv('{0}/test.csv'.format(Data_path), engine='c', chunksize=batchsize,
                        sep=",", dtype=dtypes):
    rcount += batchsize
    if rcount % (10 * batchsize) == 0:
        print(rcount)
    str_array, labels, weights = df2csr(wb, df_c)
    click_ids += df_c['click_id'].tolist()
    del (df_c)
    if p != None:
        test_preds += list(p.join())
        del (X)
    gc.collect()
    X = wb.transform(str_array)
    del (str_array)
    p = ThreadWithReturnValue(target=predict_batch, args=(clf, X))
    p.start()
if p != None:  test_preds += list(p.join())

df_sub = pd.DataFrame({"click_id": click_ids, 'is_attributed': test_preds})
df_sub.to_csv("{0}/wordbatch_fm_ftrl.csv".format(Data_path), index=False)
