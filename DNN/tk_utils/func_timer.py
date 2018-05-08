# -*- coding: utf-8 -*-
# Project: ml_more_algorithm
# Author: chaoxu create this file
# Time: 2018/4/22
# Company : Maxent
# Email: chao.xu@maxent-inc.com
import time


def timeit(method):
    def timed(*args, **kw):
        print(f'Function: {method.__name__}')
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        use_time = te - ts
        print(f'{method.__name__} used  {use_time:.3f} ms')
        return result

    return timed
