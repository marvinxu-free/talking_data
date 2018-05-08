# -*- coding: utf-8 -*-
# Project: ml_more_algorithm
# Author: chaoxu create this file
# Time: 2018/4/27
# Company : Maxent
# Email: chao.xu@maxent-inc.com
"""
本文件用于测试复杂配置文件的读取
"""
import configparser
import json
import ast

cfg_file = "./conf.ini"

cfg = configparser.ConfigParser(allow_no_value=True)
cfg.read(cfg_file)

dtypes = cfg.get("keras1","dtypes")
print(f'dtypes is {dtypes}')

tk_dtypes = json.loads(dtypes)
print(tk_dtypes)

usecols = cfg.get("keras1","use_cols")
use_cols = ast.literal_eval(usecols)
print(use_cols)

section = cfg.options('keras1')
