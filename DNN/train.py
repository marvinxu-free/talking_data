# -*- coding: utf-8 -*-
# Project: ml_more_algorithm
# Author: chaoxu create this file
# Time: 2018/4/17
# Company : Maxent
# Email: chao.xu@maxent-inc.com

from tk_utils.model import Model
from tk_utils.config import Config
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

module_path = f'{os.path.split(os.path.realpath(__file__))[0]}/tk_utils'
Data_path = os.path.realpath("{0}/../../input".format(module_path))


def get_version_cfg(version):
    """
    :param version:
    :return:
    """
    print(f'training model for keras: {version}...')

    cfg_file = f'{module_path}/conf.ini'
    cfg = Config(cfg_file=cfg_file)
    cfg_dict = cfg.get_section(version)

    cfg_dict['Data_path'] = Data_path
    cfg_dict['train_file'] = '{0}/train.csv'.format(Data_path)
    cfg_dict['train_file_tmp'] = '{0}/train_tmp.feather'.format(Data_path)
    cfg_dict['test_file'] = '{0}/test.csv'.format(Data_path)
    cfg_dict['test_file_tmp'] = '{0}/test_tmp.feather'.format(Data_path)

    cfg_dict['predict_report_file'] = f'{Data_path}/{version}_predict.csv'
    cfg_dict['best_weights_file'] = f'{Data_path}/{version}_best.hdf5'
    return cfg_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='debug mode.')
    parser.add_argument('-s', '--section', action='store', default='keras3', help='parameters section')
    FLAGS, _ = parser.parse_known_args()
    if FLAGS.debug:
        cfg_dict = get_version_cfg('keras_debug')
    else:
        section = FLAGS.section
        cfg_dict = get_version_cfg(section)

    s = Model(cfg_dict)
    s.fit()
    print(f'train model for keras: done!!!')

    print(f'begin to predict result to {cfg_dict["predict_report_file"]}')
    s.predict()
