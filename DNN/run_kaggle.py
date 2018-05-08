import sys

module = 'module3'
sys.path.insert(0, f'../input/{module}/tk_utils/')

from tk_utils.model import Model
from tk_utils.config import Config
import os
import argparse

print(os.listdir(f'../input/{module}/tk_utils/tk_utils'))

module_path = f'../input/{module}/tk_utils/tk_utils'
Data_path = os.path.realpath("{0}/../../../talkingdata-adtracking-fraud-detection/".format(module_path))


def get_version_cfg(version):
    """
    :param version:
    :return:
    """
    print(f'training model for keras: {version}...')

    cfg_file = f'{module_path}/conf.ini'
    print(cfg_file)
    cfg = Config(cfg_file=cfg_file)
    cfg_dict = cfg.get_section(version)

    cfg_dict['Data_path'] = Data_path
    cfg_dict['train_file'] = '{0}/train.csv'.format(Data_path)
    cfg_dict['train_file_tmp'] = '{0}/train_tmp.csv'.format(Data_path)
    cfg_dict['test_file'] = '{0}/test.csv'.format(Data_path)
    cfg_dict['test_file_tmp'] = '{0}/test_tmp.csv'.format(Data_path)

    cfg_dict['predict_report_file'] = f'{Data_path}/{version}_predict.csv'
    cfg_dict['best_weights_file'] = f'{Data_path}/{version}_best.hdf5'
    return cfg_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=False, help='debug mode.')
    FLAGS, _ = parser.parse_known_args()
    if FLAGS.debug:
        cfg_dict = get_version_cfg('keras_debug')
    else:
        cfg_dict = get_version_cfg('keras1')

    s = Model(cfg_dict)
    s.fit()
    print(f'train model for keras: done!!!')

    del s
    print('begin to predict, make sure use best model...')
    ps = Model(cfg_dict)
    s.predict()
    print(f"predict file is ...{cfg_dict['predict_report_file']}")