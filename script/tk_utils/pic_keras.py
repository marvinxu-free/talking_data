# -*- coding: utf-8 -*-
# Project: local-spark
# Author: chaoxu create this file
# Time: 2018/2/8
# Company : Maxent
# Email: chao.xu@maxent-inc.com
"""
this file used to draw keras report
"""
from matplotlib import pyplot as plt

import matplotlib
import seaborn as sns
import itertools

matplotlib.pyplot.switch_backend('agg')


def maxent_style(func):
    def maxent_style_inner(*args, **kwargs):
        print(func.__name__)
        with sns.axes_style('whitegrid', {"axes.grid": True, "axes.edgecolor": '.8'}):
            # with sns.axes_style('ticks', {'grid.color': '.8', 'grid.linestyle': u'-'}):
            # with sns.axes_style('ticks',{'grid.color': 'white'}):
            sns.set_palette('muted', n_colors=20)
            sns.set_style({'font.family': 'serif', 'font.serif': ['SimHei']})
            kwargs['palette'] = itertools.cycle(sns.color_palette('muted'))
            return func(*args, **kwargs)

    return maxent_style_inner


@maxent_style
def pic_history(history, metrics_reports, title, img_file, dpi=600, palette=None):
    if not isinstance(metrics_reports, list):
        print("""please give metric and it's label""")
        exit(-1)
    else:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)
        for key, label in metrics_reports:
            ax.plot(history[key], color=next(palette), label=label)
        ax.set_title(title)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(bbox_to_anchor=(1.01, 0.618), loc='upper left')
        fig.subplots_adjust(left=0.1, right=0.7)
        fig.savefig(filename=img_file, dpi=dpi, format='png')
        plt.show(block=False)
