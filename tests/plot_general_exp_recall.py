# -*- encoding: utf-8 -*-
# Author:   Hengyu Jiang
# Time:     2022/5/24 10:42
# Email:    hengyujiang.njust.edu.cn
# Project:  PydNet
# File:     plot_general_exp_recall_plot.py
# Product:  PyCharm
# Desc:


import os
import argparse
import numpy as np
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--ID', nargs='+')
parser.add_argument('--metric', choices=['r_mae', 't_mae'], default='r_mae')
parser.add_argument('--noise_type', nargs='+')
parser.add_argument('--mode', choices=['per_noise_type', 'per_ID'], required=True)

args = parser.parse_args()

mode = args.mode
metric = args.metric
noise_type = args.noise_type
IDs = args.ID
dir_root = os.path.join('./overall_eval_results/recall_percent/', metric)

clip = {
    'r_mae': 8,
    't_mae': 0.2
}

plot_color_noise_type = {
    'Clean': 'r',
    'Partial': 'b',
    'Noise': 'm',
    'Unseen': 'g',

    'A': 'r',
    'B': 'b',
    'C': 'm',
    'D': 'g'
}


def calculate_recall_percent_curve(file:str, clip:float=8):
    data = np.loadtxt(file).clip(min=-clip, max=clip)
    size = len(data)
    recall = np.zeros(shape=(101, ))
    for i in range(0, 101):
        recall[i] = np.percentile(data, i)
    return recall


def plot_recall_curve(recall_list: list,
                      label_list: list,
                      **kwargs):
    size_list = len(recall_list)
    color_list = kwargs.get('color_list', ['r', 'g', 'b', 'm'])

    for i in range(size_list):
        size_recall = len(recall_list[i])
        y = range(0, size_recall)
        plt.plot(recall_list[i], y, label=label_list[i], color=color_list[i])

    xlabel = kwargs.get('xlabel', 'x')
    save_name = kwargs.get('save_name', None)
    title = kwargs.get('title', None)
    plt.xlabel(xlabel), plt.ylabel('recall %')
    plt.legend()
    plt.title(title)
    plt.show()
    plt.cla()


if __name__ == '__main__':


    if 'per_noise_type' == mode:
        for target_noise_type in noise_type:
            recall_percent: list = []
            color_list = []
            for id in IDs:
                recall_percent.append(calculate_recall_percent_curve(os.path.join(dir_root, f'{target_noise_type}_{id}.txt'), clip=clip[metric]))
                color_list.append(plot_color_noise_type[id])

            plot_recall_curve(recall_list=recall_percent, label_list=IDs,
                              xlabel=metric, title=f'{mode}-{target_noise_type}-{metric} recall', color_list=color_list,
                              save_name=os.path.join(dir_root, f'{mode}-{target_noise_type}-{metric}_recall.png'))
        # plot

    elif 'per_ID' == mode:
        for target_id in IDs:
            color_list = []
            recall_percent: list = []
            for nt in noise_type:
                recall_percent.append(calculate_recall_percent_curve(os.path.join(dir_root, f'{nt}_{target_id}.txt'), clip=clip[metric]))
                color_list.append(plot_color_noise_type[nt])
            plot_recall_curve(recall_list=recall_percent, label_list=noise_type,
                              xlabel=metric, title=f'{mode}-{target_id}-{metric} recall', color_list=color_list,
                              save_name=os.path.join(dir_root, f'{mode}-{target_id}-{metric}_recall.png'))

    else:
        raise NotImplementedError
