# -*- encoding: utf-8 -*-

# Author:   Hengyu Jiang
# Time:     2022/5/20 10:20
# Email:    hengyujiang.njust.edu.cn
# Project:  PydNet
# File:     load_perm_metric.py
# Product:  PyCharm
# Desc:

import os
import argparse
import pickle
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--perm_metrix_file_dir', type=str, metavar='PATH', required=True)
parser.add_argument('--sample_idx', type=int, required=True)
parser.add_argument('--show_fig', action='store_true', default=False)

args = parser.parse_args()


if __name__ == '__main__':
    with open(os.path.join(args.perm_metrix_file_dir, 'perm_matrices.pickle'), 'rb') as f:
        # (num_sample, 1, num_points, num_points)
        # num_sample 是测试集的大小
        # 每个迭代生成一个num_points x num_points的方阵, [i][j]表示点i与点j匹配的概率的大小
        perm_metrix_all = pickle.load(f)
    print('loaded perm_metrix_all, shape: ({}, {}, {}, {})'.format(len(perm_metrix_all), len(perm_metrix_all[0]),
                                                                   perm_metrix_all[0][0].shape[0], perm_metrix_all[0][0].shape[1]))
    perm_metrix = perm_metrix_all[args.sample_idx][0]
    print('Selected perm metrix: sample idx={}'.format(args.sample_idx))
    save_name_txt = os.path.join(args.perm_metrix_file_dir, 'perm_matrix_sample-{}.txt'.format(args.sample_idx))
    if os.path.exists(save_name_txt):
        os.remove(save_name_txt)
    np.savetxt(save_name_txt, perm_metrix)
    print('Saving perm metrix to {}'.format(save_name_txt))
    plt.imshow(perm_metrix)
    plt.colorbar()
    save_name_png = os.path.join(args.perm_metrix_file_dir, 'perm_matrix_sample-{}.png'.format(args.sample_idx))
    if os.path.exists(save_name_png):
        os.remove(save_name_png)
    plt.savefig(save_name_png)
    print('Saving perm png to {}'.format(save_name_png))
    if args.show_fig:
        plt.show()
