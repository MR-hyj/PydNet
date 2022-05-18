# -*- encoding: utf-8 -*-

# Author:   Hengyu Jiang
# Time:     2022/5/13 19:30
# Email:    hengyujiang.njust.edu.cn
# Project:  PydNet
# File:     complete_tensor.py
# Product:  PyCharm
# Desc:

import numpy as np
from copy import deepcopy
import torch
import time


row, col = 4, 2048
a = torch.from_numpy(np.array(range(row*col)).reshape(row, col))
mask = torch.rand(row, col) > 0.5

if __name__ == '__main__':
    # print(a)
    print(mask)

    min_num = torch.min(torch.sum(mask, axis=1))
    begin = time.time()
    for each_row in mask:
        remain = torch.sum(each_row) - min_num
        cnt = 0

        for idx in range(each_row.shape[0]):
            if each_row[idx] == True:
                each_row[idx] = False
                cnt += 1
            if cnt >= remain:
                break

    print(mask)
    print(a[mask])
    b = a[mask].reshape(row, min_num)
    print(b.shape)
    print(b)
    print('time {}'.format(time.time()-begin))
