# -*- encoding: utf-8 -*-

# Author:   Hengyu Jiang
# Time:     2022/4/19 15:39
# Email:    hengyujiang.njust.edu.cn
# Project:  PydNet
# File:     plot_all_loss_png_of_different_losses.py
# Product:  PyCharm
# Desc:


import os
import cv2 as cv
from matplotlib import pyplot as plt

num_of_plots = 4
target_dir = "../logs_pydnet/fe_none/"

if __name__ == '__main__':
    for idx, dir_each_exp in enumerate(os.listdir(target_dir)[:num_of_plots]):

        path_plot_each_exp = os.path.join(os.path.join(target_dir, dir_each_exp),"loss.png")
        print(path_plot_each_exp)
        loss_image = cv.imread(path_plot_each_exp)
        plt.subplot(2, 2, idx + 1)
        plt.imshow(cv.cvtColor(loss_image, cv.COLOR_BGR2RGB))

    plt.savefig('asdf.png')
    plt.show()
