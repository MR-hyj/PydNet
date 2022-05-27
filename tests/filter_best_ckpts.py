# -*- encoding: utf-8 -*-

# Author:   Hengyu Jiang
# Time:     2022/5/27 10:16
# Email:    hengyujiang.njust.edu.cn
# Project:  PydNet
# File:     filter_best_ckpts.py
# Product:  PyCharm
# Desc:


import os
import sys
import shutil


CKPT_ROOT_DIR = "..\\logs_pydnet\\"
ckpt_dirs = [x for x in os.listdir(CKPT_ROOT_DIR) if os.path.isdir(os.path.join(CKPT_ROOT_DIR, x))]


def get_best_ckpt(best_ckpt_txt_dir: str):
    best_ckpt_txt = os.path.join(best_ckpt_txt_dir, "best_models.txt")
    if not os.path.exists(best_ckpt_txt):
        raise FileNotFoundError

    best_models: list = []
    with open(best_ckpt_txt) as f:
        while True:
            line = f.readline()[:-1]
            if line:
                best_models.append(line)
            else:
                break
    print(f'{len(best_models)} best ckpt found')
    return best_models


def delete_non_best_ckpt(dir_all_ckpt: str,
                         best_ckpts: list):
    for each_ckpt in os.listdir(dir_all_ckpt):
        if each_ckpt not in best_ckpts:
            print(f'Deleting {os.path.join(dir_all_ckpt, each_ckpt)}')
            os.remove(os.path.join(dir_all_ckpt, each_ckpt))
        else:
            print(f'Skipping {os.path.join(dir_all_ckpt, each_ckpt)}')


for each_ckpt_dir in ckpt_dirs:
    print(f'Select ckpt {each_ckpt_dir}')
    each_ckpt_dir = os.path.join(CKPT_ROOT_DIR, each_ckpt_dir)
    best_ckpts = get_best_ckpt(each_ckpt_dir)
    delete_non_best_ckpt(os.path.join(each_ckpt_dir, 'ckpt'), best_ckpts)
    print()
