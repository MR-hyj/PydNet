# -*- encoding: utf-8 -*-

# Author:   Hengyu Jiang
# Time:     2022/1/6 14:41
# Email:    hengyujiang.njust.edu.cn
# Project:  PydNet
# File:     feature_test.py
# Product:  PyCharm
# Desc:


import json
import os
import sys
import cv2 as cv
import torch
import pickle
import shutil
import logging
import datetime
import argparse
import numpy as np
import coloredlogs
from PIL import Image
from PIL import ImageOps
from common.mytorch import to_numpy
from common.io import generate_rectangular_obj, points_normals_from_ply, points_from_obj
from arguments import pydnet_eval_arguments
from modules.pointnet_util import query_ball_point
from common.mytorch import to_numpy

np.set_printoptions(suppress=True)
_logger = logging.getLogger()

__EPS = 1e-5

def PCA(data: np.ndarray,
        k: int) -> np.ndarray:
    """
    对data进行PCA降维
    :param data:
    :param k:
    """

    data = data - data.mean(axis=0)  # 标准化
    cov = np.dot(data.T, data)  # 协方差矩阵

    val, vec = np.linalg.eig(cov)  # 特征分解
    index = np.argsort(val)[-k:]
    W = vec[:, index]
    return np.dot(data, W)


def augment(rgb: np.ndarray):
    # aug = 1.85*rgb+35
    # return np.clip(aug, 0, 255)

    img = Image.fromarray(np.uint8([rgb]), mode='RGB')
    img = np.asarray(ImageOps.equalize(img))[0]
    return img


def feat_2_rgb(feat: np.ndarray):
    raw = PCA(feat, 3)
    x_min = raw.min()
    x_max = raw.max()
    rgb = (raw - x_min) / (x_max - x_min) * 255
    auged = augment(rgb).astype(np.uint8)
    return auged


def prepare_logger(opt: argparse.Namespace, log_path: str = None):
    """Creates logging directory, and installs colorlogs

    Args:
        opt: Program arguments, should include --dev and --logdir flag.
             See get_parent_parser()
        log_path: Logging path (optional). This serves to overwrite the settings in
                 argparse namespace

    Returns:
        logger (logging.Logger)
        log_path (str): Logging directory
    """

    if log_path is None:
        if opt.dev:
            log_path = '../logdev'
            shutil.rmtree(log_path, ignore_errors=True)
        else:
            datetime_str = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
            if opt.name is not None:
                log_path = os.path.join(opt.logdir, datetime_str + '_' + opt.name)
            else:
                log_path = os.path.join(opt.logdir, datetime_str)

    os.makedirs(log_path, exist_ok=True)
    logger = logging.getLogger()
    coloredlogs.install(level='INFO', logger=logger)
    file_handler = logging.FileHandler('{}/log.txt'.format(log_path))
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    print_info(opt, log_path)
    logger.info('Output and logs_pydnet will be saved to {}'.format(log_path))
    return _logger, log_path


def print_info(opt, log_path):
    """ Logs source code configuration
    """
    _logger.info('Command: {}'.format(' '.join(sys.argv)))

    # Arguments
    arg_str = ['{}: {}'.format(key, value) for key, value in vars(opt).items()]
    arg_str = ', '.join(arg_str)
    _logger.info('Arguments: {}'.format(arg_str))


def dis_hsv1_hsv2(hsv1: np.ndarray,
                  hsv2: np.ndarray,
                  r: np.float = 1,
                  h: np.float = 1,
                  l: np.float = np.sqrt(2)
                  ) -> np.float:
    """
    两个hsv空间内点的距离
    圆锥底半径r=1, 圆锥高h=1, 圆锥母线l=sqrt(2)
    Args:
        hsv1:
        hsv2:
        r:
        h:
        l:

    Returns: np.float

    """
    hsv1_h, hsv1_s, hsv1_v = hsv1
    hsv2_h, hsv2_s, hsv2_v = hsv2

    x1 = r * hsv1_v * hsv1_v * np.cos(hsv1_h / 180 * np.pi)
    y1 = r * hsv1_v * hsv1_v * np.sin(hsv1_h / 180 * np.pi)
    z1 = h * (1 - hsv1_v)

    x2 = r * hsv2_v * hsv2_v * np.cos(hsv2_h / 180 * np.pi)
    y2 = r * hsv2_v * hsv2_v * np.sin(hsv2_h / 180 * np.pi)
    z2 = h * (1 - hsv2_v)

    dx, dy, dz = x1-x2, y1-y2, z1-z2

    return np.sqrt(dx**2+dy**2+dz**2)


def dis_rgbs1_rgbs2(rgbs1: np.ndarray,
                    rgbs2: np.ndarray):


    return np.linalg.norm(rgbs1-rgbs2, axis=1)


def dis_hsvs1_hsvs2(hsvs1: np.ndarray,
                    hsvs2: np.ndarray,
                    r: np.float = 1,
                    h: np.float = 1,
                    l: np.float = np.sqrt(2)
                    ) -> np.float:
    """
    两个hsv空间内点的距离
    圆锥底半径r=1, 圆锥高h=1, 圆锥母线l=sqrt(2)
    Args:
        hsvs1:
        hsvs2:
        r:
        h:
        l:

    Returns:    np.ndarray(dtype=np.float)

    """
    hsvs1_h, hsvs1_s, hsvs1_v = hsvs1[:, 0], hsvs1[:, 1], hsvs1[:, 2]
    hsvs2_h, hsvs2_s, hsvs2_v = hsvs2[:, 0], hsvs2[:, 1], hsvs2[:, 2]

    x1 = r * hsvs1_v * hsvs1_v * np.cos(hsvs1_h / 180 * np.pi)
    y1 = r * hsvs1_v * hsvs1_v * np.sin(hsvs1_h / 180 * np.pi)
    z1 = h * (1 - hsvs1_v)

    x2 = r * hsvs2_v * hsvs2_v * np.cos(hsvs2_h / 180 * np.pi)
    y2 = r * hsvs2_v * hsvs2_v * np.sin(hsvs2_h / 180 * np.pi)
    z2 = h * (1 - hsvs2_v)

    dx, dy, dz = x1 - x2, y1 - y2, z1 - z2

    return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


def rgbs2hsvs(rgbs: np.ndarray):
    rgb_ = rgbs / 255.0
    r, g, b = rgb_[:, 0], rgb_[:, 1], rgb_[:, 2]

    c_max = rgb_.max(axis=1)
    c_min = rgb_.min(axis=1)

    delta = c_max - c_min
    h = np.zeros(shape=delta.shape)
    s = np.zeros(shape=delta.shape)

    h[c_max == c_min] = 0

    cond1 = (c_max != 0).astype(np.bool)

    cond2 = (c_max == r).astype(np.bool)
    cond = cond1 & cond2
    h[cond] = (60 * ((g[cond] - b[cond]) / (delta[cond])+__EPS) + 360) % 360
    h[cond] = np.nan_to_num(h[cond])

    cond2 = (c_max == g).astype(np.bool)
    cond = cond1 & cond2
    h[cond] = (60 * ((b[cond] - r[cond]) / (delta[cond])+__EPS) + 120) % 360
    h[cond] = np.nan_to_num(h[cond])

    cond2 = (c_max == b).astype(np.bool)
    cond = cond1 & cond2
    h[cond] = (60 * ((r[cond] - g[cond]) / (delta[cond])+__EPS) + 240) % 360
    h[cond] = np.nan_to_num(h[cond])

    s[cond1] = delta[cond1] / c_max[cond1]
    v = c_max

    return np.concatenate([[h], [s], [v]]).T


def calculate_k_neighbor_distance(radius: float, n_samples: int, points: torch.Tensor, hsv: np.ndarray) -> dict:
    indices = to_numpy(query_ball_point(radius, n_samples, points, points))  # (B, N, num_samples)
    indices = indices[0]  # (N, neighbors)
    distance = np.array([])
    for i in range(indices.shape[0]):
        hsv_centroid = hsv[i]  # (3, )
        neighbors = indices[i]
        hsv_neighbors = hsv[neighbors]  # (neighbors, 3)

        distance_tmp = dis_hsvs1_hsvs2(np.array([hsv_centroid]), hsv_neighbors)
        distance = np.concatenate([distance, distance_tmp])

    loss = {
        'n_samples': n_samples,
        'mean': np.mean(distance),
        'var': np.var(distance)
    }

    return loss


if __name__ == '__main__':
    # rgbs1 = np.random.random(size=(6, 3)) * 255
    # rgbs1 = rgbs1.astype(np.uint8)
    # hsvs1 = rgbs2hsvs(rgbs1)
    #
    # rgbs2 = np.random.random(size=(6, 3)) * 255
    # rgbs2 = rgbs2.astype(np.uint8)
    # hsvs2 = rgbs2hsvs(rgbs2)
    #
    # for idx in range(6):
    #     print(dis_hsv1_hsv2(hsvs1[idx], hsvs2[idx]))
    # print()
    # print(dis_hsvs1_hsvs2(hsvs1, hsvs2))


    parser = pydnet_eval_arguments()
    _args = parser.parse_args()
    _logger, _log_path = prepare_logger(_args)

    feature_dir_root = '../feat_dumped_pydnet/batch_0/'

    # 计算k近邻下, pydnet和rmpnet特征的特征邻域距离

    for sample_idx in range(_args.val_batch_size):

        feature_dir_sample = os.path.join(feature_dir_root, 'sample_{}'.format(sample_idx))
        _logger.info('searching directory {}'.format(feature_dir_sample))

        with open(os.path.join(feature_dir_sample, 'feat_ref.txt'), 'rb') as f:
             feat_ref = pickle.load(f)
        with open(os.path.join(feature_dir_sample, 'feat_src.txt'), 'rb') as f:
            feat_src = pickle.load(f)

        # rgb = (N, 3)
        rgb_ref = feat_2_rgb(feat_ref)
        rgb_src = feat_2_rgb(feat_src)
        hsv_src = rgbs2hsvs(rgb_src)
        hsv_ref = rgbs2hsvs(rgb_ref)
        _logger.info('rgb src shape: {}, rgb ref shape: {}'.format(rgb_src.shape, rgb_ref.shape))

        transform = np.loadtxt(os.path.join(feature_dir_sample, 'pred_transform.txt'))
        transform = np.concatenate((transform, np.array([[0, 0, 0, 1]], dtype=np.float)), axis=0)

        # points_src, normals_src = points_normals_from_ply(os.path.join(feature_dir_sample, 'src_cloud.ply'))
        # points_ref, normals_ref = points_normals_from_ply(os.path.join(feature_dir_sample, 'ref_cloud.ply'))
        points_src = points_from_obj(os.path.join(feature_dir_sample, 'src_cloud.obj'))
        points_ref = points_from_obj(os.path.join(feature_dir_sample, 'ref_cloud.obj'))
        points_pred = np.concatenate((points_src, np.ones(shape=(points_src.shape[0], 1), dtype=np.float)), axis=1)@transform.T
        points_pred = points_pred[: , :3]

        _logger.info('src cloud shape: {}, ref cloud shape: {}'.format(points_src.shape, points_ref.shape))
        points_src_tensor = torch.from_numpy(np.array([points_src])).cuda()
        points_ref_tensor = torch.from_numpy(np.array([points_ref])).cuda()


        generate_rectangular_obj(os.path.join(feature_dir_sample, 'src_cloud_colored.obj'),
                                 points=points_src, points_colors=rgb_src)
        generate_rectangular_obj(os.path.join(feature_dir_sample, 'ref_cloud_colored.obj'),
                                 points=points_ref, points_colors=rgb_ref)

        for n_samples in [4, 8, 16, 32, 64]:
            # indices = (1, N, num_samples)
            # indices[0][i] = [x0, x1, x2, ...,] 表示邻居点的下标
            loss_src = calculate_k_neighbor_distance(radius=_args.radius, n_samples=n_samples,
                                                     points=points_src_tensor, hsv=hsv_src)
            _logger.info('dumped src loss {}'.format(loss_src))
            with open(os.path.join(feature_dir_sample, 'loss_ref_{}.json'.format(n_samples)), 'w') as f:
                json.dump(loss_src, f)

            loss_ref = calculate_k_neighbor_distance(radius=_args.radius, n_samples=n_samples,
                                                     points=points_ref_tensor, hsv=hsv_ref)
            _logger.info('dumped ref loss {}\n'.format(loss_ref))
            with open(os.path.join(feature_dir_sample, 'loss_ref_{}.json'.format(n_samples)), 'w') as f:
                json.dump(loss_ref, f)


    # # 计算PCL中的特征
    # # rgb = (N, 3)
    # # rgb_ref = feat_2_rgb(feat_ref)
    # feat_names = ['SHOT', 'FPFH']
    # pcl_feat_dir = '../feat_dumped_pcl/'
    # for n_samples in [4, 8, 16, 32, 64]:
    #     for feat_name in feat_names:
    #         feat_src = np.loadtxt(os.path.join(pcl_feat_dir, '{}/{}.txt'.format(feat_name, feat_name)))
    #         rgb_src = feat_2_rgb(feat_src)
    #         hsv_src = rgbs2hsvs(rgb_src)
    #         _logger.info('rgb shape: {}'.format(rgb_src.shape))
    #
    #         points_src = points_from_obj('../feat_dumped_pydnet/batch_0/sample_6/src_cloud.obj')
    #         _logger.info('shape of src cloud: {}'.format(points_src.shape))
    #         points_src_tensor = torch.from_numpy(np.array([points_src])).cuda()
    #
    #         if n_samples == 4:
    #             generate_rectangular_obj(os.path.join(pcl_feat_dir, '{}/src_cloud_colored.obj'.format(feat_name)),
    #                                      points=points_src, points_colors=rgb_src)
    #         # indices = (1, N, num_samples)
    #         # indices[0][i] = [x0, x1, x2, ...,] 表示邻居点的下标
    #
    #         indices = to_numpy(
    #             query_ball_point(_args.radius, n_samples, points_src_tensor, points_src_tensor))  # (B, N, num_samples)
    #         indices = indices[0]  # (N, neighbors)
    #         distance = np.array([])
    #         for i in range(indices.shape[0]):
    #             hsv_centroid = hsv_src[i]  # (3, )
    #             neighbors = indices[i]
    #             hsv_neighbors = hsv_src[neighbors]  # (neighbors, 3)
    #
    #             distance_tmp = dis_hsvs1_hsvs2(np.array([hsv_centroid]), hsv_neighbors)
    #             distance = np.concatenate([distance, distance_tmp])
    #
    #         loss = {
    #             'n_samples': n_samples,
    #             'mean': np.mean(distance),
    #             'var': np.var(distance)
    #         }
    #         _logger.info('dumped loss {}\n'.format(loss))
    #         with open(os.path.join(pcl_feat_dir, '{}/loss_{}.json'.format(feat_name, n_samples)) , 'w') as f:
    #             json.dump(loss, f)
