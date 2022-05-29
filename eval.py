# -*- encoding: utf-8 -*-

# Author:   Hengyu Jiang
# Time:     2022/1/6 13:28
# Email:    hengyujiang.njust.edu.cn
# Project:  PydNet
# File:     eval.py
# Product:  PyCharm
# Desc:
# -*- encoding: utf-8 -*-

# Author:   Hengyu Jiang
# Time:     2021/12/27 14:07
# Email:    hengyujiang.njust.edu.cn
# Project:  RPMNet
# File:     eval.py
# Product:  PyCharm
# Desc:


"""Evaluate RPMNet. Also contains functionality to compute evaluation metrics given transforms

Example Usages:
    1. Evaluate RPMNet
        python eval.py --noise_type crop --resume [path-to-model.pth]

    2. Evaluate precomputed transforms (.npy file containing np.array of size (B, 3, 4) or (B, n_iter, 3, 4))
        python eval.py --noise_type crop --transform_file [path-to-transforms.npy]
"""

from collections import defaultdict
import json
import os
import pickle
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import open3d  # Need to import before torch
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import torch
from torchsummary import summary
from arguments import pydnet_eval_arguments, pydnet_eval_from_outer_arugments
from common.misc import prepare_logger
from common.mytorch import dict_all_to_device, CheckPointManager, to_numpy
from common.math import se3
from common.math_torch import se3
from common.math.so3 import dcm2euler
from data_loader.datasets import get_test_datasets
import modules.pydnet
from common.io import generate_rectangular_obj
from vispy import io
from copy import deepcopy
from matplotlib import pyplot as plt

np.set_printoptions(suppress=True)

"""
{
	"r_mse": 			欧拉角的二范数误差,  mean absolute error
	"r_mae": 			欧拉角的一范数误差,  mean absolute error
	"t_mse": 			平移的二范数误差,   mean square error
	"t_mae": 			平移的一范数误差,   mean square error
	"err_r_deg_mean": 	isotropic rotation 的平均值     mean isotropic error
	"err_r_deg_mse": 	isotropic rotation 平方和的平方根 
	"err_t_mean": 		isotropic translation 的平均值  mean isotropic error
	"err_t_mse": 		isotropic translation 平方和的平方根
	"chamfer_dist": 	改进的chamfer distance
}
"""


def distance_point_to_points(point: np.ndarray,
                             points: np.ndarray) -> np.ndarray:
    """
    calculate the distance of a point and a set of points
    @param point:     a point, marked as A, with shape (3, )
    @param points:    a set of points, marked as B, with shape (n, 3, )
    @return  the distance between A and each point in B with shape (n, )
    """

    # ignore the normals if any
    assert 1 == len(point.shape), _logger.error(
        f'Wrong shape of input point, expected 1 dimensional, but {len(point.shape)} detected.')
    assert 2 == len(points.shape), _logger.error(
        f'Wrong shape of input points, expected 2 dimensional, but {len(points.shape)} detected.')
    point, points = point.astype(np.float)[:3], points.astype(np.float)[:, :3]
    return np.linalg.norm(points - point, axis=-1)


def distance_cloud_to_cloud(cloud_1: np.ndarray,
                            cloud_2: np.ndarray,
                            normalize: bool = True) -> np.ndarray:
    """
    calculate the distance of each point in points_1 and each point in points_2
    @param cloud_1: a set of points, marked as A, with shape (na, 3, )
    @param cloud_2: a set of points, marked as B, with shape (nb, 3, )
    @return: the distance of each point in A and each point in B, with shape (na, nb)
    """
    distance_a_to_b = np.array([])
    for point in cloud_1:
        distance_each = distance_point_to_points(point, cloud_2)
        distance_a_to_b = np.concatenate([distance_a_to_b, distance_each])
    if not normalize:
        return distance_a_to_b.reshape(cloud_1.shape[0], cloud_2.shape[0])
    else:
        distance_a_to_b = distance_a_to_b.reshape(cloud_1.shape[0], cloud_2.shape[0])
        max_ = distance_a_to_b.max(axis=1)
        distance = (distance_a_to_b.T / max_).T
        return distance


def compute_metrics(data: Dict, pred_transforms, clip_val=0.1) -> Dict:
    """Compute metrics required in the paper
    """

    def square_distance(src, dst):
        return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)

    with torch.no_grad():
        pred_transforms = pred_transforms
        gt_transforms = data['transform_gt']
        points_src = data['points_src'][..., :3]
        points_ref = data['points_ref'][..., :3]
        points_raw = data['points_raw'][..., :3]

        # Euler angles, Individual translation errors (Deep Closest Point convention)
        # TODO Change rotation to torch operations
        r_gt_euler_deg = dcm2euler(gt_transforms[:, :3, :3].detach().cpu().numpy(), seq='xyz')
        r_pred_euler_deg = dcm2euler(pred_transforms[:, :3, :3].detach().cpu().numpy(), seq='xyz')
        t_gt = gt_transforms[:, :3, 3]
        t_pred = pred_transforms[:, :3, 3]
        r_mse = np.mean((r_gt_euler_deg - r_pred_euler_deg) ** 2, axis=1)
        r_mae = np.mean(np.abs(r_gt_euler_deg - r_pred_euler_deg), axis=1)
        t_mse = torch.mean((t_gt - t_pred) ** 2, dim=1)
        t_mae = torch.mean(torch.abs(t_gt - t_pred), dim=1)

        # Rotation, translation errors (isotropic, i.e. doesn't depend on error
        # direction, which is more representative of the actual error)
        concatenated = se3.concatenate(se3.inverse(gt_transforms), pred_transforms)
        rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
        residual_rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
        residual_transmag = concatenated[:, :, 3].norm(dim=-1)

        # Modified Chamfer distance
        clip_val = torch.Tensor([clip_val]).cuda()
        src_transformed = se3.transform(pred_transforms, points_src)
        ref_clean = points_raw
        src_clean = se3.transform(se3.concatenate(pred_transforms, se3.inverse(gt_transforms)), points_raw)

        dist_src = torch.min(square_distance(src_transformed, ref_clean), dim=-1)[0]
        dist_ref = torch.min(square_distance(points_ref, src_clean), dim=-1)[0]
        chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

        dist_src = torch.min(torch.min(square_distance(src_transformed, ref_clean), dim=-1)[0], clip_val)
        dist_ref = torch.min(torch.min(square_distance(points_ref, src_clean), dim=-1)[0], clip_val)
        clip_chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

        metrics = {
            'r_mse': r_mse,
            'r_mae': r_mae,
            't_mse': to_numpy(t_mse),
            't_mae': to_numpy(t_mae),
            'err_r_deg': to_numpy(residual_rotdeg),
            'err_t': to_numpy(residual_transmag),
            'chamfer_dist': to_numpy(chamfer_dist),
            'clip_chamfer_dist': to_numpy(clip_chamfer_dist)
        }

    return metrics


def calculate_recall(metrics: dict,
                     rot_lim: float = 1,
                     trans_lim: float = 0.1):
    """the proportion of samples with r_mae<rot_lim and t_mae<trans_lim
    Args:
        metrics:
        rot_lim:
        trans_lim:

    Returns:

    """
    num_samples = len(metrics['r_mae'])
    r_mae_le_than = np.where(np.array(metrics['r_mae']) < rot_lim)[0]
    t_mae_le_than = np.where(np.array(metrics['t_mae']) < trans_lim)[0]
    r_t_mae_le_than = np.intersect1d(r_mae_le_than, t_mae_le_than)
    num_goods = len(r_t_mae_le_than)
    recall = num_goods / num_samples

    return recall


def calculate_recall_percentile(metrics: dict,
                                percent: float = 0.25):
    """calculate the top percent r_mae and t_mae in metrics
    Args:
        metrics: dict
        percent: float

    Returns:

    """
    r_mae_percent = np.percentile(metrics['r_mae'], q=100 * percent)
    t_mae_percent = np.percentile(metrics['t_mae'], q=100 * percent)
    return r_mae_percent, t_mae_percent


def calculate_mean_metric_per_category(metrics: dict, key: str):
    """Calculate the specified metric of each category in test samples

    Args:
        metrics: dict ['r_mse', 'r_mae', 't_mse', 't_mae', 'err_r_deg', 'err_t', 'chamfer_dist', 'clip_chamfer_dist', 'label']
        key:     str

    Returns:
        np.ndarray
    """

    key_per_category = {}

    for idx in range(len(metrics['r_mae'])):
        if key.endswith('rmse'):
            sample_key_value = np.sqrt(metrics[key[0] + '_mse'][idx])
        else:
            sample_key_value = metrics[key][idx]

        sample_label = metrics['label'][idx]
        if sample_label not in key_per_category.keys():
            key_per_category[sample_label] = np.array([sample_key_value])
        else:
            key_per_category[sample_label] = np.concatenate([key_per_category[sample_label], [sample_key_value]])
    # _logger.info('key_per_category.keys: {}'.format(key_per_category.keys()))

    for each_category in key_per_category.keys():
        key_per_category[each_category] = np.mean(key_per_category[each_category])

    return key_per_category


def summarize_metrics(metrics: dict, eval_mode=False):
    """Summaries computed metrices by taking mean over all data instances

    Args:
        metrics: metrics[key]: (num_samples, ) ['r_mse', 'r_mae', 't_mse', 't_mae', 'err_r_deg', 'err_t', 'chamfer_dist', 'clip_chamfer_dist', 'label']

    Returns:

    """
    summarized = {}
    for k in metrics:
        if k.endswith('mse'):
            summarized[k[:-3] + 'rmse'] = np.sqrt(np.mean(metrics[k]))
        elif k.startswith('err'):
            summarized[k + '_mean'] = np.mean(metrics[k])
            summarized[k + '_rmse'] = np.sqrt(np.mean(metrics[k] ** 2))
        else:
            summarized[k] = np.mean(metrics[k])
    limits = [[1, 0.1], [0.5, 0.05],
              [0.1, 0.01], [0.05, 0.005],
              [0.01, 0.001]]
    for rot_limit, trans_limit in limits:
        summarized[f'Recall_{rot_limit}_{trans_limit}'] = calculate_recall(metrics, rot_lim=rot_limit,
                                                                           trans_lim=trans_limit)

    for recall_percent in [0.75, 0.5, 0.25, 0.10]:
        summarized[f'Recall_r_mae_{recall_percent}'], summarized[f'Recall_t_mae_{recall_percent}'] = \
            calculate_recall_percentile(metrics, recall_percent)

    mean_metric_each_category = {}
    if eval_mode:
        for key in ['r_rmse', 't_rmse', 'r_mae', 't_mae']:
            mean_metric_each_category[key] = calculate_mean_metric_per_category(metrics, key)
    return summarized, mean_metric_each_category


def print_metrics(logger, summary_metrics: Dict, losses_by_iteration: List = None,
                  title: str = 'Metrics'):
    """Prints out formated metrics to logger"""

    logger.info(title + ':')
    logger.info('=' * (len(title) + 1))

    if losses_by_iteration is not None:
        losses_all_str = ' | '.join(['{:.8f}'.format(c) for c in losses_by_iteration])
        logger.info('Losses by iteration: {}'.format(losses_all_str))

    logger.info('DeepCP metrics:{:.8f}(rot-rmse) | {:.8f}(rot-mae) | {:.8g}(trans-rmse) | {:.8g}(trans-mae)'.format(
        summary_metrics['r_rmse'], summary_metrics['r_mae'],
        summary_metrics['t_rmse'], summary_metrics['t_mae'],
    ))
    logger.info('Rotation error(ERM) {:.8f}(deg, mean) | {:.8f}(deg, rmse)'.format(
        summary_metrics['err_r_deg_mean'], summary_metrics['err_r_deg_rmse']))
    logger.info('Translation error(ETM) {:.8g}(mean) | {:.8g}(rmse)'.format(
        summary_metrics['err_t_mean'], summary_metrics['err_t_rmse']))
    logger.info('Chamfer error: {:.10f}(mean-sq)'.format(
        summary_metrics['chamfer_dist']
    ))
    logger.info('Clip Chamfer error: {:.10f}(mean-sq)'.format(
        summary_metrics['clip_chamfer_dist']
    ))

    for key in summary_metrics:
        if key.startswith('Recall_r_mae') or key.startswith('Recall_t_mae'):
            logger.info('{}: {:.6f}'.format(key, summary_metrics[key]))
        elif key.startswith('Recall'):
            logger.info('{}: {:.6f}%'.format(key, summary_metrics[key] * 100))


def save_similarity_plot(cloud_src: np.ndarray,
                         cloud_ref: np.ndarray,
                         cloud_pred: np.ndarray,
                         gt_transform: torch.Tensor,
                         save_dir: str,
                         select_begin: int = 0,
                         select_num: int = 10):
    src_transformed = to_numpy(se3.transform(gt_transform, torch.from_numpy(cloud_src)))
    similarity_src_ref = 1 - distance_cloud_to_cloud(cloud_src, cloud_ref, normalize=True)
    similarity_pred_ref = 1 - distance_cloud_to_cloud(cloud_pred, cloud_ref, normalize=True)
    similarity_ground_ref = 1 - distance_cloud_to_cloud(src_transformed, cloud_ref, normalize=True)

    plt.imshow(
        similarity_ground_ref[select_begin: select_begin + select_num, select_begin: select_begin + select_num])
    plt.title('ground transform to ref')
    plt.xticks([]), plt.yticks([]), plt.colorbar()
    plt.savefig(os.path.join(save_dir, 'sim_ground_ref.png'))
    plt.clf(), plt.cla()

    plt.imshow(
        similarity_src_ref[select_begin: select_begin + select_num, select_begin: select_begin + select_num])
    plt.title('src to ref')
    plt.xticks([]), plt.yticks([]), plt.colorbar()
    plt.savefig(os.path.join(save_dir, 'sim_src_ref.png'))
    plt.clf(), plt.cla()

    plt.imshow(
        similarity_pred_ref[select_begin: select_begin + select_num, select_begin: select_begin + select_num])
    plt.title('pred to ref')
    plt.xticks([]), plt.yticks([]), plt.colorbar()
    plt.savefig(os.path.join(save_dir, 'sim_pred_ref.png'))
    plt.clf(), plt.cla()

    np.savetxt(os.path.join(save_dir, 'sim_ground_ref.txt'), similarity_ground_ref, fmt='%.08f')
    np.savetxt(os.path.join(save_dir, 'sim_src_ref.txt'), similarity_src_ref, fmt='%.08f')
    np.savetxt(os.path.join(save_dir, 'sim_pred_ref.txt'), similarity_pred_ref, fmt='%.08f')


def inference(data_loader, model: torch.nn.Module):
    """Runs inference over entire dataset

    Args:
        data_loader (torch.utils.data.DataLoader): Dataset loader
        model (model.nn.Module): Network model to evaluate

    Returns:
        pred_transforms_all: predicted transforms (B, n_iter, 3, 4) where B is total number of instances
        endpoints_out (Dict): Network endpoints
    """

    _logger.info('Starting inference...')
    model.eval()

    pred_transforms_all = []
    all_betas, all_alphas = [], []
    total_time = 0.0
    endpoints_out = defaultdict(list)
    total_rotation = []
    cnt = 0
    num_samples = 0
    with torch.no_grad():

        if _args.feat_dumpdir is not None:
            dump_dir_split = os.path.join('./outer_methods_results/{}/'.format(_args.noise_type))
            dump_dir_together = _args.feat_dumpdir
            dir_pred_cloud = os.path.join(dump_dir_split, 'pred_cloud/pyd/')
            dir_src = os.path.join(dump_dir_split, 'src/pyd/')
            dir_ref = os.path.join(dump_dir_split, 'ref/pyd/')
            dir_raw = os.path.join(dump_dir_split, 'raw/')
            dir_pred_trans = os.path.join(dump_dir_split, 'pred_transform/pyd/')
            dir_gt_trans = os.path.join(dump_dir_split, 'gt_transform/pyd/')
            dir_feat_src = os.path.join(dump_dir_split, 'feat_src/pyd/')
            dir_feat_ref = os.path.join(dump_dir_split, 'feat_ref/pyd/')

            for dir_ in [dir_pred_cloud, dir_src, dir_raw, dir_ref, dump_dir_together,
                         dir_pred_trans, dir_gt_trans, dir_feat_src, dir_feat_ref]:
                if not os.path.exists(dir_):
                    os.makedirs(dir_)

        for batch_idx, val_data in enumerate(tqdm(data_loader)):
            if _args.feat_dumpdir is not None:
                dump_dir_together_batch = os.path.join(dump_dir_together, 'batch_{}/'.format(batch_idx))
                if not os.path.exists(dump_dir_together_batch):
                    os.makedirs(dump_dir_together_batch)
            num_samples += val_data['points_src'].shape[0]
            rot_trace = val_data['transform_gt'][:, 0, 0] + val_data['transform_gt'][:, 1, 1] + \
                        val_data['transform_gt'][:, 2, 2]

            rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
            total_rotation.append(np.abs(to_numpy(rotdeg)))

            dict_all_to_device(val_data, _device)
            time_before = time.time()
            # (iter_num, batch_size, 3, 4)
            pred_transforms, endpoints = model(val_data, _args.num_reg_iter, _args.noise_type)
            total_time += time.time() - time_before

            # (batchsize, feat_dim),  (batchsize, feat_dim), (iter, batchsize, ...)
            feat_src_batch, feat_ref_batch, perm_matrix_batch \
                = endpoints['feat_src_cluster'], endpoints['feat_ref_cluster'], endpoints['perm_matrices']
            cloud_src_batch = val_data['points_src'][..., :3]  # (val_batch_size, num_points, 6)
            cloud_ref_batch = val_data['points_ref'][..., :3]
            gt_transforms_batch = val_data['transform_gt']
            cloud_raw_batch = val_data['points_raw'][..., :3]

            mask_batch = val_data['mask'] if _args.noise_type == 'crop' else None

            pred_transforms_batch = se3.concatenate(pred_transforms[_args.num_reg_iter - 1],
                                                    se3.inverse(gt_transforms_batch))

            cloud_pred_batch = se3.transform(pred_transforms_batch, cloud_raw_batch)

            if _args.feat_dumpdir is not None:
                for sample_idx in range(cloud_src_batch.shape[0]):  # test集大小不一定整除val_batch_size
                    dump_dir_together_sample = os.path.join(dump_dir_together_batch, 'sample_{}/'.format(sample_idx))
                    if not os.path.exists(dump_dir_together_sample):
                        os.makedirs(dump_dir_together_sample)

                    sample_cloud_raw = to_numpy(cloud_raw_batch[sample_idx])
                    sample_cloud_src, sample_cloud_ref, sample_cloud_pred = \
                        to_numpy(cloud_src_batch[sample_idx]), to_numpy(cloud_ref_batch[sample_idx]), to_numpy(
                            cloud_pred_batch[sample_idx])
                    sample_feat_src, sample_feat_ref = \
                        to_numpy(feat_src_batch[sample_idx]), to_numpy(feat_ref_batch[sample_idx])
                    sample_pred_transform = to_numpy(pred_transforms_batch[sample_idx])
                    sample_gt_transform = to_numpy(gt_transforms_batch[sample_idx])
                    sample_mask = to_numpy(mask_batch[sample_idx]) if _args.noise_type == 'crop' else None
                    # sample_feat_src = {
                    #     'xyz' : to_numpy(feat_src_outdim['xyz'][0]),
                    #     'dxyz': to_numpy(feat_src_outdim['dxyz'][0]),
                    #     'ppf' : to_numpy(feat_src_outdim['ppf'][0]),
                    # }

                    # generate_rectangular_obj(os.path.join(dir_raw, 'raw_cloud_{}.obj'.format(cnt)),
                    #                          points=sample_cloud_raw)
                    # generate_rectangular_obj(os.path.join(dir_src, 'src_cloud_{}.obj'.format(cnt)),
                    #                          points=sample_cloud_src)
                    # generate_rectangular_obj(os.path.join(dir_ref, 'ref_cloud_{}.obj'.format(cnt)),
                    #                          points=sample_cloud_ref)
                    # generate_rectangular_obj(os.path.join(dir_pred_cloud, 'pred_cloud_{}.obj'.format(cnt)),
                    #                          points=sample_cloud_pred)
                    #
                    # np.savetxt(os.path.join(dir_pred_trans, 'pred_transform_{}.txt'.format(cnt)),
                    #            sample_pred_transform, fmt='%.08f')
                    # np.savetxt(os.path.join(dir_gt_trans, 'gt_transform_{}.txt'.format(cnt)),
                    #            sample_gt_transform, fmt='%.08f')
                    #
                    # with open(os.path.join(dir_feat_src, 'feat_src_{}.txt'.format(cnt)), 'wb') as f:
                    #     pickle.dump(sample_feat_src, f)
                    # with open(os.path.join(dir_feat_ref, 'feat_ref_{}.txt'.format(cnt)), 'wb') as f:
                    #     pickle.dump(sample_feat_ref, f)
                    #
                    #

                    generate_rectangular_obj(os.path.join(dump_dir_together_sample, 'raw_cloud.obj'),
                                             points=to_numpy(sample_cloud_raw))
                    generate_rectangular_obj(os.path.join(dump_dir_together_sample, 'src_cloud.obj'),
                                             points=to_numpy(sample_cloud_src))
                    generate_rectangular_obj(os.path.join(dump_dir_together_sample, 'ref_cloud.obj'),
                                             points=to_numpy(sample_cloud_ref))
                    generate_rectangular_obj(os.path.join(dump_dir_together_sample, 'pred_cloud.obj'),
                                             points=to_numpy(sample_cloud_pred))

                    np.savetxt(os.path.join(dump_dir_together_sample, 'pred_transform.txt'),
                               to_numpy(sample_pred_transform), fmt='%.08f')
                    np.savetxt(os.path.join(dump_dir_together_sample, 'gt_transform.txt'),
                               to_numpy(sample_gt_transform), fmt='%.08f')

                    # with open(os.path.join(dump_dir_together_sample, 'feat_src.txt'), 'wb') as f:
                    #     pickle.dump(to_numpy(sample_feat_src), f)
                    # with open(os.path.join(dump_dir_together_sample, 'feat_ref.txt'), 'wb') as f:
                    #     pickle.dump(to_numpy(sample_feat_ref), f)

                    if _args.noise_type == 'crop':
                        save_similarity_plot(cloud_src=sample_cloud_src, cloud_ref=sample_cloud_ref,
                                             cloud_pred=sample_cloud_pred[sample_mask],
                                             gt_transform=torch.from_numpy(sample_gt_transform),
                                             save_dir=dump_dir_together_sample)
                    else:
                        save_similarity_plot(cloud_src=sample_cloud_src, cloud_ref=sample_cloud_ref,
                                             cloud_pred=sample_cloud_pred,
                                             gt_transform=torch.from_numpy(sample_gt_transform),
                                             save_dir=dump_dir_together_sample)

                    cnt += 1

            if _args.method == 'pydnet':
                all_betas.append(endpoints['beta'])
                all_alphas.append(endpoints['alpha'])

            if isinstance(pred_transforms[-1], torch.Tensor):
                pred_transforms_all.append(to_numpy(torch.stack(pred_transforms, dim=1)))
            else:
                pred_transforms_all.append(np.stack(pred_transforms, axis=1))

            # Saves match matrix. We only save the top matches to save storage/time.
            # However, this still takes quite a bit of time to save. Comment out if not needed.
            # if 'perm_matrices' in endpoints:
            #     perm_matrices = to_numpy(torch.stack(endpoints['perm_matrices'], dim=1))
            #     # thresh = np.percentile(perm_matrices, 99.9, axis=[2, 3])  # Only retain top 0.1% of entries
            #     # below_thresh_mask = perm_matrices < thresh[:, :, None, None]
            #     # perm_matrices[below_thresh_mask] = 0.0
            #
            #     for i_data in range(perm_matrices.shape[0]):
            #         sparse_perm_matrices = []
            #
            #         if _args.save_sparse_perm_matrix:
            #             sparse_perm_matrices.append(
            #                 sparse.coo_matrix(perm_matrices[i_data, perm_matrices.shape[1] - 1, :, :]))
            #         else:
            #             sparse_perm_matrices.append(perm_matrices[i_data, perm_matrices.shape[1] - 1, :, :])
            #         # for i_iter in range(perm_matrices.shape[1]):
            #         # sparse_perm_matrices.append(sparse.coo_matrix(perm_matrices[i_data, i_iter, :, :]))
            #         endpoints_out['perm_matrices'].append(sparse_perm_matrices)
    _logger.info('Total sample: {}'.format(num_samples))
    _logger.info('Total inference time: {}s, avg time sample: {}s'.format(total_time, total_time/num_samples))
    total_rotation = np.concatenate(total_rotation, axis=0)
    _logger.info('Rotation range in data: {}(avg), {}(max)'.format(np.mean(total_rotation), np.max(total_rotation)))
    pred_transforms_all = np.concatenate(pred_transforms_all, axis=0)

    return pred_transforms_all, endpoints_out


def evaluate(pred_transforms, data_loader: torch.utils.data.dataloader.DataLoader):
    """ Evaluates the computed transforms against the groundtruth

    Args:
        pred_transforms: Predicted transforms (B, [iter], 3/4, 4)
        data_loader: Loader for dataset.

    Returns:
        Computed metrics (List of dicts), and summary metrics (only for last iter)
    """

    _logger.info('Evaluating transforms...')
    num_processed, num_total = 0, len(pred_transforms)

    if pred_transforms.ndim == 4:
        pred_transforms = torch.from_numpy(pred_transforms).to(_device)
    else:
        assert pred_transforms.ndim == 3 and \
               (pred_transforms.shape[1:] == (4, 4) or pred_transforms.shape[1:] == (3, 4))
        pred_transforms = torch.from_numpy(pred_transforms[:, None, :, :]).to(_device)

    metrics_for_iter = [defaultdict(list) for _ in range(pred_transforms.shape[1])]

    # each batch
    for data in tqdm(data_loader, leave=False):
        dict_all_to_device(data, _device)
        # data.keys: (batch_size, n_points, X)

        batch_size = 0

        for i_iter in range(pred_transforms.shape[1]):
            batch_size = data['points_src'].shape[0]

            cur_pred_transforms = pred_transforms[num_processed:num_processed + batch_size, i_iter, :, :]  # each batch

            metrics = compute_metrics(data, cur_pred_transforms)
            metrics['label'] = to_numpy(data['label'])
            for k in metrics:
                metrics_for_iter[i_iter][k].append(metrics[k])
        num_processed += batch_size

    for i_iter in range(len(metrics_for_iter)):
        # metrics_for_iter[i_iter][key]: (num_samples,)
        metrics_for_iter[i_iter] = {k: np.concatenate(metrics_for_iter[i_iter][k], axis=0)
                                    for k in metrics_for_iter[i_iter]}

        # summary_metrics[key]: (num_samples, )
        summary_metrics, mean_metrics_each_category = summarize_metrics(metrics_for_iter[i_iter], eval_mode=True)
        print_metrics(_logger, summary_metrics, title='Evaluation result (iter {})'.format(i_iter))

    return metrics_for_iter, summary_metrics, mean_metrics_each_category


def save_eval_data(pred_transforms, endpoints, metrics: List, summary_metrics, save_path, metrics_each_category=None):
    """Saves out the computed transforms
    """

    # Save transforms
    np.save(os.path.join(save_path, 'pred_transforms.npy'), pred_transforms)

    # Save endpoints if any
    for k in endpoints:

        if isinstance(endpoints[k], np.ndarray):
            np.save(os.path.join(save_path, '{}.npy'.format(k)), endpoints[k])
        else:
            if not _args.do_not_save_perm_matrix:
                with open(os.path.join(save_path, '{}.pickle'.format(k)), 'wb') as fid:
                    pickle.dump(endpoints[k], fid)

    # Save metrics: Write each iteration to a different worksheet.
    writer = pd.ExcelWriter(os.path.join(save_path, 'metrics.xlsx'))
    for i_iter in range(len(metrics)):
        metrics[i_iter]['r_rmse'] = np.sqrt(metrics[i_iter]['r_mse'])
        metrics[i_iter]['t_rmse'] = np.sqrt(metrics[i_iter]['t_mse'])
        metrics[i_iter].pop('r_mse')
        metrics[i_iter].pop('t_mse')
        metrics_df = pd.DataFrame.from_dict(metrics[i_iter])
        metrics_df.to_excel(writer, sheet_name='Iter_{}'.format(i_iter + 1))
    writer.close()

    # Save overall summary metrics
    summary_metrics_float = {k: float(summary_metrics[k]) for k in summary_metrics}
    summary_metrics_float.pop('label')
    with open(os.path.join(save_path, 'summary_metrics.json'), 'w') as json_out:
        json.dump(summary_metrics_float, json_out)

    # Save each category summary metrics
    # metrics_each_category: { key: {label: float} }
    if metrics_each_category is not None:
        metrics_each_category_float = {}
        for key, value in metrics_each_category.items():
            metrics_each_category_float[key] = {str(k): float(value[k]) for k in value}
        with open(os.path.join(save_path, 'mean_metrics_each_category.json'), 'w') as json_out:
            json.dump(metrics_each_category_float, json_out)
        # for key, value in metrics_each_category.items():
        #     with open(os.path.join(save_path, 'mean_{}_metrics_each_category.json'.format(key)), 'w') as json_out:
        #         value_float = {str(k): float(value[k]) for k in value}
        #         json.dump(value_float, json_out)

    _logger.info('Saved evaluation results to {}'.format(save_path))


def get_model():
    _logger.info('Computing transforms using {}'.format(_args.method))
    if _args.method == 'pydnet':
        assert _args.resume is not None
        model = modules.pydnet.get_model(_args)
        model.to(_device)
        saver = CheckPointManager(os.path.join(_log_path, 'ckpt', 'models'))
        saver.load(_args.resume, model)
    else:
        raise NotImplementedError
    return model


def main():
    # Load data_loader
    test_dataset = get_test_datasets(_args)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=_args.val_batch_size, shuffle=False)

    if _args.transform_file is not None:
        _logger.info('Loading from precomputed transforms: {}'.format(_args.transform_file))
        pred_transforms = np.load(_args.transform_file)
        endpoints = {}
    else:
        model = get_model()
        pred_transforms, endpoints = inference(test_loader, model)  # Feedforward transforms

    # Compute evaluation matrices
    eval_metrics, summary_metrics, mean_metrics_each_category = evaluate(pred_transforms, data_loader=test_loader)

    save_eval_data(pred_transforms, endpoints, eval_metrics,
                   summary_metrics, _args.eval_save_path, metrics_each_category=mean_metrics_each_category)
    _logger.info('Finished')


if __name__ == '__main__':
    # Arguments and logging
    parser = pydnet_eval_arguments()
    _args = parser.parse_args()
    _logger, _log_path = prepare_logger(_args, log_path=_args.eval_save_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)

    if _args.feat_dumpdir is not None:
        _logger.info('Feature will be dumped into {}'.format(_args.feat_dumpdir))

    if _args.gpu >= 0 and (_args.method == 'pyramid' or _args.method == 'pydnet'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
        _device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    else:
        _device = torch.device('cpu')

    main()
