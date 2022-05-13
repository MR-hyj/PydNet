# -*- encoding: utf-8 -*-

# Author:   Hengyu Jiang
# Time:     2022/3/3 13:47
# Email:    hengyujiang.njust.edu.cn
# Project:  PydNet
# File:     eval_from_outer.py
# Product:  PyCharm
# Desc:     evaluate pcl regression results


import json
import os
import pickle
import time
from typing import Dict, List
from collections import defaultdict
import numpy as np
import open3d  # Need to import before torch
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import torch
from vispy import io
from arguments import pydnet_eval_from_outer_arugments
from common.misc import prepare_logger
from common.mytorch import dict_all_to_device, CheckPointManager, to_numpy
from common.math import se3
from common.math_torch import se3
from common.math.so3 import dcm2euler
from data_loader.datasets import get_test_datasets
import modules.pydnet
from common.io import generate_rectangular_obj
from copy import deepcopy


parser = pydnet_eval_from_outer_arugments()
_args = parser.parse_args()
_logger, _log_path = prepare_logger(_args, log_path=_args.eval_save_path)
cnt = 0

if _args.gpu >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
    _device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
else:
     _device = torch.device('cpu')


class OuterDataSet(torch.utils.data.Dataset):
    """

    """

    def __init__(self):
        self._src_clouds = read_obj_from_dir(_args.src_clouds_dir, prefix='src_cloud_')
        self._target_clouds = read_obj_from_dir(_args.ref_clouds_dir, prefix='src_cloud_')
        self._gt_transforms = read_transform_from_dir(_args.gt_transform_dir, prefix='gt_transform_')


    def __getitem__(self, item):
        sample = {
            'points_src': self._src_clouds[item, :, :],
            'points_ref': self._target_clouds[item, :, :],
            'points_raw': deepcopy(self._src_clouds[item, :, :]),
            'transform_gt': self._gt_transforms[item, :, :]
        }
        return sample

    def __len__(self):
        return self._src_clouds.shape[0]


def print_metrics(logger, summary_metrics: Dict, losses_by_iteration: List = None,
                  title: str = 'Metrics'):
    """Prints out formated metrics to logger"""

    logger.info(title + ':')
    logger.info('=' * (len(title) + 1))

    if losses_by_iteration is not None:
        losses_all_str = ' | '.join(['{:.5f}'.format(c) for c in losses_by_iteration])
        logger.info('Losses by iteration: {}'.format(losses_all_str))

    logger.info('DeepCP metrics:{:.4f}(rot-rmse) | {:.4f}(rot-mae) | {:.4g}(trans-rmse) | {:.4g}(trans-mae)'.format(
        summary_metrics['r_rmse'], summary_metrics['r_mae'],
        summary_metrics['t_rmse'], summary_metrics['t_mae'],
    ))
    logger.info('Rotation error {:.4f}(deg, mean) | {:.4f}(deg, rmse)'.format(
        summary_metrics['err_r_deg_mean'], summary_metrics['err_r_deg_rmse']))
    logger.info('Translation error {:.4g}(mean) | {:.4g}(rmse)'.format(
        summary_metrics['err_t_mean'], summary_metrics['err_t_rmse']))
    logger.info('Chamfer error: {:.7f}(mean-sq)'.format(
        summary_metrics['chamfer_dist']
    ))



def evaluate(pred_transforms, data_loader: torch.utils.data.dataloader.DataLoader):
    """ Evaluates the computed transforms against the groundtruth

    Args:
        pred_transforms: Predicted transforms (B, 3/4, 4)
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
        pred_transforms = torch.from_numpy(pred_transforms[:, :, :]).to(_device)

    metrics_for_iter = [defaultdict(list)]

    # each batch
    for data in tqdm(data_loader, leave=False):
        dict_all_to_device(data, _device)

        batch_size = data['points_src'].shape[0]

        cur_pred_transforms = pred_transforms[num_processed:num_processed+batch_size, :, :] # each batch
        metrics = compute_metrics(data, cur_pred_transforms)
        for k in metrics:
            metrics_for_iter[0][k].append(metrics[k])
        num_processed += batch_size

    for i_iter in range(len(metrics_for_iter)):
        metrics_for_iter[i_iter] = {k: np.concatenate(metrics_for_iter[i_iter][k], axis=0)
                                    for k in metrics_for_iter[i_iter]}
        summary_metrics = summarize_metrics(metrics_for_iter[i_iter])
        print_metrics(_logger, summary_metrics, title='Evaluation result (iter {})'.format(i_iter))

    return metrics_for_iter, summary_metrics


def summarize_metrics(metrics):
    """Summaries computed metrics by taking mean over all data instances"""
    summarized = {}
    for k in metrics:
        if k.endswith('mse'):
            summarized[k[:-3] + 'rmse'] = np.sqrt(np.mean(metrics[k]))
        elif k.startswith('err'):
            summarized[k + '_mean'] = np.mean(metrics[k])
            summarized[k + '_rmse'] = np.sqrt(np.mean(metrics[k]**2))
        else:
            summarized[k] = np.mean(metrics[k])

    return summarized


def compute_metrics(data: Dict, pred_transforms) -> Dict:
    """Compute metrics required in the paper
    eval single
    """

    def square_distance(src, dst):
        return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)

    with torch.no_grad():
        pred_transforms = pred_transforms
        gt_transforms = data['transform_gt']
        points_src = data['points_src'][..., :3]
        points_ref = data['points_ref'][..., :3]
        points_raw = data['points_raw'][..., :3]

        _logger.debug('computing metrics, input samples {}'.format(points_raw.shape[0]))

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
        src_transformed = se3.transform(pred_transforms, points_src)
        global cnt
        if _args.pred_cloud_dir is not None:
            if not os.path.exists(_args.pred_cloud_dir):
                os.makedirs(_args.pred_cloud_dir)
            _logger.debug('saving predicted clouds')
            for idx, pred_cloud in enumerate(src_transformed):
                pred_cloud = to_numpy(pred_cloud)
                save_name = os.path.join(_args.pred_cloud_dir, 'pred_cloud_{}.obj'.format(cnt))
                generate_rectangular_obj(save_name, points=pred_cloud)
                cnt += 1

        ref_clean = points_raw
        src_clean = se3.transform(se3.concatenate(pred_transforms, se3.inverse(gt_transforms)), points_raw)
        dist_src = torch.min(square_distance(src_transformed, ref_clean), dim=-1)[0]
        dist_ref = torch.min(square_distance(points_ref, src_clean), dim=-1)[0]
        chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)


        metrics = {
            'r_mse': r_mse,
            'r_mae': r_mae,
            't_mse': to_numpy(t_mse),
            't_mae': to_numpy(t_mae),
            'err_r_deg': to_numpy(residual_rotdeg),
            'err_t': to_numpy(residual_transmag),
            'chamfer_dist': to_numpy(chamfer_dist)
        }

    return metrics


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
    with torch.no_grad():
        for batch_idx, val_data in tqdm(enumerate(data_loader)):
            if _args.feat_dumpdir is not None:
                dump_dir_batch = os.path.join(_args.feat_dumpdir, 'batch_{}/'.format(batch_idx))
            rot_trace = val_data['transform_gt'][:, 0, 0] + val_data['transform_gt'][:, 1, 1] + \
                        val_data['transform_gt'][:, 2, 2]


            rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
            total_rotation.append(np.abs(to_numpy(rotdeg)))

            dict_all_to_device(val_data, _device)
            time_before = time.time()
            pred_transforms, endpoints, feat_indim, feat_outdim = model(val_data, _args.num_reg_iter)
            total_time+= time.time() - time_before

            if _args.method == 'pydnet':
                all_betas.append(endpoints['beta'])
                all_alphas.append(endpoints['alpha'])

            if isinstance(pred_transforms[-1], torch.Tensor):
                pred_transforms_all.append(to_numpy(torch.stack(pred_transforms, dim=1)))
            else:
                pred_transforms_all.append(np.stack(pred_transforms, axis=1))

            # Saves match matrix. We only save the top matches to save storage/time.
            # However, this still takes quite a bit of time to save. Comment out if not needed.
            if 'perm_matrices' in endpoints:
                perm_matrices = to_numpy(torch.stack(endpoints['perm_matrices'], dim=1))
                thresh = np.percentile(perm_matrices, 99.9, axis=[2, 3])  # Only retain top 0.1% of entries
                below_thresh_mask = perm_matrices < thresh[:, :, None, None]
                perm_matrices[below_thresh_mask] = 0.0

                for i_data in range(perm_matrices.shape[0]):
                    sparse_perm_matrices = []
                    for i_iter in range(perm_matrices.shape[1]):
                        sparse_perm_matrices.append(sparse.coo_matrix(perm_matrices[i_data, i_iter, :, :]))
                    endpoints_out['perm_matrices'].append(sparse_perm_matrices)

    _logger.info('Total inference time: {}s'.format(total_time))
    total_rotation = np.concatenate(total_rotation, axis=0)
    _logger.info('Rotation range in data: {}(avg), {}(max)'.format(np.mean(total_rotation), np.max(total_rotation)))
    pred_transforms_all = np.concatenate(pred_transforms_all, axis=0)

    return pred_transforms_all, endpoints_out



def save_eval_data(pred_transforms, metrics, summary_metrics, save_path, endpoints=None):
    """Saves out the computed transforms
    """

    # Save transforms
    np.save(os.path.join(save_path, 'pred_transforms.npy'), pred_transforms)

    # Save endpoints if any
    if endpoints is not None:
        for k in endpoints:
            if isinstance(endpoints[k], np.ndarray):
                np.save(os.path.join(save_path, '{}.npy'.format(k)), endpoints[k])
            else:
                with open(os.path.join(save_path, '{}.pickle'.format(k)), 'wb') as fid:
                    pickle.dump(endpoints[k], fid)

    # Save metrics: Write each iteration to a different worksheet.
    writer = pd.ExcelWriter(os.path.join(save_path, 'metrics.xlsx'))
    for i_iter in range(len(metrics)):
        metrics[i_iter]['r_mse'] = np.sqrt(metrics[i_iter]['r_mse'])
        metrics[i_iter]['t_mse'] = np.sqrt(metrics[i_iter]['t_mse'])
        metrics[i_iter].pop('r_mse')
        metrics[i_iter].pop('t_mse')
        metrics_df = pd.DataFrame.from_dict(metrics[i_iter])
        metrics_df.to_excel(writer, sheet_name='Iter_{}'.format(i_iter+1))
    writer.close()

    # Save summary metrics
    summary_metrics_float = {k: float(summary_metrics[k]) for k in summary_metrics}
    with open(os.path.join(save_path, 'summary_metrics.json'), 'w') as json_out:
        json.dump(summary_metrics_float, json_out)

    _logger.info('Saved evaluation results to {}'.format(save_path))


def read_obj_from_dir(path, prefix=None):

    files = os.listdir(path)
    _logger.debug('loading obj from {}, {} files to go'.format(path, len(files)))
    files = sorted(files, key=lambda f: int(f[len(prefix): len(f)-4]))
    clouds = []
    for idx, file in enumerate(files):
        verts, faces, normals, nothing = io.read_mesh(os.path.join(path, file))
        clouds.append(verts)
    return torch.from_numpy(np.array(clouds, dtype=np.float)).to(_device)

def read_transform_from_dir(path, prefix=None):

    files = os.listdir(path)
    _logger.debug('loading transform from {}, {} files to go'.format(path, len(files)))

    files = sorted(files, key=lambda f: int(f[len(prefix): len(f) - 4]))
    transforms = []
    for idx, file in enumerate(files):
        transform = np.loadtxt(os.path.join(path, file))[:3, :]
        transforms.append(transform)
    return np.array(transforms, dtype=np.float)



def main():


    _logger.info('evaluating {}'.format(_args.pred_transform_dir))
    data_set = OuterDataSet()
    data_loader = torch.utils.data.DataLoader(data_set,
                                              batch_size=_args.val_batch_size, shuffle=False)

    pred_transforms = read_transform_from_dir(_args.pred_transform_dir, prefix='pred_transform_')


    eval_metrics, summary_metrics = evaluate(pred_transforms, data_loader)
    save_eval_data(pred_transforms, eval_metrics, summary_metrics, _args.eval_save_path)


if __name__ == '__main__':

    main()
