# -*- encoding: utf-8 -*-

# Author:   Hengyu Jiang
# Time:     2022/1/5 16:09
# Email:    hengyujiang.njust.edu.cn
# Project:  PydNet
# File:     train.py
# Product:  PyCharm
# Desc:
import pickle
from matplotlib import pyplot as plt
from modules.feature_net import FeatExtractionEarlyFusion
from arguments import pydnet_train_arguments
from collections import defaultdict
import os
import random
from typing import Dict, List
from matplotlib.pyplot import cm as colormap
import numpy as np
import open3d  # Ensure this is imported before pytorch
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
import data_loader.datasets
from common.colors import BLUE, ORANGE
from common.misc import prepare_logger
from common.mytorch import dict_all_to_device, CheckPointManager, TorchDebugger, to_numpy
from common.math_torch import se3
from data_loader.datasets import get_train_datasets
from eval import compute_metrics, summarize_metrics, print_metrics
from modules.pydnet import get_model
from common.io import generate_rectangular_obj

parser = pydnet_train_arguments()
_args = parser.parse_args()
_logger, _log_path = prepare_logger(_args)
if _args.gpu >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
    _device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
else:
    _device = torch.device('cpu')
_logger.info('Using device {}'.format(_device))


def compute_losses(data: Dict, pred_transforms: List, endpoints: Dict,
                   loss_type: str = 'mse', reduction: str = 'mean', model: torch.nn.Module = None,
                   lamb: float = 0.15, omega_1: float = 3, omega_2: float = 1) -> Dict:
    """
    Compute losses


    @param data: Current mini-batch data
    @param pred_transforms: Predicted transform, to compute main registration loss
    @param endpoints: Endpoints for training. For computing outlier penalty
    @param loss_type: Registration loss type, either 'cluster' or 'pmd'
    @param reduction: Either 'mean' or 'none'. Use 'none' to accumulate losses outside (useful for accumulating losses for entire validation dataset)
    @param model:
    @param lamb:
    @param omega_1:
    @param omega_2:
    @return: Dict containing various fields. Total loss to be optimized is in losses['total']
    """
    # train_data = {
    #     idx:              tensor, (batchsize,)
    #     points_raw:       tensor, (batchsize, 2048, 6), (xyz, nxyz)
    #     points_src:       tensor, (batchsize, 717, 6),  (xyz, nxyz)
    #     points_ref:       tensor, (batchsize, 717, 6),  (xyz, nxyz)
    #     crop_propotion:   tensor, (batchsize, 2)
    #     transform_gt:     tensor, (batchsize, 3, 4), 前三列是so3, 最后一列是translation
    # }

    # pred_transform = (iter, batchsize, 3, 4)

    # endpoints = {
    #     'perm_matrices_init':   tensor          (iter, batchsize, 717, 717)
    #     'perm_matrices':        tensor          (iter, batchsize, 717, 717)
    #     'weighted_ref':         tensor          (iter, batchsize, 717, 3)
    #     'beta':                 np.ndarray      (iter, batchsize, )
    #     'alpha':                np.ndarray      (iter, batchsize, )
    # }

    # losses = {}
    # num_iter = len(pred_transforms)
    #
    # # Compute losses
    # gt_src_transformed = se3.transform(data['transform_gt'], data['points_src'][..., :3])
    # if loss_type == 'mse':
    #     # MSE loss to the groundtruth (does not take into account possible symmetries)
    #     criterion = nn.MSELoss(reduction=reduction)
    #     for i in range(num_iter):
    #         pred_src_transformed = se3.transform(pred_transforms[i], data['points_src'][..., :3])
    #         if reduction.lower() == 'mean':
    #             losses['mse_{}'.format(i)] = criterion(pred_src_transformed, gt_src_transformed)
    #         elif reduction.lower() == 'none':
    #             losses['mse_{}'.format(i)] = torch.mean(criterion(pred_src_transformed, gt_src_transformed),
    #                                                     dim=[-1, -2])
    # elif loss_type == 'mae':
    #     # MSE loss to the groundtruth (does not take into account possible symmetries)
    #     criterion = nn.L1Loss(reduction=reduction)
    #     for i in range(num_iter):
    #         pred_src_transformed = se3.transform(pred_transforms[i], data['points_src'][..., :3])
    #         if reduction.lower() == 'mean':
    #             losses['mae_{}'.format(i)] = criterion(pred_src_transformed, gt_src_transformed)
    #         elif reduction.lower() == 'none':
    #             losses['mae_{}'.format(i)] = torch.mean(criterion(pred_src_transformed, gt_src_transformed),
    #                                                     dim=[-1, -2])
    # else:
    #     raise NotImplementedError

    # # Penalize outliers
    # for i in range(num_iter):
    #     ref_outliers_strength = (1.0 - torch.sum(endpoints['perm_matrices'][i], dim=1)) * _args.wt_inliers
    #     src_outliers_strength = (1.0 - torch.sum(endpoints['perm_matrices'][i], dim=2)) * _args.wt_inliers
    #     if reduction.lower() == 'mean':
    #         losses['outlier_{}'.format(i)] = torch.mean(ref_outliers_strength) + torch.mean(src_outliers_strength)
    #     elif reduction.lower() == 'none':
    #         losses['outlier_{}'.format(i)] = torch.mean(ref_outliers_strength, dim=1) + \
    #                                          torch.mean(src_outliers_strength, dim=1)

    # discount_factor = 0.5  # Early iterations will be discounted
    # _logger.info('{}'.format(losses))
    # # 将所有的loss添加到total_loss, 然后求和
    # total_losses = []
    # for k in losses:
    #     discount = discount_factor ** (num_iter - int(k[k.rfind('_') + 1:]) - 1)
    #     total_losses.append(losses[k] * discount)
    # losses['total'] = torch.sum(torch.stack(total_losses), dim=0)

    # return losses

    losses = {}
    num_iter = len(pred_transforms)
    gt_src_transformed = se3.transform(data['transform_gt'], data['points_src'][..., :3])

    # MSE loss to the groundtruth (does not take into account possible symmetries)
    criterion = nn.MSELoss(reduction=reduction)
    for i in range(num_iter):
        pred_src_transformed = se3.transform(pred_transforms[i], data['points_src'][..., :3])
        if reduction.lower() == 'mean':
            losses['mse_{}'.format(i)] = criterion(pred_src_transformed, gt_src_transformed)
        elif reduction.lower() == 'none':
            losses['mse_{}'.format(i)] = torch.mean(criterion(pred_src_transformed, gt_src_transformed),
                                                    dim=[-1, -2])

    for i in range(num_iter):
        ref_outliers_strength = (1.0 - torch.sum(endpoints['perm_matrices'][i], dim=1)) * _args.wt_inliers
        src_outliers_strength = (1.0 - torch.sum(endpoints['perm_matrices'][i], dim=2)) * _args.wt_inliers
        if reduction.lower() == 'mean':
            losses['outlier_{}'.format(i)] = torch.mean(ref_outliers_strength) + torch.mean(src_outliers_strength)
        elif reduction.lower() == 'none':
            losses['outlier_{}'.format(i)] = torch.mean(ref_outliers_strength, dim=1) + \
                                             torch.mean(src_outliers_strength, dim=1)

    discount_factor = 0.5  # Early iterations will be discounted

    # 将所有的loss添加到total_loss, 然后求和
    total_losses = []

    # loss_tr
    for k in losses:
        discount = discount_factor ** (num_iter - int(k[k.rfind('_') + 1:]) - 1)
        total_losses.append(losses[k] * discount)
    loss_tr = torch.sum(torch.stack(total_losses), dim=0)

    # loss_fe
    # cluster: (B, npoints, C)
    #
    # pmd: (xyz, {dxyz}, {pmd})
    # xyz:  (B, npoints, 3)
    # dxyz: (B, npoints, neighbors, 3)
    # pmd:  (B, npoints, 3)
    if 'cluster' == loss_type:
        feat_src_cluster = endpoints['feat_src_cluster']
        xyz_src, norm_src = data['points_src'][:, :, :3], data['points_src'][:, :, 3:6]
        xyz_gt, norm_gt = se3.transform(data['transform_gt'], xyz_src, normals=norm_src)
        feat_ref_cluster, feat_ref = model.feat_extractor(xyz_gt, norm_gt)
        loss_fe = distance_between_clusters(feat_src_cluster, feat_ref_cluster)
    elif 'pmd' == loss_type:
        gt_transform, pred_transform = data['transform_gt'], pred_transforms[num_iter - 1]
        feat_src_cluster = endpoints['feat_src_pmd']
        loss_fe = distance_between_pmds(feat_src_cluster, gt_transform, pred_transform, omega_1=omega_1,
                                        omega_2=omega_2)
    else:
        raise NotImplementedError

    _logger.debug('loss_tr: {}, loss_fe: {}, loss_fe type: {}'.format(loss_tr, loss_fe, loss_type))
    losses['tr'] = loss_tr
    losses['fe'] = loss_fe
    losses['total'] = (1 - lamb) * loss_tr + lamb * loss_fe
    return losses


def transform_pmd(pmd: dict, transform: np.ndarray):
    """
    set transform to a pmd feature
    # pmd: (xyz, {dxyz}, {pmd})
    # transform (B, 3, 1, 4)
    # xyz:  (B, npoints, 3)
    # dxyz: (B, npoints, neighbors, 3)
    # pmd:  (B, npoints, 1, 3)
    # 只有xyz与dxyz是对transform敏感的
    """
    B, npoints, neighbors, _ = pmd['dxyz'].shape
    xyz_ = torch.reshape(pmd['xyz'], (B, npoints, 3))
    xyz_transform = torch.reshape(se3.transform(transform, xyz_), (B, npoints, 1, 3))
    dxyz_ = torch.reshape(pmd['dxyz'], shape=(B, npoints * neighbors, _))
    dxyz_transform = se3.transform(transform, dxyz_)
    dxyz_transform = torch.reshape(dxyz_transform, shape=(B, npoints, neighbors, _))
    return {'xyz': xyz_transform, 'dxyz': dxyz_transform, 'pmd': pmd['pmd']}


def distance_between_pmds(pmd_src: dict, gt_transform: np.ndarray, pred_transform: np.ndarray,
                          omega_1: float = 3, omega_2: float = 1):
    # pmd: (xyz, {dxyz}, {pmd})
    # transform (B, 3, 4)
    # xyz:  (B, npoints, 1, 3)
    # dxyz: (B, npoints, neighbors, 3)
    # pmd:  (B, npoints, 1, 4)
    B, npoints, neighbors, _ = pmd_src['dxyz'].shape
    _logger.debug('B={}, npoints={}, neighbors={}, _={}'.format(B, npoints, neighbors, _))
    _logger.debug('input pmd: xyz: {}, dxyz: {}, pmd: {}'.format(pmd_src['xyz'].shape, pmd_src['dxyz'].shape,
                                                                 pmd_src['pmd'].shape))

    pmd_gt = transform_pmd(pmd_src, gt_transform)
    pmd_pred = transform_pmd(pmd_src, pred_transform)

    pmd_loss = []
    # 对于xyz, 直接求欧氏距离作为误差
    # xyz_gt = (B, npoints, 1, 3)
    xyz_gt, xyz_pred = pmd_gt['xyz'], pmd_pred['xyz']
    xyz_gt = torch.reshape(xyz_gt, (B * npoints, 3))
    xyz_pred = torch.reshape(xyz_pred, (B * npoints, 3))
    pmd_loss.append(
        torch.mean(torch.norm(xyz_gt - xyz_pred, p=2, dim=1, keepdim=False)) * (omega_1 / (omega_1 + omega_2)))

    # 对于dxyz同理
    dxyz_gt, dxyz_pred = pmd_gt['dxyz'], pmd_pred['dxyz']
    dxyz_gt = torch.reshape(dxyz_gt, (B * npoints * neighbors, 3))
    dxyz_pred = torch.reshape(dxyz_pred, (B * npoints * neighbors, 3))
    pmd_loss.append(
        torch.mean(torch.norm(dxyz_gt - dxyz_pred, p=2, dim=1, keepdim=False)) * (omega_1 / (omega_1 + omega_2)))

    # 对于pmd, (B, npoint, 1, 5)包含4个个角度和1个偏移距离, 计算所有角度值的cos值的差的绝对值的平均值, 这个值作为相似度, 用1-相似度得到距离
    # TODO 是否可以将最后一维偏移距离也当作角度求解相似度?
    angles_gt, angles_pred = pmd_gt['pmd'], pmd_pred['pmd']
    cos_gt, cos_pred = torch.cos(angles_gt), torch.cos(angles_pred)
    similarity = torch.mean(torch.abs(cos_gt - cos_pred))
    # assert 0 <= similarity <= 1, _logger.error('similarity not in [0, 1]')

    pmd_loss.append((1 - similarity) * (omega_2 / (omega_1 + omega_2)))

    # /10 是为了减小其量级, 方便在一张png上画曲线
    return np.sum(pmd_loss)/30


def distance_between_clusters(cluster_1: torch.Tensor, cluster_2: torch.Tensor, p: int = 2):
    # cluster: (B, npoints, C) --> (B*npoints, C)
    _logger.debug('input cluster_1: {}'.format(cluster_1.shape))
    _logger.debug('input cluster_2: {}'.format(cluster_2.shape))

    cluster_1 = cluster_1.reshape(-1, cluster_1.shape[2])
    cluster_2 = cluster_2.reshape(-1, cluster_2.shape[2])

    _logger.debug('view cluster_1: {}'.format(cluster_1.shape))
    _logger.debug('view cluster_2: {}'.format(cluster_2.shape))

    # cluster_mean: (1, C)
    cluster_1_mean = torch.mean(cluster_1, dim=0, keepdim=False)
    cluster_2_mean = torch.mean(cluster_2, dim=0, keepdim=False)

    _logger.debug('mean cluster_1: {}'.format(cluster_1_mean.shape))
    _logger.debug('mean cluster_2: {}'.format(cluster_2_mean.shape))

    return torch.exp(torch.norm(cluster_1_mean - cluster_2_mean, p=p))


def save_summaries(writer: SummaryWriter, data: Dict, predicted: List, endpoints: Dict = None,
                   losses: Dict = None, metrics: Dict = None, step: int = 0):
    """Save tensorboard summaries"""

    subset = [0, 1]

    with torch.no_grad():
        # Save clouds
        if 'points_src' in data:

            points_src = data['points_src'][subset, ..., :3]
            points_ref = data['points_ref'][subset, ..., :3]

            colors = torch.from_numpy(
                np.concatenate([np.tile(ORANGE, (*points_src.shape[0:2], 1)),
                                np.tile(BLUE, (*points_ref.shape[0:2], 1))], axis=1))

            iters_to_save = [0, len(predicted) - 1] if len(predicted) > 1 else [0]

            # Save point cloud at iter0, iter1 and after last iter
            concat_cloud_input = torch.cat((points_src, points_ref), dim=1)
            writer.add_mesh('iter_0', vertices=concat_cloud_input, colors=colors, global_step=step)
            for i_iter in iters_to_save:
                src_transformed_first = se3.transform(predicted[i_iter][subset, ...], points_src)
                concat_cloud_first = torch.cat((src_transformed_first, points_ref), dim=1)
                writer.add_mesh('iter_{}'.format(i_iter + 1), vertices=concat_cloud_first, colors=colors,
                                global_step=step)

            if endpoints is not None and 'perm_matrices' in endpoints:
                color_mapper = colormap.ScalarMappable(norm=None, cmap=colormap.get_cmap('coolwarm'))
                for i_iter in iters_to_save:
                    ref_weights = torch.sum(endpoints['perm_matrices'][i_iter][subset, ...], dim=1)
                    ref_colors = color_mapper.to_rgba(ref_weights.detach().cpu().numpy())[..., :3]
                    writer.add_mesh('ref_weights_{}'.format(i_iter), vertices=points_ref,
                                    colors=torch.from_numpy(ref_colors) * 255, global_step=step)

        if endpoints is not None:
            if 'perm_matrices' in endpoints:
                for i_iter in range(len(endpoints['perm_matrices'])):
                    src_weights = torch.sum(endpoints['perm_matrices'][i_iter], dim=2)
                    ref_weights = torch.sum(endpoints['perm_matrices'][i_iter], dim=1)
                    writer.add_histogram('src_weights_{}'.format(i_iter), src_weights, global_step=step)
                    writer.add_histogram('ref_weights_{}'.format(i_iter), ref_weights, global_step=step)

        # Write losses and metrics
        if losses is not None:
            for loss_ in losses:
                writer.add_scalar('losses/{}'.format(loss_), losses[loss_], step)
        if metrics is not None:
            for m in metrics:
                writer.add_scalar('metrics/{}'.format(m), metrics[m], step)

        writer.flush()


def validate(data_loader, model: torch.nn.Module, summary_writer: SummaryWriter, step: int):
    """Perform a single validation run, and saves results into tensorboard summaries"""

    _logger.info('Starting validation run...')

    with torch.no_grad():
        all_val_losses = defaultdict(list)
        all_val_metrics_np = defaultdict(list)
        for val_data in data_loader:
            dict_all_to_device(val_data, _device)
            pred_test_transforms, endpoints = model(val_data, _args.num_reg_iter)
            val_losses = compute_losses(val_data, pred_test_transforms, endpoints,
                                        loss_type=_args.loss_type, reduction='none',
                                        lamb=_args.loss_lambda, omega_1=_args.loss_omega_1,
                                        omega_2=_args.loss_omega_2, model=model)
            val_metrics = compute_metrics(val_data, pred_test_transforms[-1])

            for k in val_losses:
                if k not in ['fe', 'tr']:
                    all_val_losses[k].append(val_losses[k])
            for k in val_metrics:
                all_val_metrics_np[k].append(val_metrics[k])

        all_val_losses = {k: torch.cat(all_val_losses[k]) for k in all_val_losses}
        all_val_metrics_np = {k: np.concatenate(all_val_metrics_np[k]) for k in all_val_metrics_np}
        mean_val_losses = {k: torch.mean(all_val_losses[k]) for k in all_val_losses}

    # Rerun on random and worst data instances and save to summary
    rand_idx = random.randint(0, all_val_losses['total'].shape[0] - 1)
    worst_idx = torch.argmax(all_val_losses['{}_{}'.format('mse', _args.num_reg_iter - 1)]).cpu().item()
    indices_to_rerun = [rand_idx, worst_idx]
    data_to_rerun = defaultdict(list)
    for i in indices_to_rerun:
        cur = data_loader.dataset[i]
        for k in cur:
            data_to_rerun[k].append(cur[k])
    for k in data_to_rerun:
        data_to_rerun[k] = torch.from_numpy(np.stack(data_to_rerun[k], axis=0))
    dict_all_to_device(data_to_rerun, _device)
    pred_transforms, endpoints = model(data_to_rerun, _args.num_reg_iter)

    summary_metrics = summarize_metrics(all_val_metrics_np)
    losses_by_iteration = torch.stack([mean_val_losses['{}_{}'.format('mse', k)]
                                       for k in range(_args.num_reg_iter)]).cpu().numpy()
    print_metrics(_logger, summary_metrics, losses_by_iteration, 'Validation results')

    save_summaries(summary_writer, data=data_to_rerun, predicted=pred_transforms, endpoints=endpoints,
                   losses=mean_val_losses, metrics=summary_metrics, step=step)

    score = -summary_metrics['chamfer_dist']
    return score, to_numpy(mean_val_losses['total']), summary_metrics


def run(train_set: data_loader.datasets.ModelNetHdf,
        val_set: data_loader.datasets.ModelNetHdf):
    """Main train/val loop"""

    _logger.debug('Trainer (PID=%d), %s', os.getpid(), _args)

    model = get_model(_args)
    model.to(_device)
    global_step = 0


    loss_train_all, loss_val_all, loss_tr_all, loss_fe_all = [], [], [], []

    err_r_deg_mean, err_r_deg_rmse, err_t_mean, err_t_rmse = [], [], [], []
    chamfer_distance = []

    # dataloaders
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=_args.train_batch_size, shuffle=True,
                                               num_workers=_args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=_args.val_batch_size, shuffle=False,
                                             num_workers=_args.num_workers)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=_args.lr)

    # Summary writer and Checkpoint manager
    train_writer = SummaryWriter(os.path.join(_log_path, 'train'), flush_secs=10)
    val_writer = SummaryWriter(os.path.join(_log_path, 'val'), flush_secs=10)
    saver = CheckPointManager(os.path.join(_log_path, 'ckpt', 'model'), keep_checkpoint_every_n_hours=0.5)
    if _args.resume is not None:
        global_step = saver.load(_args.resume, model, optimizer)

    # trainings
    torch.autograd.set_detect_anomaly(_args.debug)
    model.train()

    steps_per_epoch = len(train_loader)
    if _args.summary_every < 0:
        _args.summary_every = abs(_args.summary_every) * steps_per_epoch
    if _args.validate_every < 0:
        _args.validate_every = abs(_args.validate_every) * steps_per_epoch

    if _args.feat_dumpdir is not None:
        _logger.info('Features will be dumped into {}'.format(_args.feat_dumpdir))
        if not os.path.exists(_args.feat_dumpdir):
            os.makedirs(_args.feat_dumpdir)

    try:
    # if True:
        for epoch in range(0, _args.epochs):
            _logger.info('Begin epoch {} (steps {} - {})'.format(epoch, global_step, global_step + len(train_loader)))
            tbar = tqdm(total=len(train_loader), ncols=100)

            # train_data = {
            #     idx:              tensor, (batchsize,)
            #     points_raw:       tensor, (batchsize, 2048, 6), (xyz, nxyz)
            #     points_src:       tensor, (batchsize, 717, 6),  (xyz, nxyz)
            #     points_ref:       tensor, (batchsize, 717, 6),  (xyz, nxyz)
            #     crop_propotion:   tensor, (batchsize, 2)
            #     transform_gt:     tensor, (batchsize, 3, 4), 前三列是so3, 最后一列是translation
            # }
            loss_train_epoch, loss_tr_epoch, loss_fe_epoch = [], [], []
            for batch_idx, train_data in enumerate(train_loader):
                global_step += 1

                optimizer.zero_grad()

                if global_step == 1:
                    _logger.info('Sample shape: src {}, ref {}'.format(train_data['points_src'].shape,
                                                                       train_data['points_ref'].shape))

                # Forward through neural network
                dict_all_to_device(train_data, _device)

                # pred_transforms (iter, batchsize, 3, 4)
                #
                # endpoints = {
                #     'perm_matrices_init':   tensor          (iter, batchsize, 717, 717)
                #     'perm_matrices':        tensor          (iter, batchsize, 717, 717)
                #     'weighted_ref':         tensor          (iter, batchsize, 717, 3)
                #     'beta':                 np.ndarray      (iter, batchsize, )
                #     'alpha':                np.ndarray      (iter, batchsize, )
                # }
                #
                #
                # feat_src_cluster = {
                #     'xyz':    tensor  (batchsize, 717, 1, 3)
                #     'dxyz':   tensor  (batchsize, 717, 64, 3)
                #     'ppf':    tensor  (batchsize, 717, 64, 4/5)     # batchsize, 每个点云717个点, 每个点取周围64个点, 每个ppf特征4维或5维
                # }
                #
                # feat_src_cluster   tensor    (batchsize, 717, 96)      # 96由参数_args.feat_dim确定
                #
                # feat_ref_cluster   tensor    (batchsize, 717, 96)      # 96由参数_args.feat_dim确定

                # pred_transform = (iter, batchsize, 3, 4)
                pred_transforms, endpoints = model(train_data, _args.num_train_reg_iter, _args.noise_type)  # Use less iter during training

                if _args.feat_dumpdir is not None:

                    cloud_batch = to_numpy(train_data['points_src'])  # (8, 717, 6)
                    feat_src_cluster = endpoints['feat_src_cluster']
                    feat_ref_cluster = endpoints['feat_ref_cluster']
                    feat_src_cluster = to_numpy(feat_src_cluster)  # (8, 717, 96)
                    feat_ref_cluster = to_numpy(feat_ref_cluster)  # (8, 717, 96)

                    for sample_idx in range(_args.train_batch_size):
                        dump_dir_epoch = os.path.join(_args.feat_dumpdir, 'epoch_{}/'.format(epoch))
                        dump_dir_batch = os.path.join(dump_dir_epoch, 'batch_{}/'.format(batch_idx))
                        dump_dir_sample = os.path.join(dump_dir_batch, 'sample_{}/'.format(sample_idx))
                        sample_cloud, sample_feat_src_cluster, sample_feat_ref_cluster = \
                            cloud_batch[sample_idx], feat_src_cluster[sample_idx], feat_ref_cluster[sample_idx]
                        if not os.path.exists(dump_dir_sample):
                            os.makedirs(dump_dir_sample)
                        # sample_feat_src_cluster = {
                        #     'xyz' : to_numpy(feat_src_cluster['xyz'][0]),
                        #     'dxyz': to_numpy(feat_src_cluster['dxyz'][0]),
                        #     'ppf' : to_numpy(feat_src_cluster['ppf'][0]),
                        # }

                        with open(os.path.join(dump_dir_sample, 'feat_in.txt'), 'wb') as f:
                            pickle.dump(sample_feat_src_cluster, f)
                        with open(os.path.join(dump_dir_sample, 'feat_ref_cluster.txt'), 'wb') as f:
                            pickle.dump(sample_feat_ref_cluster, f)

                        generate_rectangular_obj(os.path.join(dump_dir_sample, 'src_cloud.obj'),
                                                 points=sample_cloud[:, :3],
                                                 points_normals=sample_cloud[:, 3:])

                # # Compute loss, and optimize

                train_losses = compute_losses(train_data, pred_transforms, endpoints,
                                              loss_type=_args.loss_type, reduction='mean',
                                              lamb=_args.loss_lambda, omega_1=_args.loss_omega_1,
                                              omega_2=_args.loss_omega_2, model=model)
                loss_train_epoch.append(to_numpy(train_losses['total']))
                loss_fe_epoch.append(to_numpy(train_losses['fe']))
                loss_tr_epoch.append(to_numpy(train_losses['tr']))
                if _args.debug:
                    with TorchDebugger():
                        train_losses['total'].backward()
                else:
                    train_losses['total'].backward()
                optimizer.step()
                #
                tbar.set_description('Loss total:{:.4f}, Loss tr:{:.4f}, Loss fe:{:.4f}'.
                                     format(train_losses['total'], train_losses['tr'], train_losses['fe']))
                tbar.update(1)

                if global_step % _args.summary_every == 0:  # Save tensorboard logs_pydnet
                    save_summaries(train_writer, data=train_data, predicted=pred_transforms, endpoints=endpoints,
                                   losses=train_losses, step=global_step)

                if global_step % _args.validate_every == 0:  # Validation loop. Also saves checkpoints
                    model.eval()
                    # val_score = -chamfer distance
                    _, val_loss, metrics_batch = validate(val_loader, model, val_writer, global_step)

                    # cd 1e-6, rot mean 1e-2  translation mean 1e-4
                    val_score = - (metrics_batch['chamfer_dist'] + (1e-4) * metrics_batch['err_r_deg_mean'] + (1e-2) * metrics_batch['err_t_mean'])

                    err_r_deg_mean.append(metrics_batch['err_r_deg_mean'])
                    err_r_deg_rmse.append(metrics_batch['err_r_deg_rmse'])
                    err_t_mean.append(metrics_batch['err_t_mean'])
                    err_t_rmse.append(metrics_batch['err_t_rmse'])
                    chamfer_distance.append(metrics_batch['chamfer_dist'])
                    loss_val_all.append(val_loss)

                    saver.save(model, optimizer, step=global_step, score=val_score)
                    model.train()

            loss_train_all.append(np.mean(loss_train_epoch))
            loss_fe_all.append(np.mean(loss_fe_epoch))
            loss_tr_all.append(np.mean(loss_tr_epoch))
            tbar.close()
    except Exception as e:
        _logger.info('Error encountered ending training at step {}'.format(global_step))
        _logger.error('{}'.format(e))
    finally:
    # if True:
        _logger.info('Ending training. Number of steps = {}.'.format(global_step))

        plot_loss_curve(loss_train=loss_train_all, loss_val=loss_val_all, loss_fe=loss_fe_all,
                        loss_cd=chamfer_distance, loss_tr=loss_tr_all, loss_tr_type=_args.loss_type,
                        erm=err_r_deg_mean, err=err_r_deg_rmse, etm=err_t_mean, etr=err_t_rmse)


def plot_single_curve(data,
                      save_name,
                      title,
                      xlabel=None,
                      ylabel=None,
                      label=None):
    len_cd = len(data)
    x = range(0, len_cd)
    plt.plot(x, data, color='r', label=label)
    plt.xlabel(xlabel), plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join(_log_path, '{}.png'.format(save_name)))
    np.savetxt(os.path.join(_log_path, '{}.txt'.format(save_name)), np.array(data))
    plt.cla()


def plot_loss_curve(loss_tr_type: str = None,
                    *args,
                    **kwargs):

    _logger.info('Saving loss curve in log directory')
    loss_train = kwargs.get('loss_train', None)
    loss_val = kwargs.get('loss_val', None)
    loss_fe = kwargs.get('loss_fe', None)
    loss_tr = kwargs.get('loss_tr', None)
    loss_cd = kwargs.get('loss_cd', None)
    erm = kwargs.get('erm', None)
    err = kwargs.get('err', None)
    etm = kwargs.get('etm', None)
    etr = kwargs.get('etr', None)

    assert loss_train is not None, _logger.error('No train loss in {}'.format(kwargs.keys()))

    if loss_train is not None and loss_val is not None:
        len_train, len_eval = len(loss_train), len(loss_val)
        x = range(0, len_train, 1)
        plt.plot(x, loss_train, color='r', label='train loss')

        x = range(0, len_eval * (len_train // len_eval), len_train // len_eval)
        plt.plot(x, loss_val, color='b', label='val loss')

        np.savetxt(os.path.join(_log_path, 'loss_train.txt'), np.array(loss_train))
        np.savetxt(os.path.join(_log_path, 'loss_val.txt'), np.array(loss_val))

    if loss_train is not None and loss_fe is not None:
        len_fe = len(loss_fe)
        x = range(0, len_fe * (len_train // len_fe), len_train // len_fe)
        plt.plot(x, loss_fe, color='g', linestyle='dashed', label='loss fe {}'.format(loss_tr_type))
        np.savetxt(os.path.join(_log_path, 'loss_fe.txt'), np.array(loss_fe))
    if loss_train is not None and loss_tr is not None:
        len_tr = len(loss_tr)
        x = range(0, len_tr * (len_train // len_tr), len_train // len_tr)
        plt.plot(x, loss_tr, color='m', linestyle='dashed', label='loss tr')
        np.savetxt(os.path.join(_log_path, 'loss_tr.txt'), np.array(loss_tr))


    plt.xlabel('epoch'), plt.ylabel('loss')
    plt.legend()
    plt.title('train & val loss')
    plt.savefig(os.path.join(_log_path, 'loss.png'))
    plt.cla()


    if loss_cd is not None:
        plot_single_curve(data=loss_cd, save_name='chamfer_distance', title='chamfer distance',
                          xlabel='validate epoch', ylabel='chamfer distance', label='chamfer distance')
    if erm is not None:
        plot_single_curve(data=erm, save_name='err_rotation_mean', title='err rotation mean',
                          xlabel='validate epoch', ylabel='err rotation mean', label='err rotation mean')

    if err is not None:
        plot_single_curve(data=err, save_name='err_rotation_rmse', title='err rotation rmse',
                          xlabel='validate epoch', ylabel='err rotation rmse', label='err rotation rmse')

    if etm is not None:
        plot_single_curve(data=etm, save_name='err_translation_mean', title='err translation mean',
                          xlabel='validate epoch', ylabel='err translation mean', label='err translation mean')

    if etr is not None:
        plot_single_curve(data=etr, save_name='err_translation_rmse', title='err translation rmse',
                          xlabel='validate epoch', ylabel='err translation rmse', label='err translation rmse')



def main():
    train_set, val_set = get_train_datasets(_args)
    run(train_set, val_set)


if __name__ == '__main__':
    main()
