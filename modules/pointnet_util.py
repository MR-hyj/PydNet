# -*- encoding: utf-8 -*-

# Author:   Hengyu Jiang
# Time:     2022/1/5 17:00
# Email:    hengyujiang.njust.edu.cn
# Project:  PydNet
# File:     pointnet_util.py
# Product:  PyCharm
# Desc:     Utilities for PointNet related functions


"""


Modified from:
    Pytorch Implementation of PointNet and PointNet++
    https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""

import torch
import numpy as np
from common.mytorch import to_numpy
import logging

_logger = logging.getLogger(__file__)

def angle_difference(src, dst):
    """Calculate angle between each pair of vectors.
    Assumes points are l2-normalized to unit length.

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = torch.matmul(src, dst.permute(0, 2, 1))
    dist = torch.acos(dist)

    return dist


def square_distance(src, dst):
    """Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
             = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1)[:, :, None]
    dist += torch.sum(dst ** 2, dim=-1)[:, None, :]
    return dist


def index_points(points, idx):
    """Array indexing, i.e. retrieves relevant points based on indices

    Args:
        points: input points data_loader, [B, N, C]
        idx: sample index data_loader, [B, S]. S can be 2 dimensional
    Returns:
        new_points:, indexed points data_loader, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """Iterative farthest point sampling

    Args:
        xyz: pointcloud data_loader, [B, N, C]
        npoint: number of samples
    Returns:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz, itself_indices=None):
    """ Grouping layer in PointNet++.

    Args:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, (B, N, C)
        new_xyz: query points, (B, S, C)
        itself_indices (Optional): Indices of new_xyz into xyz (B, S).
          Used to try and prevent grouping the point itself into the neighborhood.
          If there is insufficient points in the neighborhood, or if left is none, the resulting cluster will
          still contain the center point.
    Returns:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])  # (B, S, N)
    sqrdists = square_distance(new_xyz, xyz)

    if itself_indices is not None:
        # Remove indices of the center points so that it will not be chosen
        batch_indices = torch.arange(B, dtype=torch.long).to(device)[:, None].repeat(1, S)  # (B, S)
        row_indices = torch.arange(S, dtype=torch.long).to(device)[None, :].repeat(B, 1)  # (B, S)
        group_idx[batch_indices, row_indices, itself_indices] = N

    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    if itself_indices is not None:
        group_first = itself_indices[:, :, None].repeat([1, 1, nsample])
    else:
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint: int, radius: float, nsample: int, xyz: torch.Tensor, points: torch.Tensor,
                     returnfps: bool = False):
    """
    Args:
        npoint (int): Set to negative to compute for all points
        radius:
        nsample:
        xyz: input points position data_loader, [B, N, C]
        points: input points data_loader, [B, N, D]
        returnfps (bool) Whether to return furthest point indices
    Returns:
        new_xyz: sampled points position data_loader, [B, 1, C]
        new_points: sampled points data_loader, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape

    if npoint > 0:
        S = npoint
        fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
        new_xyz = index_points(xyz, fps_idx)
    else:
        S = xyz.shape[1]
        fps_idx = torch.arange(0, xyz.shape[1])[None, ...].repeat(xyz.shape[0], 1)
        new_xyz = xyz

    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # (B, N, nsample)
    grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, C)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def angle(v1: torch.Tensor, v2: torch.Tensor):
    """Compute angle between 2 vectors

    For robustness, we use the same formulation as in PPFNet, i.e.
        angle(v1, v2) = atan2(cross(v1, v2), dot(v1, v2)).
    This handles the case where one of the vectors is 0.0, since torch.atan2(0.0, 0.0)=0.0

    Args:
        v1: (B, *, 3)
        v2: (B, *, 3)

    Returns:

    """

    cross_prod = torch.stack([v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1],
                              v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2],
                              v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]], dim=-1)
    cross_prod_norm = torch.norm(cross_prod, dim=-1)
    dot_prod = torch.sum(v1 * v2, dim=-1)

    return torch.atan2(cross_prod_norm, dot_prod)


def sample_and_group_multi(npoint: int, radius: float, nsample: int, xyz: torch.Tensor, normals: torch.Tensor,
                           returnfps: bool = False):
    """Sample and group for xyz, dxyz and ppf features

    Args:
        npoint(int): Number of clusters (equivalently, keypoints) to sample.
                     Set to negative to compute for all points
        radius(int): Radius of cluster for computing local features
        nsample: Maximum number of points to consider per cluster
        xyz: XYZ coordinates of the points
        normals: Corresponding normals for the points (required for ppf computation)
        returnfps: Whether to return indices of FPS points and their neighborhood

    Returns:
        Dictionary containing the following fields ['xyz', 'dxyz', 'ppf'].
        If returnfps is True, also returns: grouped_xyz, fps_idx
        ppf contains: angle of nc and Δx(c,i), angle of ni and Δx(c,i), angle of nc and ni, ||Δx(c,i)||^2
    """

    B, N, C = xyz.shape

    if npoint > 0:
        S = npoint
        fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
        new_xyz = index_points(xyz, fps_idx)
        nr = index_points(normals, fps_idx)[:, :, None, :]          # neighbor normals
    else:
        S = xyz.shape[1]
        fps_idx = torch.arange(0, xyz.shape[1])[None, ...].repeat(xyz.shape[0], 1).to(xyz.device)
        new_xyz = xyz
        nr = normals[:, :, None, :]     # neighbor normals

    idx = query_ball_point(radius, nsample, xyz, new_xyz, fps_idx)  # (B, npoint, nsample)
    grouped_xyz = index_points(xyz, idx)                            # (B, npoint, nsample, C), 邻接点
    d = grouped_xyz - new_xyz.view(B, S, 1, C)                      # d = p_r - p_i    (B, npoint, nsample, 3)
    xyz_feat = d  # (B, npoint, nsample, 3)

    ni = index_points(normals, idx)


    # nr_d = angle(nr, d)     # (B, npoint, nsample)
    # ni_d = angle(ni, d)     # (B, npoint, nsample)
    # nr_ni = angle(nr, ni)   # (B, npoint, nsample)
    # d_norm = torch.norm(d, dim=-1)  # (B, npoint, nsample)
    # ppf_feat = torch.stack([nr_d, ni_d, nr_ni, d_norm], dim=-1)  # (B, npoint, 4) 论文公式(8)
    # _logger.debug('nr_d shape: {}, d_norm shape: {}, ppf shape: {}'.format(nr_d.shape, d_norm.shape, ppf_feat.shape))
    #
    # if returnfps:  # 论文公式(6)
    #     return {'xyz': new_xyz, 'dxyz': xyz_feat, 'pmd': ppf_feat}, grouped_xyz, fps_idx
    # else:
    #     return {'xyz': new_xyz, 'dxyz': xyz_feat, 'pmd': ppf_feat}



    # TODO 自定义特征
    # pmd = {'xyz': (B, npoint, 3), 'dxyz': (B, npoint, n_sample, 3), 'pymf': (B, npoint, 4)

    """
    计算三角锥顶点S所在的3个面的法向量, 及其夹角
    设S为顶点, ABC为底面的3个点
    """
    if isinstance(d, np.ndarray):
        SA = torch.from_numpy(d[:, :, 0, :])  # (B, npoint, 3)
        SB = torch.from_numpy(d[:, :, 1, :])  # (B, npoint, 3)
        SC = torch.from_numpy(d[:, :, 2, :])  # (B, npoint, 3)

    elif isinstance(d, torch.Tensor):
        SA = d[:, :, 0, :]  # (B, npoint, 3)
        SB = d[:, :, 1, :]  # (B, npoint, 3)
        SC = d[:, :, 2, :]  # (B, npoint, 3)
    else:
        _logger.error('Argument type not implemented yet')
        raise NotImplementedError

    # 3个侧面的法向量, 同时向内或者向外
    n_SAB = torch.cross(SA, SB)
    n_SCA = torch.cross(SC, SA)
    n_SBC = torch.cross(SB, SC)

    # 每两个面的法向量的夹角
    angle_SA = angle(n_SAB, n_SCA)  # (B, npoint)
    angle_SB = angle(n_SAB, n_SBC)  # (B, npoint)
    angle_SC = angle(n_SCA, n_SBC)  # (B, npoint)
    # _logger.debug('angle SA shape: {}'.format(angle_SA.shape))

    angle_SA = angle_SA.reshape(B, N, 1)
    angle_SB = angle_SB.reshape(B, N, 1)
    angle_SC = angle_SC.reshape(B, N, 1)
    # _logger.debug('angle SA reshape shape: {}'.format(angle_SA.shape))

    angle_SA = angle_SA.repeat(1, 1, nsample)     # (B, npoint, 3)
    angle_SB = angle_SB.repeat(1, 1, nsample)     # (B, npoint, 3)
    angle_SC = angle_SC.repeat(1, 1, nsample)     # (B, npoint, 3)
    nr_ni = angle(nr, ni)                   # (B, npoint, nsample)
    # _logger.debug('angle SA repeat shape: {}'.format(angle_SA.shape))
    #
    d_norm = torch.norm(d, dim=-1)
    # xyz:  (B, npoint, 3)
    # dxyz: (B, npoint, nsample, 3)
    # pmd:  (B, npoint, 5, nsample)

    pmd = torch.stack([angle_SA, angle_SB, angle_SC, nr_ni, d_norm], dim=-1)
    _logger.debug('angle SA shape: {}, d_norm shape: {}, pmd shape: {}'.format(angle_SA.shape, d_norm.shape, pmd.shape))
    return {'xyz': new_xyz, 'dxyz': xyz_feat, 'pmd': pmd}




