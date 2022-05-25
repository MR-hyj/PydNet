# -*- encoding: utf-8 -*-

# Author:   Hengyu Jiang
# Time:     2022/1/5 17:00
# Email:    hengyujiang.njust.edu.cn
# Project:  PydNet
# File:     feature_net.py
# Product:  PyCharm
# Desc:     Feature Extraction  network


import logging
import numpy as np
import torch
import torch.nn as nn
from modules.pointnet_util import sample_and_group_multi


_raw_features_sizes = {'xyz': 3, 'dxyz': 3, 'pmd': 1}
_raw_features_order = {'xyz': 0, 'dxyz': 1, 'pmd': 2}


def get_prepool(in_dim, out_dim):
    """Shared FC part in PointNet before max pooling"""
    net = nn.Sequential(
        nn.Conv2d(in_dim, out_dim // 2, 1),
        nn.GroupNorm(8, out_dim // 2),
        # nn.InstanceNorm2d(num_features=in_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim // 2, out_dim // 2, 1),
        nn.GroupNorm(8, out_dim // 2),
        # nn.InstanceNorm2d(num_features=in_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim // 2, out_dim, 1),
        nn.GroupNorm(8, out_dim),
        # nn.InstanceNorm2d(num_features=in_dim),
        nn.ReLU(),
    )
    return net


def get_postpool(in_dim, out_dim):
    """Linear layers in PointNet after max pooling

    Args:
        in_dim:  Number of input channels
        out_dim: Number of output channels. Typically smaller than in_dim

    """
    net = nn.Sequential(
        nn.Conv1d(in_dim, in_dim, 1),
        nn.GroupNorm(8, in_dim),
        # nn.InstanceNorm1d(num_features=in_dim),
        nn.ReLU(),
        nn.Conv1d(in_dim, out_dim, 1),
        nn.GroupNorm(8, out_dim),
        # nn.InstanceNorm1d(num_features=in_dim),
        nn.ReLU(),
        nn.Conv1d(out_dim, out_dim, 1),
    )

    return net


class FeatExtractionEarlyFusion(nn.Module):
    """Feature extraction Module that extracts hybrid features"""
    def __init__(self, features, feature_dim, radius, num_neighbors):
        """

        :param features:            which features to use, default: ['pymf', 'dxyz', 'xyz']
        :param feature_dim:         feature dimension (to compute distance), default 96
        :param radius:              Neighborhood radius for computing pointnet features, default 0.3
        :param num_neighbors:       Max num of neighbors to use, default 64
        """
        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.info('Using early fusion, feature dim = {}'.format(feature_dim))
        self.radius = radius
        self.n_sample = num_neighbors

        self.features = sorted(features, key=lambda f: _raw_features_order[f])
        self._logger.info('Feature extraction using features {}'.format(', '.join(self.features)))

        # Layers
        # number of channels after concat, 计算features中所有特征的总维度和
        raw_dim = np.sum([_raw_features_sizes[f] for f in self.features])
        self.prepool = get_prepool(raw_dim, feature_dim * 2)
        self.postpool = get_postpool(feature_dim * 2, feature_dim)

    def forward(self, xyz, normals):
        """Forward pass of the feature extraction network

        Args:
            xyz: (B, N, 3)
            normals: (B, N, 3)

        Returns:
            cluster features (B, N, C)

        """
        # 提取特征, (xyz, {dxyz}, {pmd})
        # xyz:  (B, npoint, 3)
        # dxyz: (B, npoint, nsample, 3)
        # pmd:  (B, npoint, 3)
        features = sample_and_group_multi(-1, self.radius, self.n_sample, xyz, normals)
        features['xyz'] = features['xyz'][:, :, None, :]

        # Gate and concat
        concat = []
        for i in range(len(self.features)):
            f = self.features[i]
            # f in ['xyz', 'dxyz', 'pymf']
            # xyz:  (B, npoint, 3)
            # dxyz: (B, npoint, nsample, 3)
            # pmd:  (B, npoint, 3)
            expanded = (features[f]).expand(-1, -1, self.n_sample, -1)
            concat.append(expanded)
        fused_input_feat = torch.cat(concat, -1)            # (B, N, n_sample, 9)

        # Prepool_FC, pool, postpool-FC
        new_feat = fused_input_feat.permute(0, 3, 2, 1)     # [B, 9, n_sample, N]
        new_feat = self.prepool(new_feat)

        pooled_feat = torch.max(new_feat, 2)[0]  # Max pooling (B, C, N)

        post_feat = self.postpool(pooled_feat)  # Post pooling dense layers
        cluster_feat = post_feat.permute(0, 2, 1)
        cluster_feat = cluster_feat / torch.norm(cluster_feat, dim=-1, keepdim=True)

        return cluster_feat, features  # (B, N, C)
