# -*- encoding: utf-8 -*-

# Author:   Hengyu Jiang
# Time:     2022/1/5 17:01
# Email:    hengyujiang.njust.edu.cn
# Project:  PydNet
# File:     parameter_net.py
# Product:  PyCharm
# Desc:     Parameter Prediction network


import torch
import logging
import numpy as np
from torch import nn as nn
from torch.nn import functional as F


class ParameterPredictionNet(nn.Module):
    def __init__(self, weights_dim, norm_type='InstanceNorm'):
        """PointNet based Parameter prediction network

        Args:
            weights_dim: Number of weights to predict (excluding beta), should be something like
                         [3], or [64, 3], for 3 types of features
        """

        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)
        self.weights_dim = weights_dim
        self.norm_type = norm_type

        # Pointnet

        if 'InstanceNorm' == self.norm_type:
            self.prepool = nn.Sequential(
                nn.Conv1d(4, 64, 1),
                nn.InstanceNorm1d(num_features=64),
                nn.ReLU(),

                nn.Conv1d(64, 64, 1),
                nn.InstanceNorm1d(num_features=64),
                nn.ReLU(),

                nn.Conv1d(64, 64, 1),
                nn.InstanceNorm1d(num_features=64),
                nn.ReLU(),

                nn.Conv1d(64, 128, 1),
                nn.InstanceNorm1d(num_features=128),
                nn.ReLU(),

                nn.Conv1d(128, 1024, 1),
                nn.InstanceNorm1d(num_features=1024),
                nn.ReLU(),
            )
        else:
            self.prepool = nn.Sequential(
                nn.Conv1d(4, 64, 1),
                nn.GroupNorm(8, 64),
                nn.ReLU(),

                nn.Conv1d(64, 64, 1),
                nn.GroupNorm(8, 64),
                nn.ReLU(),

                nn.Conv1d(64, 64, 1),
                nn.GroupNorm(8, 64),
                nn.ReLU(),

                nn.Conv1d(64, 128, 1),
                nn.GroupNorm(8, 128),
                nn.ReLU(),

                nn.Conv1d(128, 1024, 1),
                nn.GroupNorm(16, 1024),
                nn.ReLU(),
            )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.postpool = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(num_features=512),
            nn.GroupNorm(16, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            # nn.BatchNorm1d(num_features=256),
            nn.GroupNorm(16, 256),
            nn.ReLU(),

            nn.Linear(256, 2 + np.prod(weights_dim)),
        )

        self._logger.info('Predicting weights with dim {}.'.format(self.weights_dim))

    def forward(self, x):
        """ Returns alpha, beta, and gating_weights (if needed)\n
        论文章节5.2. Parameter Prediction Network\n

        Args:
            x: List containing two point clouds, x[0] = src (B, J, 3), x[1] = ref (B, K, 3)

        Returns:
            beta, alpha, weightings
        """

        src_padded = F.pad(x[0], (0, 1), mode='constant', value=0)
        ref_padded = F.pad(x[1], (0, 1), mode='constant', value=1)
        concatenated = torch.cat([src_padded, ref_padded], dim=1)

        prepool_feat = self.prepool(concatenated.permute(0, 2, 1))
        pooled = torch.flatten(self.pooling(prepool_feat), start_dim=-2)
        raw_weights = self.postpool(pooled)

        beta = F.softplus(raw_weights[:, 0])
        alpha = F.softplus(raw_weights[:, 1])

        return beta, alpha


class ParameterPredictionNetConstant(nn.Module):
    def __init__(self, weights_dim):
        """Parameter Prediction Network with single alpha/beta as parameter.

        See: Ablation study (Table 4) in paper
        """

        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)

        self.anneal_weights = nn.Parameter(torch.zeros(2 + np.prod(weights_dim)))
        self.weights_dim = weights_dim

        self._logger.info('Predicting weights with dim {}.'.format(self.weights_dim))

    def forward(self, x):
        """Returns beta, gating_weights"""

        batch_size = x[0].shape[0]
        raw_weights = self.anneal_weights
        beta = F.softplus(raw_weights[0].expand(batch_size))
        alpha = F.softplus(raw_weights[1].expand(batch_size))

        return beta, alpha