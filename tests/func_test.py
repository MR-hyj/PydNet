# -*- encoding: utf-8 -*-

# Author:   Hengyu Jiang
# Time:     2022/5/21 16:04
# Email:    hengyujiang.njust.edu.cn
# Project:  PydNet
# File:     func_test.py
# Product:  PyCharm
# Desc:

import numpy as np
import torch
from common.mytorch import to_numpy

def distance_point_to_points(point: np.ndarray,
                             points: np.ndarray):
    """
    calculate the distance of a point and a set of points
    @param point:
    @param points:
    """

    # ignore the normals if any

    point, points = point.astype(np.float)[:3], points.astype(np.float)[:, :3]
    return np.linalg.norm(points-point, axis=-1)


def distance_points_to_points(points_a: np.ndarray,
                             points_b: np.ndarray) -> np.ndarray:
    """
    calculate the distance of each point in points_1 and each point in points_2
    @param points_a: a set of points, marked as A, with shape (na, 3, )
    @param points_b: a set of points, marked as B, with shape (nb, 3, )
    @return: the distance of each point in A and each point in B, with shape (na, nb)
    """
    distance_a_to_b = np.array([])
    for point in points_a:
        distance_each = distance_point_to_points(point, points_b)
        distance_a_to_b = np.concatenate([distance_a_to_b, distance_each])
    return distance_a_to_b.reshape(points_a.shape[0], points_b.shape[0])

if __name__ == '__main__':
    point = np.array([[1, 2, 3], [4, 2, 1]])
    points = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])

    distance = distance_points_to_points(point, points)
    max = distance.max(axis=1)
    print(distance)
    print()
    print(max)
    print()
    print((distance.T/max).T)