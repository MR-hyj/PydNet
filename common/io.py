# -*- encoding: utf-8 -*-

# Author:   Hengyu Jiang
# Time:     2022/1/6 17:16
# Email:    hengyujiang.njust.edu.cn
# Project:  PydNet
# File:     io.py
# Product:  PyCharm
# Desc:


import os
import logging
import numpy as np
from vispy import io
from plyfile import PlyData
from typing import Union, Tuple, List

_logger = logging.getLogger(__name__)


def points_normals_from_ply(path_ply: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    load points and their normals from a ply file\n
    :param path_ply:
    :returns:           Tuple[np.ndarray, np.ndarray]
    """
    _logger.info('Loading points, normals from {}'.format(path_ply))
    data = PlyData.read(path_ply)['vertex']
    length = data['x'].shape[0]
    x, y, z = data['x'].reshape((length, 1)), data['y'].reshape((length, 1)), data['z'].reshape((length, 1))
    nx, ny, nz = data['nx'].reshape((length, 1)), data['ny'].reshape((length, 1)), data['nz'].reshape((length, 1))
    points = np.hstack([x, y, z]).astype(np.float)
    normals = np.hstack([nx, ny, nz]).astype(np.float)
    _logger.info('Loaded successfully')
    return points, normals


def points_meshes_from_ply(path_ply: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    load points and meshes from a ply file\n
    :param path_ply:
    :return:
    """
    _logger.info('Loading points, meshes from {}'.format(path_ply))
    file = PlyData.read(path_ply)
    data_points = file['vertex']
    data_faces = file['face']
    length = data_points['x'].shape[0]
    x, y, z = data_points['x'].reshape((length, 1)), data_points['y'].reshape((length, 1)), data_points['z'].reshape(
        (length, 1))
    points = np.hstack([x, y, z]).astype(np.float)
    meshes = np.array([face for face in data_faces.data['vertex_indices']])
    return points, meshes


def points_from_obj(path_obj: str) -> np.ndarray:
    """
    load points, their normals and meshes from an obj file\n
    :param path_obj:
    :return:            Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    _logger.info('Loading points, normals from {}'.format(path_obj))
    vertexes, faces, normals, nothing = io.read_mesh(path_obj)
    return np.array(vertexes).astype(np.float)


def planes_bounds_from_obj(path_obj: str) -> np.ndarray:
    """
    load the bounding points of meshes from an obj file\n

    BE NOTICED: This function is abandoned for some latent risks. Use it carefully.
    :param path_obj:
    :returns:                       np.ndarray
    """
    _logger.info('Loading planes bounds from {}'.format(path_obj))

    verts, faces, normals, nothing = io.read_mesh(path_obj)
    if 4 == faces.shape[1]:
        return verts[faces]
    else:
        lens = faces.shape[0]
        extend = faces[:, 2]
        extend = extend[(np.array(range(lens)) + 1) % lens]
        faces = np.hstack([faces, np.reshape(extend, (lens, 1))])
        sample = np.array(range(0, lens, 2))
        return verts[faces[sample]]


def generate_rectangular_obj(path_obj: str,
                             points: Union[np.ndarray, List] = None,
                             points_normals: Union[np.ndarray, List] = None,
                             points_colors: Union[np.ndarray, List] = None,
                             points_intensities: Union[np.ndarray, List] = None,
                             planes_bounds: Union[np.ndarray, List] = None,
                             planes_normals: Union[np.ndarray, List] = None,
                             use_texture: bool = False,
                             as_mesh: bool = False,
                             show_log: bool = False) -> None:
    """
    generate an obj file with rectangular boundings.\n
    :param points_normals:              normals of points, points_normals.shape == (Nv, 3), where Nv is the number of points.
    :param planes_bounds:               bounding points of planes. each plane shall contain exactly 4 points.
                                        planes_bounds.shape == (Np, 4, 3), where Np is the number of the planes.
    :param path_obj:                    the generated obj file.
    :param planes_normals:              normals of planes, planes_normals.shape == (Np, 3), where Np is the number of the planes.
    :param points:                      points.shape == (Nv, 3)
    :param points_colors:
    :param points_intensities:          points_intensities.shape == (Nv, )
    :param use_texture:                 whether to use texture
    :param show_log:                    whether to show log
    :param as_mesh:                     whether to save the obj file as mesh. If set False, the obj file contains only points
    :return:                            None
    """

    writer = create_file_writer(path_obj)

    # configuration information
    writer.write('# The units used in this file are meters.\n')
    if use_texture:
        writer.write('mtllib ./planes.mtl\n')

    # boundings of planes
    if planes_bounds is not None:
        if not isinstance(planes_bounds, np.ndarray):
            planes_bounds = np.array(planes_bounds)
        assert len(planes_bounds.shape) == 3, _logger.error(
            f'Wrong shape of param planes_bounds.shape={planes_bounds.shape}')
        num_faces = len(planes_bounds)
        for i in range(num_faces):
            bound = planes_bounds[i]
            out_str1 = 'v {} {} {}'.format(bound[0][0], bound[0][1], bound[0][2])
            out_str2 = 'v {} {} {}'.format(bound[1][0], bound[1][1], bound[1][2])
            out_str3 = 'v {} {} {}'.format(bound[2][0], bound[2][1], bound[2][2])
            out_str4 = 'v {} {} {}'.format(bound[3][0], bound[3][1], bound[3][2])
            writer.write(out_str1 + '\n')
            writer.write(out_str2 + '\n')
            writer.write(out_str3 + '\n')
            writer.write(out_str4 + '\n')

    # points
    if points is not None:
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        assert len(points.shape) == 2, _logger.error(f'Wrong shape of param points.shape={points.shape}')
        num_points = len(points)
        for i in range(num_points):
            point = points[i]
            out_str = 'v {} {} {}'.format(point[0], point[1], point[2])
            if points_intensities is not None:
                out_str += ' {}'.format(points_intensities[i])
            if points_colors is not None:
                out_str += ' {} {} {}'.format(points_colors[i][0], points_colors[i][1], points_colors[i][2])
            writer.write(out_str + '\n')

    # normals of the points
    if points_normals is not None:
        if not isinstance(points_normals, np.ndarray):
            points_normals = np.array(points_normals)
        assert len(points_normals.shape) == 2, _logger.error(
            f'Wrong shape of param points_normals.shape={points_normals.shape}')
        num_normals = len(points_normals)
        for i in range(num_normals):
            normal = points_normals[i]
            out_str1 = 'vn {} {} {}'.format(normal[0], normal[1], normal[2])
            writer.write(out_str1 + '\n')

    # normals of the meshes
    if planes_normals is not None:
        if not isinstance(planes_normals, np.ndarray):
            planes_normals = np.array(planes_normals)
        assert len(planes_normals.shape) == 2, \
            _logger.error(f'Wrong shape of param planes_normals={planes_normals.shape}')
        num_normals = len(planes_normals)
        for i in range(num_normals):
            normal = planes_normals[i]
            out_str1 = 'vn {} {} {}'.format(normal[0], normal[1], normal[2])
            writer.write(out_str1 + '\n')

    if as_mesh:  # if set True, this obj would contain meshes, rather than points.

        assert planes_bounds is not None, _logger.error('Param  {planes_bounds} is not included in param')
        if planes_normals is None:
            _logger.warning('Generating rectangle obj file without plane normals.')
        else:
            assert planes_normals is not None, _logger.error('Param {planes_normals} is not included in param')
            assert planes_bounds.shape[0] == planes_normals.shape[0], \
                _logger.error(f'Planes_bounds.shape={planes_bounds.shape} and '
                              f'planes_normals.shape={planes_normals.shape} are not the same')

        num_faces = len(planes_bounds)
        textures = ['cyan', 'gold', 'orange', 'tomato', 'fuchsia', 'blueviolet']
        for i in range(0, num_faces):
            if use_texture:
                out_str1 = 'usemtl {}'.format(textures[i % len(textures)])
                writer.write(out_str1 + '\n')
            if planes_normals is None:
                out_str2 = 'f {} {} {} {}'.format(i * 4 + 1, i * 4 + 2, i * 4 + 3, i * 4 + 4)
            else:
                out_str2 = 'f {}//{} {}//{} {}//{} {}//{}' \
                    .format(i * 4 + 1, i + 1, i * 4 + 2, i + 1, i * 4 + 3, i + 1, i * 4 + 4, i + 1)
            writer.write(out_str2 + '\n')

    writer.close()


def generate_triangular_obj(path_obj: str,
                            points: Union[np.ndarray, List] = None,
                            points_normals: Union[np.ndarray, List] = None,
                            points_colors: Union[np.ndarray, List] = None,
                            points_intensities: Union[np.ndarray, List] = None,
                            planes_bounds: Union[np.ndarray, List] = None,
                            planes_normals: Union[np.ndarray, List] = None,
                            use_texture: bool = False,
                            as_mesh: bool = False,
                            show_log: bool = False) -> None:
    """
    generate an obj file with triangular boundings.\n
    :param points_normals:              normals of points, points_normals.shape == (Nv, 3), where Nv is the number of points.
    :param planes_bounds:               bounding points of planes. each plane shall contain exactly 3 points.
                                        planes_bounds.shape == (Np, 3, 3), where Np is the number of the planes.
    :param path_obj:                    the generated obj file.
    :param planes_normals:              normals of planes, planes_normals.shape == (Np, 3), where Np is the number of the planes.
    :param points:                      points.shape == (Nv, 3)
    :param points_colors:
    :param points_intensities:          points_intensities.shape == (Nv, )
    :param use_texture:                 whether to use texture
    :param show_log:                    whether to show log
    :param as_mesh:                     whether to save the obj file as mesh. If set False, the obj file contains only points
    :return:                            None
    """

    writer = create_file_writer(path_obj)

    # configuration information
    writer.write('# The units used in this file are meters.\n')
    if use_texture:
        writer.write('mtllib ./planes.mtl\n')

    # boundings of meshes
    if planes_bounds is not None:
        if not isinstance(planes_bounds, np.ndarray):
            planes_bounds = np.array(planes_bounds)
        assert len(planes_bounds.shape) == 3, _logger.error(
            f'Wrong shape of param planes_bounds.shape={planes_bounds.shape}')
        num_faces = len(planes_bounds)
        for i in range(num_faces):
            bound = planes_bounds[i]
            out_str1 = 'v {} {} {}'.format(bound[0][0], bound[0][1], bound[0][2])
            out_str2 = 'v {} {} {}'.format(bound[1][0], bound[1][1], bound[1][2])
            out_str3 = 'v {} {} {}'.format(bound[2][0], bound[2][1], bound[2][2])
            writer.write(out_str1 + '\n')
            writer.write(out_str2 + '\n')
            writer.write(out_str3 + '\n')

    # points
    if points is not None:
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        assert len(points.shape) == 2, _logger.error(f'Wrong shape of param points.shape={points.shape}')
        num_points = len(points)
        for i in range(num_points):
            point = points[i]
            out_str = 'v {} {} {}'.format(point[0], point[1], point[2])
            if points_intensities is not None:
                out_str += ' {}'.format(points_intensities[i])
            if points_colors is not None:
                out_str += ' {} {} {}'.format(points_colors[i][0], points_colors[i][1], points_colors[i][2])
            writer.write(out_str + '\n')

    # normals of the points
    if points_normals is not None:
        if not isinstance(points_normals, np.ndarray):
            points_normals = np.array(points_normals)
        assert len(points_normals.shape) == 2, _logger.error(
            f'Wrong shape of param points_normals.shape={points_normals.shape}')
        num_normals = len(points_normals)
        for i in range(num_normals):
            normal = points_normals[i]
            out_str1 = 'vn {} {} {}'.format(normal[0], normal[1], normal[2])
            writer.write(out_str1 + '\n')

    # normals of the meshes.
    if planes_normals is not None:
        if not isinstance(planes_normals, np.ndarray):
            planes_normals = np.array(planes_normals)
        assert len(planes_normals.shape) == 2, \
            _logger.error(f'Wrong shape of param planes_normals={planes_normals.shape}')
        num_normals = len(planes_normals)
        for i in range(num_normals):
            normal = planes_normals[i]
            out_str1 = 'vn {} {} {}'.format(normal[0], normal[1], normal[2])
            writer.write(out_str1 + '\n')

    if as_mesh:  # if set True, this obj would contain meshes, rather than points.
        assert planes_bounds is not None, _logger.error('Param  {planes_bounds} is not included in param')
        if planes_normals is None:
            _logger.warning('Generating triangular obj file without plane normals.')
        else:
            assert planes_normals is not None, _logger.error('Param {planes_normals} is not included in param')
            assert planes_bounds.shape[0] == planes_normals.shape[0], \
                _logger.error(f'Planes_bounds.shape={planes_bounds.shape} and '
                              f'planes_normals.shape={planes_normals.shape} are not the same')
        num_faces = len(planes_bounds)
        textures = ['cyan', 'gold', 'orange', 'tomato', 'fuchsia', 'blueviolet']
        for i in range(0, num_faces):
            if use_texture:
                out_str1 = 'usemtl {}'.format(textures[i % len(textures)])
                writer.write(out_str1 + '\n')
            if planes_normals is None:
                out_str2 = 'f {} {} {}'.format(i * 4 + 1, i * 4 + 2, i * 4 + 3)
            else:
                out_str2 = 'f {}//{} {}//{} {}//{}' \
                    .format(i * 4 + 1, i + 1, i * 4 + 2, i + 1, i * 4 + 3, i + 1)
            writer.write(out_str2 + '\n')

    writer.close()


def create_file_writer(path_file: str,
                       mode: str = 'w',
                       show_log: bool = False):
    """
    :param path_file:
    :param mode:
    :param show_log:
    :return:            file stream
    """
    if os.path.exists(path_file):
        if show_log:
            _logger.info('File {} exists, overwrite'.format(path_file, mode))
    else:
        if show_log:
            _logger.info('Create writer, path={}'.format(path_file))
    return open(path_file, mode=mode)
