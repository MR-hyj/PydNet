# -*- encoding: utf-8 -*-

# Author:   Hengyu Jiang
# Time:     2022/1/6 10:58
# Email:    hengyujiang.njust.edu.cn
# Project:  PydNet
# File:     arguments.py
# Product:  PyCharm
# Desc:


"""Common arguments for train and evaluation for PydNet"""
import argparse


def pydnet_arguments():
    """Arguments used for both training and testing"""
    parser = argparse.ArgumentParser(add_help=False)

    # Logging
    parser.add_argument('--logdir', default='./logs_pydnet/', type=str,
                        help='Directory to store logs_pydnet, summaries, checkpoints.')
    parser.add_argument('--log_level', default='INFO', type=str)
    parser.add_argument('--dev', action='store_true', help='If true, will ignore logdir and log to ./logdev instead')
    parser.add_argument('--name', type=str, help='Prefix to add to logging directory')
    parser.add_argument('--debug', action='store_true', help='If set, will enable autograd anomaly detection')
    # settings for input data_loader
    parser.add_argument('-i', '--dataset_path',
                        default='./datasets/modelnet40_ply_hdf5_2048',
                        type=str, metavar='PATH',
                        help='path to the processed dataset. Default: ./datasets/modelnet40_ply_hdf5_2048')
    parser.add_argument('--dataset_type', default='modelnet_hdf',
                        choices=['modelnet_hdf', 'bunny', 'armadillo', 'buddha', 'dragon'],
                        metavar='DATASET', help='dataset type (default: modelnet_hdf)')
    parser.add_argument('--num_points', default=1024, type=int,
                        metavar='N', help='points in point-cloud (default: 1024)')
    parser.add_argument('--noise_type', default='crop', choices=['clean', 'jitter', 'crop'],
                        help='Types of perturbation to consider')
    parser.add_argument('--rot_mag', default=45.0, type=float,
                        metavar='T', help='Maximum magnitude of rotation perturbation (in degrees)')
    parser.add_argument('--trans_mag', default=0.5, type=float,
                        metavar='T', help='Maximum magnitude of translation perturbation')
    parser.add_argument('--partial', default=[0.7, 0.7], nargs='+', type=float,
                        help='Approximate proportion of points to keep for partial overlap (Set to 1.0 to disable)')
    # Model
    parser.add_argument('--method', type=str, default='pydnet', choices=['pydnet'],
                        help='Model to use. Note: Only pydnet is supported for training.'
                             '\'eye\' denotes identity (no registration), \'gt\' denotes groundtruth transforms')
    # PointNet settings
    parser.add_argument('--radius', type=float, default=0.3, help='Neighborhood radius for computing pointnet features')
    parser.add_argument('--num_neighbors', type=int, default=3, metavar='N', help='Max num of neighbors to use')
    # RPMNet settings
    parser.add_argument('--features', type=str, choices=['pmd', 'dxyz', 'xyz'], default=['pmd', 'dxyz', 'xyz'],
                        nargs='+', help='Which features to use. Default: all')
    parser.add_argument('--feat_dim', type=int, default=96,
                        help='Feature dimension (to compute distances on). Other numbers will be scaled accordingly')
    parser.add_argument('--no_slack', action='store_true', help='If set, will not have a slack column.')
    parser.add_argument('--num_sk_iter', type=int, default=5,
                        help='Number of inner iterations used in sinkhorn normalization')
    parser.add_argument('--num_reg_iter', type=int, default=5,
                        help='Number of outer iterations used for registration (only during inference)')
    # parser.add_argument('--loss_type', type=str, choices=['mse', 'mae'], default='mae',
    #                     help='Loss to be optimized')
    parser.add_argument('--loss_type', type=str, choices=['cluster', 'pmd'], default='pmd',
                        help='Loss_fe to be optimized')
    parser.add_argument('--loss_lambda', type=float, default=0.05, help='Feature loss')
    parser.add_argument('--loss_omega_1', type=float, default=3, help='Weight of xyz and dxyz in pmd distance')
    parser.add_argument('--loss_omega_2', type=float, default=1, help='Weight of pmd in pmd distance')
    parser.add_argument('--wt_inliers', type=float, default=1e-2, help='Weight to encourage inliers')
    # Training parameters
    parser.add_argument('--train_batch_size', default=8, type=int, metavar='N',
                        help='training mini-batch size (default 8)')
    parser.add_argument('-b', '--val_batch_size', default=16, type=int, metavar='N',
                        help='mini-batch size during validation or testing (default: 16)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='Pretrained network to load from. Optional for train, required for inference.')
    parser.add_argument('--gpu', default=0, type=int, metavar='DEVICE',
                        help='GPU to use, ignored if no GPU is present. Set to negative to use cpu')
    parser.add_argument('--mode', default='client', type=str, help='For Pycharm DEBUG Only')
    parser.add_argument('--port', default=59347, type=int, help='For Pycharm DEBUG Only')
    parser.add_argument('--feat_dumpdir', '-fd', type=str, metavar='PATH',
                        default=None,
                        help='directory in which the extracted src features will be dumped. Not save when set False')
    parser.add_argument('--do_not_shuffle_points', '-sp',
                        action='store_true',
                        help='whether to shuffle points when loading data')
    parser.add_argument('--parameter_net_norm_type', type=str, choices=['InstanceNorm', 'GroupNorm'], default='InstanceNorm')

    return parser


def pydnet_train_arguments():
    """Used only for training"""
    parser = argparse.ArgumentParser(parents=[pydnet_arguments()])

    parser.add_argument('--train_categoryfile', type=str, metavar='PATH', default='./data_loader/modelnet40_half1.txt',
                        help='path to the categories to be trained')  # eg. './dataset/modelnet40_half1.txt'
    parser.add_argument('--val_categoryfile', type=str, metavar='PATH', default='./data_loader/modelnet40_half1.txt',
                        help='path to the categories to be val')  # eg. './sampledata/modelnet40_half1.txt'

    # Training parameters
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate during training')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--summary_every', default=200, type=int, metavar='N',
                        help='Frequency of saving summary (number of steps if positive, number of epochs if negative)')
    parser.add_argument('--validate_every', default=-4, type=int, metavar='N',
                        help='Frequency of evaluation (number of steps if positive, number of epochs if negative).'
                             'Also saves checkpoints at the same interval')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers for data_loader loader (default: 4).')
    parser.add_argument('--num_train_reg_iter', type=int, default=2,
                        help='Number of outer iterations used for registration (only during training)')
    parser.description = 'Train PydNet'
    return parser


def pydnet_eval_arguments():
    """Used during evaluation"""
    parser = argparse.ArgumentParser(parents=[pydnet_arguments()])

    # settings for input data_loader
    parser.add_argument('--test_category_file', type=str, metavar='PATH', default='./data_loader/modelnet40_half2.txt',
                        help='path to the categories to be val')
    # Provided transforms
    parser.add_argument('--transform_file', type=str,
                        help='If provided, will use transforms from this provided pickle file')
    # Save out evaluation data_loader for further analysis
    parser.add_argument('--eval_save_path', type=str, default='./eval_results',
                        help='Output data_loader to save evaluation results')
    parser.description = 'PydNet evaluation'
    return parser


def pydnet_eval_from_outer_arugments():
    """
    evaluate pcl regression results
    Returns: argparse.ArgumentParser()

    """

    parser = argparse.ArgumentParser(parents=[pydnet_arguments()])

    parser.add_argument('--src_clouds_dir', type=str, required=True)
    parser.add_argument('--ref_clouds_dir', type=str, required=True)
    parser.add_argument('--gt_transform_dir', type=str, required=True)
    parser.add_argument('--pred_transform_dir', type=str, required=True)
    parser.add_argument('--pred_cloud_dir', type=str, default=None)
    parser.add_argument('--eval_save_path', type=str, required=True)

    return parser


def pydnet_param_binary2json_arguments():
    """

    """

    parser = argparse.ArgumentParser(parents=[pydnet_arguments()])

    parser.add_argument('--map_location', type=str, default='cpu')
    parser.add_argument('--model_param_save_dir', type=str, default='pydnet_param_json/')
    parser.add_argument('--indent', type=int, default=2)
    parser.add_argument('--weights_path', type=str, required=True)


    return parser
