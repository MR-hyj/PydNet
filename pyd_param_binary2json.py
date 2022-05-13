import argparse
import logging
import os
import torch
import json
from common.mytorch import to_numpy
from arguments import pydnet_param_binary2json_arguments
from modules.feature_net import FeatExtractionEarlyFusion
from modules.parameter_net import ParameterPredictionNet
from torchsummary import summary


def save_model_param_json(pth_path, _args: argparse.Namespace):
    net = torch.load(pth_path, map_location=_args.map_location)
    for key in net['state_dict'].keys():
        net['state_dict'][key] = to_numpy(net['state_dict'][key]).tolist()

    for key in net['optimizer']['state'].keys():
        for subkey in net['optimizer']['state'][key].keys():
            if isinstance(net['optimizer']['state'][key][subkey], torch.Tensor):
                net['optimizer']['state'][key][subkey] = to_numpy(net['optimizer']['state'][key][subkey]).tolist()

    assert 1 == len(net['optimizer']['param_groups']), logging.error('too long in net[optimizer][param_groups]')
    param_dict = net['optimizer']['param_groups'][0]
    for key in param_dict:
        if isinstance(param_dict[key], torch.Tensor):
            param_dict[key] = to_numpy(param_dict[key]).tolist()

    if not os.path.exists(_args.model_param_save_dir):
        os.makedirs(_args.model_param_save_dir)

    with open(os.path.join(_args.model_param_save_dir, 'pydnet_params_{}.json'.format(net['step'])), 'w') as f:
        f.write(json.dumps(net, indent=_args.indent))


if __name__ == '__main__':
    _args = pydnet_param_binary2json_arguments().parse_args()
    pth_path = _args.weights_path
    save_model_param_json(pth_path, _args)

    feat_extractor = FeatExtractionEarlyFusion(
        features=_args.features, feature_dim=_args.feat_dim,
        radius=_args.radius, num_neighbors=_args.num_neighbors)
    weights_net = ParameterPredictionNet(weights_dim=[0])


    if _args.noise_type == 'crop':
        # src = int(_args.num_points*_args.partial[0])
        # ref = int(_args.num_points*_args.partial[0])
        print('feat extractor network: ')
        summary(feat_extractor, [(int(_args.num_points*_args.partial[0]), 3), (int(_args.num_points*_args.partial[0]), 3)])
        print()
        print()
        print('parameter estimate network: ')
        summary(weights_net, (2, int(_args.num_points*_args.partial[0]), 3))
    else:
        print('feat extractor network: ')
        summary(feat_extractor, [(_args.num_points, 3), (_args.num_points, 3)])
        print()
        print()
        print('parameter estimate network: ')
        summary(weights_net, (2, int(_args.num_points*_args.partial[0]), 3))


