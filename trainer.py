import torch
import torch.nn as nn
import numpy as np
import os
import pickle as cp
from new_augmentations import *
from data_preprocess import data_preprocess_ucihar
from data_preprocess import data_preprocess_hhar
from data_preprocess import data_preprocess_usc
from data_preprocess import data_preprocess_ieee_small
from data_preprocess import data_preprocess_ieee_big
from data_preprocess import data_preprocess_dalia
from data_preprocess import data_preprocess_clemson
from data_preprocess import data_preprocess_chapman
from data_preprocess import data_preprocess_sleep
from data_preprocess import data_preprocess_epilepsy    
from data_preprocess import data_preprocess_physio17
from data_preprocess import data_preprocess_respTR

from models.backbones import *
from models.scatterWave import *
from models.WaveletNet import *
from models.ModernTCN import *

import seaborn as sns
from copy import deepcopy

# create directory for saving models and plots
global model_dir_name
model_dir_name = 'results'
if not os.path.exists(model_dir_name):
    os.makedirs(model_dir_name)
global plot_dir_name
plot_dir_name = 'plot'
if not os.path.exists(plot_dir_name):
    os.makedirs(plot_dir_name)


def setup_dataloaders(args):
    if args.dataset == 'ucihar':
        args.n_feature = 9
        args.len_sw = 128
        args.n_class = 6
        train_loaders, val_loader, test_loader = data_preprocess_ucihar.prep_ucihar(args)
    if args.dataset == 'usc':
        args.n_feature = 6
        args.len_sw = 100
        args.n_class = 12
        train_loaders, val_loader, test_loader = data_preprocess_usc.prep_usc(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int( args.len_sw * 0.5))
    if args.dataset == 'clemson':
        args.n_feature = 1
        args.len_sw = 480
        args.n_class = 48
        train_loaders, val_loader, test_loader = data_preprocess_clemson.prep_clemson(args)
    if args.dataset == 'ieee_small':
        args.n_feature = 1
        args.len_sw = 200
        args.n_class = 180 # 30 -- 210 bpm
        train_loaders, val_loader, test_loader = data_preprocess_ieee_small.prep_ieeesmall(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int( args.len_sw * 0.5))
    if args.dataset == 'ieee_big':
        args.n_feature = 1
        args.len_sw = 200
        args.n_class = 180 # 30 -- 210 bpm
        train_loaders, val_loader, test_loader = data_preprocess_ieee_big.prep_ieeebig(args)     
    if args.dataset == 'dalia':
        args.n_feature = 1
        args.len_sw = 200
        args.n_class = 190 # 30 -- 210 bpm
        train_loaders, val_loader, test_loader = data_preprocess_dalia.prep_dalia(args)         
    if args.dataset == 'respTR':
        args.n_feature = 12
        args.len_sw = 0
        args.n_class = 5
        train_loaders, val_loader, test_loader = data_preprocess_respTR.prep_respTR(args)
    if args.dataset == 'chapman':
        args.n_feature = 4
        args.len_sw = 1000
        n_class = 4 
        setattr(args, 'n_class', n_class)
        train_loaders, val_loader, test_loader = data_preprocess_chapman.prep_chapman(args)              
    if args.dataset == 'hhar':
        args.n_feature = 6
        args.len_sw = 100
        args.n_class = 6
        source_domain = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'] if args.cases == 'subject_large' else ['a', 'b', 'c', 'd']
        train_loaders, val_loader, test_loader = data_preprocess_hhar.prep_hhar(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int(args.len_sw * 0.5),
                                                                                device=args.device,
                                                                                train_user=source_domain,
                                                                                test_user=args.target_domain)
    if args.dataset == 'sleep':
        args.n_feature = 1
        args.len_sw = 3000
        args.n_class = 5
        train_loaders, val_loader, test_loader = data_preprocess_sleep.prep_sleep(args)
    if args.dataset == 'physio':
        args.n_feature = 1
        args.len_sw = 6000
        args.n_class = 4
        train_loaders, val_loader, test_loader = data_preprocess_physio17.prep_physio(args)
    return train_loaders, val_loader, test_loader


def build_model(args, part=None):
    """Instantiate and return a model based on args.backbone and related options."""
    if args.backbone == 'TWaveNet':
        # Set default part and weight decay for TWaveNet if not provided.
        if part is None:
            part = [[1, 0], [1, 1], [1, 1], [1, 0], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0]]
            args.weight_decay = 1e-4
        return TWaveNet(num_classes=args.n_class, first_conv=args.n_feature,
                        number_level_part=part, kernel_size=5, backbone=False)

    elif args.backbone == 'FCN':
        if args.blur:
            return FCN_blur(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, args=args)
        elif args.aps:
            return FCN_aps(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, args=args)
        else:
            return FCN(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, args=args)

    elif args.backbone == 'FCN_b':
        return FCN_big(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, args=args)

    elif args.backbone == 'DCL':
        return DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class,
                            conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=False)

    elif args.backbone == 'LSTM':
        return LSTM(n_channels=args.n_feature, n_classes=args.n_class,
                    LSTM_units=128, backbone=False)

    elif args.backbone == 'AE':
        return AE(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class,
                  outdim=128, backbone=False)

    elif args.backbone == 'CNN_AE':
        return CNN_AE(n_channels=args.n_feature, n_classes=args.n_class,
                      out_channels=128, backbone=False)

    elif args.backbone == 'Transformer':
        return Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class,
                           dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=False)

    elif args.backbone == 'wavelet':
        return ScatterWave(args)

    elif args.backbone == 'WaveletNet':
        return WaveletNet(args=args)

    elif args.backbone == 'ModernTCN':
        return ModernTCN(args=args, class_num=args.n_class, seq_len=args.len_sw)

    elif args.backbone == 'resnet':
        if args.blur:
            return ResNet1D_blur(in_channels=args.n_feature, base_filters=32, kernel_size=5,
                                 stride=args.stride, groups=1, n_block=args.block,
                                 n_classes=args.n_class, downsample_gap=2, increasefilter_gap=4, backbone=False)
        elif args.aps:
            return ResNet1D_aps(in_channels=args.n_feature, base_filters=32, kernel_size=5,
                                stride=args.stride, groups=1, n_block=args.block,
                                n_classes=args.n_class, downsample_gap=2, increasefilter_gap=4, backbone=False)
        else:
            return ResNet1D(in_channels=args.n_feature, base_filters=32, kernel_size=5,
                            stride=args.stride, groups=1, n_block=args.block,
                            n_classes=args.n_class, downsample_gap=2, increasefilter_gap=4, backbone=False)
    else:
        raise NotImplementedError(f"Backbone {args.backbone} not implemented.")
