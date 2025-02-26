# encoding=utf-8
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from models.backbones import *
from models.loss import *
from models.ffc_resnet import *
from models.fno import *
from models.STF import *
from models.SE import *
from models.scatterWave import *
from models.WaveletNet import *
from models.ModernTCN import *
from equiadapt import *
from trainer import *
import torch
import torch.nn as nn
import argparse
from datetime import datetime
import pickle
import numpy as np
import os
import logging
import sys
from data_preprocess.data_preprocess_utils import normalize
from models.pgd import *
from vae_quant import setup_the_VAE, VAE
from scipy import signal
from copy import deepcopy
import fitlog
from utils import tsne, mds, _logger, metrics_TR
from new_augmentations import vanilla_mixup_sup, cutmix_sup
# fitlog.debug()


parser = argparse.ArgumentParser(description='argument setting of network')
parser.add_argument('--cuda', default=0, type=int, help='cuda device ID, 0/1')
# hyperparameter
parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
parser.add_argument('--n_epoch', type=int, default=60, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--lr_cls', type=float, default=1e-3, help='learning rate for linear classifier')
parser.add_argument('--VAE', action='store_true')
parser.add_argument('--VanillaMixup', action='store_true')
parser.add_argument('--BinaryMix', action='store_true')
parser.add_argument('--Cutmix', action='store_true')
parser.add_argument('--Magmix', action='store_true')
parser.add_argument('--phase_shift', action='store_true')
parser.add_argument('--MSE', action='store_true')
parser.add_argument('--robust_check', action='store_true')
parser.add_argument('--controller', action='store_true')
parser.add_argument('--random_aug', action='store_true')
parser.add_argument('--cano', action='store_true')
#
parser.add_argument('--adversary_robust', action='store_true')
parser.add_argument('--eps', type=float, default=0.15, help='epsilon for adversary robustness')

parser.add_argument('--blur', action='store_true')
parser.add_argument('--aps', action='store_true')
# dataset
parser.add_argument('--dataset', type=str, default='ucihar', choices=['physio', 'ucihar', 'hhar', 'usc', 'ieee_small', 'respTR', 'ieee_big', 'dalia', 'chapman', 'clemson', 'sleep', 'ptb'], help='name of dataset')
parser.add_argument('--n_feature', type=int, default=77, help='name of feature dimension')
parser.add_argument('--len_sw', type=int, default=30, help='length of sliding window')
parser.add_argument('--n_class', type=int, default=18, help='number of class')
parser.add_argument('--cases', type=str, default='subject_val', choices=['random', 'subject', 'subject_large', 'cross_device', 'joint_device'], help='name of scenarios')
parser.add_argument('--split_ratio', type=float, default=0.2, help='split ratio of test/val: train(0.64), val(0.16), test(0.2)')
parser.add_argument('--target_domain', type=str, default='0', help='the target domain, [0 to 29] for ucihar, '
                                                                   '[1,2,3,5,6,9,11,13,14,15,16,17,19,20,21,22,23,24,25,29] for shar, '
                                                                   '[a-i] for hhar')

# backbone model
parser.add_argument('--backbone', type=str, default='DCL', choices=['FCN', 'FCN_b', 'DCL', 'LSTM', 'Transformer', 'SE', 'ffc', 'fno', 'resnet', 'TWaveNet','multirate2', 'FreTS', 'STF', 'wavelet', 'WaveletNet', 'ModernTCN'], help='name of framework')
# model paramters
parser.add_argument('--block', type=int, default=3, help='number of groups')
parser.add_argument('--stride', type=int, default=2, help='stride')

# log
parser.add_argument('--logdir', type=str, default='log/', help='log directory')

# AE & CNN_AE
parser.add_argument('--lambda1', type=float, default=1.0, help='weight for reconstruction loss when backbone in [AE, CNN_AE]')
# Multi rate
parser.add_argument('--level', type=int, default=7, help='level for multirate')
parser.add_argument('--conv_kernel', type=int, default=16, help='conv kernel')
parser.add_argument('--mag_ratio', type=float, default=0.2, help='magnitude effect ratio')


# python main_supervised_baseline.py --dataset 'ieee_small' --backbone 'resnet' --block 8 --lr 5e-4 --n_epoch 999 --cuda 0 --phase_shift
# python main_supervised_baseline.py --dataset 'clemson' --backbone 'FCN'  --lr 5e-4 --n_epoch 999 --cuda 3 --aps --random_aug 

# hhar
parser.add_argument('--device', type=str, default='Phones', choices=['Phones', 'Watch'], help='data of which device to use (random case); data of which device to be used as training data (cross-device case, data from the other device as test data)')


# VAE
parser.add_argument('-dist', default='normal', type=str, choices=['normal', 'laplace', 'flow'])
parser.add_argument('-n', '--num-epochs', default=100, type=int, help='number of training epochs')
parser.add_argument('-b', '--batch-size', default=2048, type=int, help='batch size')
parser.add_argument('-l', '--learning-rate', default=1e-3, type=float, help='learning rate')
parser.add_argument('-z', '--latent_dim', default=10, type=int, help='size of latent dimension')
parser.add_argument('--beta', default=5, type=float, help='ELBO penalty term')
parser.add_argument('--tcvae', action='store_true')
parser.add_argument('--exclude-mutinfo', action='store_true')
parser.add_argument('--beta-anneal', action='store_true')
parser.add_argument('--lambda-anneal', action='store_true')
parser.add_argument('--mss', action='store_true', help='use the improved minibatch estimator')
parser.add_argument('--conv', action='store_true')
parser.add_argument('--save', type=str, default='test3')
parser.add_argument('--log_freq', default=200, type=int, help='num iterations per log')


parser.add_argument('--mean', type=float, default=1, help='Mean of Gaussian')
parser.add_argument('--std', type=float, default=0.1, help='std of Gaussian')
parser.add_argument('--low_limit', type=float, default=0.7, help='low limit of Gaussian')
parser.add_argument('--high_limit', type=float, default=1, help='high limit of Gaussian')
parser.add_argument('--alpha', default=0.2, type=float, help='beta term')

############### Parser done ################


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    if args.VAE:
        loss = (lam * nn.CrossEntropyLoss(reduction='none')(pred, y_a) + (1 - lam) * nn.CrossEntropyLoss(reduction='none')(pred, y_b)).sum()/y_a.size(0)
    else:
        loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    return loss

def adjust_learning_rate(optimizer,  epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = BASE_lr * (0.5 ** (epoch // 30))
    # lr = 0.003 * (0.95)**epoch
    lr = 0.005 * (0.95)**epoch
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args, train_loaders, val_loader, model, DEVICE, criterion, save_dir='results/', model_c=None, model_cano=None):
    if args.n_epoch == 0: 
        best_model = deepcopy(model.state_dict())
        model_dir = save_dir + args.model_name + '.pt' if args.phase_shift == False else save_dir + args.model_name + '_phase_shift.pt'
        torch.save(model.state_dict(), model_dir)
    
    if model_c is not None: 
        optimizer_model_c = optim.Adam(model_c.parameters(), lr=5e-4) # it was 5e-4, 5e-4 for clemson and resp, 1e-3 for HHAR
        optimizer_model = optim.Adam(list(model.parameters()) + list(model_c.parameters()), lr=args.lr)
    elif args.cano and model_cano is not None:
        canonicalizer = GroupEquivariantSignalCanonicalization(model_cano, num_translations=16, in_shape = (args.n_feature, args.len_sw)) ### wrap it using equiadapt's canonicalization wrapper
        optimizer_model = torch.optim.Adam([{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}, {'params': canonicalizer.parameters(), 'lr': 1e-3},])        
    else:
        parameters = model.parameters()
        optimizer_model = torch.optim.Adam(parameters, args.lr, weight_decay=args.weight_decay)

    min_val_loss, counter = 1e8, 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_model, mode='min', patience=15, factor=0.5, min_lr=1e-7, verbose=False)
    for epoch in range(args.n_epoch):
        #logger.debug(f'\nEpoch : {epoch}')
        train_loss, n_batches, total, correct = 0, 0, 0, 0
        if args.backbone == 'TWaveNet': 
            adjust_learning_rate(optimizer_model, epoch)
        if args.backbone == 'WaveletNet':
            wave_loss = WaveletLoss(weight_loss=1.)
        model.train()
        if model_c is not None: model_c.train()
        assigned_phase = np.array([])
        for loader_idx, train_loader in enumerate(train_loaders):
            for idx, (sample, target, domain) in enumerate(train_loader):
                n_batches += 1

                if args.controller: 
                    fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
                    # sample_framed = constant_phase_shift(sample, args, DEVICE)
                    ref_frame = model_c(torch.abs(fftsamples).to(DEVICE).float())
                    loss_c = torch.std(ref_frame)
                    sample_c = frame_transform(sample, fftsamples, ref_frame, args, DEVICE)
                    # assigned_phase = np.concatenate((assigned_phase, ref_frame.detach().cpu().numpy())) if assigned_phase.size != 0 else ref_frame.detach().cpu().numpy()
                
                if args.cano:
                    sample = canonicalizer.canonicalize(sample.to(DEVICE).float()) ### canonicalize the inputs

                if args.phase_shift:
                    sample = constant_phase_shift(sample, args, DEVICE)
                elif args.controller:
                    sample = sample_c
                
                if args.random_aug:
                    sample, all_shifts = random_time_shift(sample)

                if not args.dataset == 'spectral':
                    sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
                else:
                    sample, target = sample.to(DEVICE).float(), target.to(DEVICE).float()

                if args.backbone[-2:] == 'AE':
                    out, x_decoded = model(sample)
                else:
                    if args.backbone == 'TWaveNet':
                        out, regus = model(sample)
                    else:
                        out, _ = model(sample)

                loss = criterion(out, target)

                # loss = criterion(out, target) if args.MSE == False else criterion(out.squeeze(), target.float())

                if args.backbone == 'TWaveNet': 
                    loss += sum(regus)
                elif args.backbone == 'multirate2':
                    out, regus_fft = model(sample)
                elif args.backbone == 'WaveletNet':
                    loss = loss + wave_loss(model)

                if args.cano:
                    prior_loss = canonicalizer.get_prior_regularization_loss()
                    loss += prior_loss * 10

                train_loss += loss.item()
                optimizer_model.zero_grad()

                if args.controller: 
                    optimizer_model_c.zero_grad()
                    loss_c.backward(retain_graph=True)

                loss.backward()
                optimizer_model.step()

                if args.controller: 
                    optimizer_model_c.step()

        if val_loader is None:
            best_model = deepcopy(model.state_dict())
            model_dir = save_dir + args.model_name + '.pt'
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, model_dir)
        else:
            with torch.no_grad():
                model.eval()
                if args.controller: model_c.eval()
                val_loss, n_batches, total, correct = 0, 0, 0, 0
                for idx, (sample, target, domain) in enumerate(val_loader):
                    n_batches += 1

                    if args.controller: 
                        fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
                        # sample_framed = constant_phase_shift(sample, args, DEVICE)
                        ref_frame = model_c(torch.abs(fftsamples).to(DEVICE).float())
                        # ref_frame = model_c(sample_framed.to(DEVICE).float())
                        sample_c = frame_transform(sample, fftsamples, ref_frame, args, DEVICE)

                    if args.phase_shift:
                        sample = constant_phase_shift(sample, args, DEVICE)
                    elif args.controller:
                        sample = sample_c

                    sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()

                    if args.backbone[-2:] == 'AE':
                        out, x_decoded = model(sample)
                    else:
                        out, _ = model(sample)

                    if args.MSE == False:
                        loss = criterion(out, target)
                    else:
                        loss = criterion(out.squeeze(), target.float())

                    # loss = criterion(out, target) if args.MSE == False else criterion(out.squeeze(), target.float())

                    if args.backbone[-2:] == 'AE': 
                        loss += nn.MSELoss()(sample, x_decoded) * args.lambda1
                    elif args.backbone == 'TWaveNet':
                        out, regus = model(sample)
                    elif args.backbone == 'multirate2':
                        out, regus_fft = model(sample)

                    if args.backbone == 'multirate2': 
                        loss += regus_fft
                    elif args.backbone == 'TWaveNet':
                        loss += sum(regus)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(out.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum()

                if val_loss <= min_val_loss:
                    min_val_loss = val_loss
                    best_model = deepcopy(model.state_dict())
                    model_dir = save_dir + args.model_name + '.pt' if args.phase_shift == False else save_dir + args.model_name + '_phase_shift.pt'
                    torch.save(model.state_dict(), model_dir)
                    if args.controller:
                        model_dir = save_dir + args.model_name + '_controller.pt'
                        torch.save(model_c.state_dict(), model_dir)
                    if args.cano and model_cano is not None:
                        model_dir = save_dir + args.model_name + '_cano.pt'
                        torch.save(model_cano.state_dict(), model_dir)
                else:
                    counter += 1
                    if counter > 90: 
                        # import pdb;pdb.set_trace();
                        return best_model
                if not args.backbone == 'TWaveNet':
                    scheduler.step(val_loss)
    # import pdb;pdb.set_trace();
    return best_model

def test(test_loader, model, DEVICE, criterion, plot=False, model_c=None, model_cano=None):
    if args.adversary_robust: atk = PGD(model, eps=args.eps, alpha=args.eps/5, steps=10, device=DEVICE)
    # with torch.no_grad():
    model.eval()
    if model_c is not None: 
        model_c.eval()
    if model_cano is not None:
        canonicalizer = GroupEquivariantSignalCanonicalization(model_cano, num_translations=16, in_shape = (args.n_feature, args.len_sw)) ### wrap it using equiadapt's canonicalization wrapper

    total_loss, final_const, n_batches, total, correct = 0, 0, 0, 0, 0
    feats, prds, trgs, cnst = None, None, None, None
    otp, confusion_matrix, assigned_phase = np.array([]), torch.zeros(args.n_class, args.n_class), np.array([])
    for idx, (sample, target, domain) in enumerate(test_loader):
        # import pdb;pdb.set_trace();
        n_batches += 1
        B = sample.shape[0]
        if args.robust_check: # Robustness to time shift
            # continous_shift_evaluate(sample, model, DEVICE, target)
            sample_shifted, all_shifts = random_time_shift(sample)
            sample_shifted_2, _ = random_time_shift(sample)
            sample = torch.cat((sample, sample_shifted_2), 0)
            target = torch.cat((target, target), 0)
        
        if args.controller: 
            fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
            # sample_framed = constant_phase_shift(sample, args, DEVICE)
            ref_frame = model_c(torch.abs(fftsamples).to(DEVICE).float())
            sample_c = frame_transform(sample, fftsamples, ref_frame, args, DEVICE)
            # assigned_phase = np.concatenate((assigned_phase, ref_frame.detach().cpu().numpy())) if assigned_phase.size != 0 else ref_frame.detach().cpu().numpy()

        if args.cano:
            sample = canonicalizer.canonicalize(sample.to(DEVICE).float()) ### canonicalize the inputs

        if args.phase_shift:
            sample = constant_phase_shift(sample, args, DEVICE)
        elif args.controller:
            sample = sample_c

        sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
        out, _ = model(sample)
        # debugging
        # import pdb;pdb.set_trace();
        # shifted_sample = time_shift_one_sample(sample[0,:]) # was 61
        # shifted_out, embeddings_shifted = model(shifted_sample.to(DEVICE))
        # _, predicted_shifted = torch.max(shifted_out.data, 1) # --> confuse with 0 
        # indexes = target == 0
        # embeddings_2 = torch.cat((embeddings[indexes,:], embeddings_shifted), 0)
        # tsne(embeddings_2, torch.cat((torch.zeros(embeddings[indexes,:].size(0),), 1+predicted_shifted.detach().cpu())), 'gg')
        # tsne(embeddings_2, torch.cat((predicted_shifted.detach().cpu(), 1+predicted_shifted.detach().cpu())), 'gg')
        #
        out = out.detach()
        if args.MSE == False:
            loss = criterion(out, target)
        else:
            loss = criterion(out.squeeze(), target.float())
        # loss = criterion(out, target) if args.MSE == False else criterion(out.squeeze(), target.float())

        total_loss += loss.item()
        _, predicted = torch.max(out.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum()
        otp = np.vstack((otp, out.data.cpu().numpy())) if otp.size != 0 else out.data.cpu().numpy()

        if prds is None:
            prds = predicted
            trgs = target
            const = predicted[0:B] - predicted[B:] if args.robust_check else None
        else:
            prds = torch.cat((prds, predicted))
            trgs = torch.cat((trgs, target))
            const = torch.cat((const, predicted[0:B] - predicted[B:])) if args.robust_check else None

    acc_test = float(correct) * 100.0 / total
    maF = f1_score(trgs.cpu().numpy(), prds.cpu().numpy(), average='weighted') * 100
    correlation = f1_score(trgs.cpu().numpy(), prds.cpu().numpy(), average='macro') * 100
    
    if args.dataset == 'ieee_small' or args.dataset =='ieee_big' or args.dataset == 'dalia':
        acc_test = np.sqrt(torch.mean(((trgs-prds)**2).float()).cpu())
        maF = torch.mean((torch.abs(trgs-prds)).float()).cpu()
        correlation = np.corrcoef(trgs.cpu(), prds.cpu())[0,1]
        if np.isnan(correlation): correlation = 0
        if args.robust_check: 
            final_const = 100-100*sum((const != 0)).item()/(const.shape[0])
            print(f'Consistency : {100-100*sum((const != 0)).item()/(const.shape[0]):.2f}')
    elif args.dataset == 'ecg' or args.dataset == 'chapman' or args.dataset == 'physio':
        # import pdb;pdb.set_trace();
        otp1 = softmax(otp,axis=1)
        maF = roc_auc_score(trgs.cpu(), otp1, multi_class='ovo')
        correlation = f1_score(trgs.cpu().numpy(), prds.cpu().numpy(), average='macro') * 100
        if args.robust_check:
            final_const = 100-100*sum((const != 0)).item()/(const.shape[0])
            print(f'Consistency : {100-100*sum((const != 0)).item()/(const.shape[0]):.2f}')
    elif args.dataset == 'clemson': 
        trgs, prds = trgs + 29, prds + 29
        acc_test = 100*torch.mean(torch.abs((trgs-prds)/trgs)).cpu()
        maF = torch.mean((torch.abs(trgs-prds)).float()).cpu()
        correlation = 1
        if args.robust_check: 
            final_const = 100-100*sum((const != 0)).item()/(const.shape[0])
            print(f'Consistency : {100-100*sum((const != 0)).item()/(const.shape[0]):.2f}')
    elif args.dataset == 'sleep':
        maF = f1_score(trgs.cpu().numpy(), prds.cpu().numpy(), average='macro') * 100
        correlation = cohen_kappa_score(trgs.cpu().numpy(), prds.cpu().numpy())
        if args.robust_check:
            final_const = 100-100*sum((const != 0)).item()/(const.shape[0])
            print(f'Consistency : {100-100*sum((const != 0)).item()/(const.shape[0]):.2f}')
    elif args.dataset == 'respTR':
        acc_test = trgs.cpu().numpy()
        maF = softmax(otp,axis=1)
        correlation = prds.cpu().numpy()
        if args.robust_check: 
            final_const = 100-100*sum((const != 0)).item()/(const.shape[0])
            print(f'Consistency : {100-100*sum((const != 0)).item()/(const.shape[0]):.2f}')
    else:
        if args.robust_check: 
            final_const = 100-100*sum((const != 0)).item()/(const.shape[0])
            print(f'Consistency : {100-100*sum((const != 0)).item()/(const.shape[0]):.2f}')
    if plot == True:
        tsne(feats, trgs, domain=None, save_dir=plot_dir_name + args.model_name + '_tsne.png')
        mds(feats, trgs, domain=None, save_dir=plot_dir_name + args.model_name + 'mds.png')
        sns_plot = sns.heatmap(confusion_matrix, cmap='Blues', annot=True)
        sns_plot.get_figure().savefig(plot_dir_name + args.model_name + '_confmatrix.png')

    return acc_test, maF, correlation, final_const, assigned_phase

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_sup(args):
    train_loaders, val_loader, test_loader = setup_dataloaders(args)
    
    if args.MSE: args.n_class = 1

    if args.backbone == 'TWaveNet': 
        part = [[1, 0], [1, 1], [1, 1], [1, 0], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0]]
        args.weight_decay = 1e-4

    if args.backbone == 'FCN':
        if args.blur:
            model = FCN_blur(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, args=args)
        elif args.aps:
            model = FCN_aps(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, args=args)
        else:
            model = FCN(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, args=args)
    elif args.backbone == 'FCN_b':
        model = FCN_big(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, args=args)
    elif args.backbone == 'DCL':
        model = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=False)
    elif args.backbone == 'LSTM':
        model = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=128, backbone=False)
    elif args.backbone == 'AE':
        model = AE(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, outdim=128, backbone=False)
    elif args.backbone == 'CNN_AE':
        model = CNN_AE(n_channels=args.n_feature, n_classes=args.n_class, out_channels=128, backbone=False)
    elif args.backbone == 'Transformer':
        model = Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=False)
    elif args.backbone == 'TWaveNet':
        model = TWaveNet(num_classes=args.n_class, first_conv=args.n_feature, number_level_part=part, kernel_size=5, backbone=False)       
    elif args.backbone == 'fno':
        model = FNO1d(num_class=args.n_class, feature_size=args.n_feature, seq_length=args.len_sw)
    elif args.backbone == 'wavelet':
        model = ScatterWave(args)
    elif args.backbone == 'WaveletNet':
        model = WaveletNet(args=args)  
    elif args.backbone == 'ModernTCN':
        model = ModernTCN(args=args, class_num=args.n_class, seq_len=args.len_sw)      
    elif args.backbone == 'SE':
        model = aia_irm_trans_ri(args)
    elif args.backbone == 'resnet':
        if args.blur:
            model = ResNet1D_blur(in_channels=args.n_feature, base_filters=32, kernel_size=5, stride=args.stride, groups=1, n_block=args.block, n_classes=args.n_class, downsample_gap=2, increasefilter_gap=4, backbone=False)
        elif args.aps:
            model = ResNet1D_aps(in_channels=args.n_feature, base_filters=32, kernel_size=5, stride=args.stride, groups=1, n_block=args.block, n_classes=args.n_class, downsample_gap=2, increasefilter_gap=4, backbone=False)
        else:
            model = ResNet1D(in_channels=args.n_feature, base_filters=32, kernel_size=5, stride=args.stride, groups=1, n_block=args.block, n_classes=args.n_class, downsample_gap=2, increasefilter_gap=4, backbone=False)
    else:
        NotImplementedError

    if args.controller: 
        model_c = FCN_controller(n_channels=args.n_feature, args=args)
        model_c = model_c.to(DEVICE)
    else: 
        model_c = None

    model_cano = ESCNN_translation_EquivariantNetwork(in_shape = (args.n_feature, args.len_sw), out_channels=3).to(DEVICE) if args.cano else None

    model = model.to(DEVICE)
    if args.target_domain == '17' or args.target_domain == 'a' or args.target_domain == '10' or args.target_domain == '0': 
        print('Number of parameters: ', sum(p.numel() for p in model.parameters()))
    args.model_name = args.backbone + '_'+args.dataset + '_cuda' + str(args.cuda) + '_bs' + str(args.batch_size) + '_sw' + str(args.len_sw)

    save_dir = 'results/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # log
    if os.path.isdir(args.logdir) == False:
        os.makedirs(args.logdir)

    criterion = nn.CrossEntropyLoss() # if MSE, criterion = nn.MSELoss()

    best_model = train(args, train_loaders, val_loader, model, DEVICE, criterion, model_c=model_c, model_cano=model_cano)

    if args.backbone == 'FCN':
        if args.blur:
            model_test = FCN_blur(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, args=args)
        elif args.aps:
            model_test = FCN_aps(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, args=args)
        else:
            model_test = FCN(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, args=args)
    if args.backbone == 'FCN_b':
        model_test = FCN_big(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, args=args)
    elif args.backbone == 'DCL':
        model_test = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=False)
    elif args.backbone == 'LSTM':
        model_test = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=128, backbone=False)
    elif args.backbone == 'AE':
        model_test = AE(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, outdim=128, backbone=False)
    elif args.backbone == 'CNN_AE':
        model_test = CNN_AE(n_channels=args.n_feature, n_classes=args.n_class, out_channels=128, backbone=False)
    elif args.backbone == 'Transformer':
        model_test = Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=False)
    elif args.backbone == 'TWaveNet':
        model_test = TWaveNet(num_classes=args.n_class, first_conv=args.n_feature, number_level_part=part, kernel_size=5, backbone=False)    
    elif args.backbone == 'wavelet':
        model_test = ScatterWave(args) 
    elif args.backbone == 'WaveletNet':
        model_test = WaveletNet(args=args)    
    elif args.backbone == 'ModernTCN':
        model_test = ModernTCN(args=args, class_num=args.n_class, seq_len=args.len_sw)          
    elif args.backbone == 'resnet':
        if args.blur:
            model_test = ResNet1D_blur(in_channels=args.n_feature, base_filters=32, kernel_size=5, stride=args.stride, groups=1, n_block=args.block, n_classes=args.n_class, downsample_gap=2, increasefilter_gap=4, backbone=False)
        elif args.aps:
            model_test = ResNet1D_aps(in_channels=args.n_feature, base_filters=32, kernel_size=5, stride=args.stride, groups=1, n_block=args.block, n_classes=args.n_class, downsample_gap=2, increasefilter_gap=4, backbone=False)
        else:
            model_test = ResNet1D(in_channels=args.n_feature, base_filters=32, kernel_size=5, stride=args.stride, groups=1, n_block=args.block, n_classes=args.n_class, downsample_gap=2, increasefilter_gap=4, backbone=False)
        # model_test = ResNet1D(in_channels=args.n_feature, base_filters=32, kernel_size=6, stride=2, groups=1, n_block=2, n_classes=args.n_class, downsample_gap=2, increasefilter_gap=4)
    else:
        NotImplementedError

    if args.controller:
        model_c = FCN_controller(n_channels=args.n_feature, args=args)
        model_dir = save_dir + args.model_name + '_controller.pt'
        model_c.load_state_dict(torch.load(model_dir))
        model_c = model_c.to(DEVICE)

    if args.cano:
        model_cano = ESCNN_translation_EquivariantNetwork(in_shape = (args.n_feature, args.len_sw), out_channels=3).to(DEVICE)
        model_dir = save_dir + args.model_name + '_cano.pt'
        model_cano.load_state_dict(torch.load(model_dir))
        model_cano = model_cano.to(DEVICE)

    model_dir = save_dir + args.model_name + '.pt' if args.phase_shift == False else save_dir + args.model_name + '_phase_shift.pt'
    model_test.load_state_dict(torch.load(model_dir))
    model_test = model_test.to(DEVICE)

    if args.controller:
        acc, mf1, correlation, const, assignedPhase = test(test_loader, model_test, DEVICE, criterion, plot=False, model_c=model_c, model_cano=None)
    elif args.cano:
        acc, mf1, correlation, const, assignedPhase = test(test_loader, model_test, DEVICE, criterion, plot=False, model_c=None, model_cano=model_cano)
    else:    
        acc, mf1, correlation, const, assignedPhase = test(test_loader, model_test, DEVICE, criterion, plot=False)

    return acc, mf1, correlation, const, assignedPhase
    #training_end = datetime.now()
    #training_time = training_end - training_start
    #logger.debug(f"Training time is : {training_time}")


def set_domains(args):
    args = parser.parse_args()
    if args.dataset == 'shar':
        domain = [1, 2, 3, 5]
    elif args.dataset == 'usc':
        domain = [10, 11, 12, 13]        
    elif args.dataset == 'ucihar':
        domain = [0, 1, 2, 3, 4]
    elif args.dataset == 'ieee_small':
        domain = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    elif args.dataset == 'ieee_big':
        domain = [17, 18, 19, 20, 21]
        # domain = [17] # ablation study
    elif args.dataset == 'dalia':
        domain = [0, 1, 2, 3, 4]  
        # domain = [0]  
    elif args.dataset == 'ecg':
        domain = [1, 3]
    elif args.dataset == 'hhar':
        domain = ['a', 'b', 'c', 'd']
    elif args.dataset == 'emg':
        domain = [2, 5, 10]
    elif args.dataset == 'clemson':
        domain = [i for i in range(0, 10)]
    elif args.dataset == 'respTR':
        domain = [i for i in range(0, 9)]
    elif args.dataset == 'chapman' or args.dataset == 'physio' or args.dataset == 'sleep' or args.dataset == 'epilepsy' or args.dataset == 'ptb':
        domain = [0]
    return domain

if __name__ == '__main__':
    args = parser.parse_args()
    domain = set_domains(args)
    all_metrics = []
    DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print('device:', DEVICE, 'dataset:', args.dataset)
    for i in range(3):
        set_seed(i*10+1)
        print(f'Training for seed {i}')
        seed_metric, wholePhase = [], []
        for k in domain:
            setattr(args, 'target_domain', str(k))
            setattr(args, 'save', args.dataset + str(k))
            setattr(args, 'cases', 'subject_val')
            # if args.dataset == 'hhar':
            #     setattr(args, 'cases', 'subject')
            # else:
            #     setattr(args, 'cases', 'subject_large')
            mif,maf,mac, const, assignedPhase = train_sup(args)
            seed_metric.append([mif,maf,mac,const])

            # assignedPhase = np.mod(np.array(assignedPhase), 2*np.pi)
            # import pdb;pdb.set_trace();
            # np.save('wholePhase_array_incr_var.npy', assignedPhase)
        
        if args.dataset == 'respTR':
            auc, accuracy, f1 = metrics_TR(seed_metric)
            all_metrics.append([accuracy, auc, f1, np.mean([seed_metric[i][-1] for i in range(len(seed_metric))])])
        else:
            seed_metric = np.array(seed_metric)
            all_metrics.append([np.mean(seed_metric[:,0]), np.mean(seed_metric[:,1]), np.mean(seed_metric[:,2]), np.mean(seed_metric[:,3])])

    values = np.array(all_metrics)
    mean = np.mean(values,0)
    std = np.std(values,0)
    print('M1: {:.3f}, M2: {:.4f}, M3: {:.4f}'.format(mean[0], mean[1], mean[2]))
    print('Std1: {:.3f}, Std2: {:.4f}, Std3: {:.4f}'.format(std[0], std[1], std[2]))
    if args.robust_check: print('Mean consistency: {:.4f}, Std consistency: {:.4f}'.format(mean[3], std[3]))