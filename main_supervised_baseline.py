# encoding=utf-8
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score
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
from scipy import signal
from copy import deepcopy
import fitlog
from utils import tsne, mds, _logger, metrics_TR
# fitlog.debug()


parser = argparse.ArgumentParser(description='argument setting of network')
parser.add_argument('--cuda', default=0, type=int, help='cuda device ID, 0/1')
# hyperparameter
parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
parser.add_argument('--n_epoch', type=int, default=60, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
# 
parser.add_argument('--phase_shift', action='store_true')
parser.add_argument('--robust_check', action='store_true')
parser.add_argument('--controller', action='store_true')
parser.add_argument('--random_aug', action='store_true')
parser.add_argument('--cano', action='store_true')
parser.add_argument('--blur', action='store_true')
parser.add_argument('--aps', action='store_true')
# dataset
parser.add_argument('--dataset', type=str, default='ucihar', choices=['physio', 'ucihar', 'hhar', 'usc', 'ieee_small', 'respTR', 'ieee_big', 'dalia', 'chapman', 'clemson', 'sleep'], help='name of dataset')
parser.add_argument('--n_feature', type=int, default=77, help='name of feature dimension')
parser.add_argument('--len_sw', type=int, default=30, help='length of sliding window')
parser.add_argument('--n_class', type=int, default=18, help='number of class')
parser.add_argument('--cases', type=str, default='subject_val', choices=['random', 'subject', 'subject_large', 'cross_device', 'joint_device'], help='name of scenarios')
parser.add_argument('--split_ratio', type=float, default=0.2, help='split ratio of test/val: train(0.64), val(0.16), test(0.2)')
parser.add_argument('--target_domain', type=str, default='0', help='the target domain, [0 to 29] for ucihar, '
                                                                   '[1,2,3,5,6,9,11,13,14,15,16,17,19,20,21,22,23,24,25,29] for shar, '
                                                                   '[a-i] for hhar')

# models
parser.add_argument('--backbone', type=str, default='DCL', choices=['FCN', 'FCN_b', 'DCL', 'LSTM', 'Transformer', 'resnet', 'TWaveNet','multirate2', 'wavelet', 'WaveletNet', 'ModernTCN'], help='name of framework')
# model parameters
parser.add_argument('--block', type=int, default=3, help='number of groups')
parser.add_argument('--stride', type=int, default=2, help='stride')

# log
parser.add_argument('--logdir', type=str, default='log/', help='log directory')

# AE & CNN_AE
parser.add_argument('--lambda1', type=float, default=1.0, help='weight for reconstruction loss when backbone in [AE, CNN_AE]')


# python main_supervised_baseline.py --dataset 'ieee_small' --backbone 'resnet' --block 8 --lr 5e-4 --n_epoch 999 --cuda 0 --phase_shift
# python main_supervised_baseline.py --dataset 'ieee_small' --backbone 'resnet' --block 8 --lr 5e-4 --n_epoch 999 --cuda 0 --controller
# python main_supervised_baseline.py --dataset 'clemson' --backbone 'FCN'  --lr 5e-4 --n_epoch 999 --cuda 3 --aps --random_aug 

# hhar
parser.add_argument('--device', type=str, default='Phones', choices=['Phones', 'Watch'], help='data of which device to use (random case); data of which device to be used as training data (cross-device case, data from the other device as test data)')


############### Parser done ################

def adjust_learning_rate(optimizer,  epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = BASE_lr * (0.5 ** (epoch // 30))
    # lr = 0.003 * (0.95)**epoch
    lr = 0.005 * (0.95)**epoch
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args, train_loaders, val_loader, model, DEVICE, criterion, save_dir='results/', model_c=None, model_cano=None):
    if args.n_epoch == 0: # no training
        best_model = deepcopy(model.state_dict())
        model_dir = save_dir + args.model_name + '.pt' if args.phase_shift == False else save_dir + args.model_name + '_phase_shift.pt'
        torch.save(model.state_dict(), model_dir)
    
    canonicalizer = None
    if model_c is not None: 
        optimizer_model_c = torch.optim.Adam(model_c.parameters(), lr=5e-4) # it was 5e-4 for clemson and resp, 1e-3 for HHAR
        optimizer_model = torch.optim.Adam(list(model.parameters()) + list(model_c.parameters()), lr=args.lr)
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

        for loader_idx, train_loader in enumerate(train_loaders):
            for idx, (sample, target, domain) in enumerate(train_loader):
                n_batches += 1
                target = target.to(DEVICE).long()

                sample, loss_c = process_sample(sample, args, model_c, canonicalizer, DEVICE)
                
                if args.random_aug:
                    sample, all_shifts = random_time_shift(sample)

                if args.backbone[-2:] == 'AE':
                    out, x_decoded = model(sample)
                else:
                    if args.backbone == 'TWaveNet':
                        out, regus = model(sample)
                    else:
                        out, _ = model(sample)

                loss = criterion(out, target)

                if args.backbone == 'TWaveNet': 
                    loss += sum(regus)
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

                if args.controller: 
                    model_c.eval()

                val_loss, total, correct = 0, 0, 0
                for idx, (sample, target, domain) in enumerate(val_loader):

                    if args.controller: 
                        fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
                        ref_frame = model_c(torch.abs(fftsamples).to(DEVICE).float())
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

                    loss = criterion(out.squeeze(), target)

                    if args.backbone[-2:] == 'AE': 
                        loss += nn.MSELoss()(sample, x_decoded) * args.lambda1
                    elif args.backbone == 'TWaveNet':
                        out, regus = model(sample)

                    if args.backbone == 'TWaveNet':
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
                        return best_model
                if not args.backbone == 'TWaveNet':
                    scheduler.step(val_loss)
    return best_model

def process_sample(sample, args, model_c, canonicalizer, DEVICE):
    """Process the sample with potential time shifting, controller transform,
    canonicalization, and phase shifting without a guidance network."""
    # Apply robust time shift if required
    if args.robust_check:
        sample_shifted, _ = random_time_shift(sample)
        sample_shifted2, _ = random_time_shift(sample)
        sample = torch.cat((sample, sample_shifted2), 0)
    
    # Apply controller transformation if enabled
    sample_c = None
    loss_c = None
    if args.controller:
        fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
        ref_frame = model_c(torch.abs(fftsamples).to(DEVICE).float())
        loss_c = torch.std(ref_frame)
        sample_c = frame_transform(sample, fftsamples, ref_frame, args, DEVICE)
    
    # Apply canonicalization if enabled
    if args.cano and canonicalizer is not None:
        sample = canonicalizer.canonicalize(sample.to(DEVICE).float())
    
    # Apply phase shift if enabled; otherwise, use controller output if available
    if args.phase_shift:
        sample = constant_phase_shift(sample, args, DEVICE)
    elif args.controller and sample_c is not None:
        sample = sample_c

    return sample.to(DEVICE).float(), loss_c

def compute_metrics(args, targets, predictions, acc_test, otp):
    """Compute evaluation metrics based on the dataset."""
    from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score
    # Default metric calculations
    # acc_test = f1_score(targets.cpu().numpy(), predictions.cpu().numpy(), average='weighted') * 100
    maF = f1_score(targets.cpu().numpy(), predictions.cpu().numpy(), average='weighted') * 100
    correlation = f1_score(targets.cpu().numpy(), predictions.cpu().numpy(), average='macro') * 100

    if args.dataset in ('ieee_small', 'ieee_big', 'dalia'): # RMSE | MAE | correlation
        acc_test = np.sqrt(torch.mean(((targets - predictions) ** 2).float()).cpu())
        maF = torch.mean(torch.abs(targets - predictions).float()).cpu()
        correlation = np.corrcoef(targets.cpu(), predictions.cpu())[0, 1]
        correlation = 0 if np.isnan(correlation) else correlation

    elif args.dataset in ('ecg', 'chapman', 'physio'): # W-F1 | AUC | F1
        otp1 = softmax(otp, axis=1)
        maF = roc_auc_score(targets.cpu(), otp1, multi_class='ovo')
        correlation = f1_score(targets.cpu().numpy(), predictions.cpu().numpy(), average='macro') * 100

    elif args.dataset == 'clemson':
        targets, predictions = targets + 29, predictions + 29
        acc_test = 100 * torch.mean(torch.abs((targets - predictions) / targets)).cpu()
        maF = torch.mean(torch.abs(targets - predictions).float()).cpu()
        correlation = 1

    elif args.dataset == 'sleep': # ACC | W-F1 | Kappa
        maF = f1_score(targets.cpu().numpy(), predictions.cpu().numpy(), average='macro') * 100
        correlation = cohen_kappa_score(targets.cpu().numpy(), predictions.cpu().numpy())

    elif args.dataset == 'respTR':
        acc_test = targets.cpu().numpy()
        maF = softmax(otp, axis=1)
        correlation = predictions.cpu().numpy()

    return acc_test, maF, correlation # If activity -->  Acc | W-F1 | F1

def compute_consistency(predicted, batch_size):
    """Compute consistency metric for robust checking."""
    return 100 - 100 * (predicted[:batch_size] - predicted[batch_size:]).ne(0).sum().item() / batch_size

def test(test_loader, model, DEVICE, criterion, plot=False, model_c=None, model_cano=None):
    model.eval()
    if model_c is not None:
        model_c.eval()
    canonicalizer = None
    if model_cano is not None:
        canonicalizer = GroupEquivariantSignalCanonicalization(
            model_cano, num_translations=16, in_shape=(args.n_feature, args.len_sw)
        )

    total_loss = 0.0
    n_batches, total_samples, correct_preds = 0, 0, 0
    predictions, targets = None, None
    otp = None  # stores output probabilities/values
    final_consistency = None

    for idx, (sample, target, domain) in enumerate(test_loader):
        n_batches += 1
        batch_size = sample.shape[0]

        sample = process_sample(sample, args, model_c, canonicalizer, DEVICE)
        
        sample = sample.to(DEVICE).float()
        target = target.to(DEVICE).long()

        out, _ = model(sample)
        out = out.detach()

        # Compute loss and update totals
        loss = criterion(out.squeeze(), target)
        total_loss += loss.item()
        _, predicted = torch.max(out.data, 1)
        total_samples += target.size(0)
        correct_preds += (predicted == target).sum()

        # Collect output for metrics
        current_otp = out.data.cpu().numpy()
        otp = np.vstack((otp, current_otp)) if otp is not None else current_otp

        # Aggregate predictions and targets
        if predictions is None:
            predictions = predicted
            targets = target
            if args.robust_check:
                final_consistency = compute_consistency(predicted, batch_size)
        else:
            predictions = torch.cat((predictions, predicted))
            targets = torch.cat((targets, target))
            if args.robust_check:
                cons = compute_consistency(predicted, batch_size)
                final_consistency = (final_consistency + cons) / 2

    acc_test = float(correct_preds) * 100.0 / total_samples
    acc_test, maF, correlation = compute_metrics(args, targets, predictions, acc_test, otp)

    # Optional plotting
    if plot:
        tsne(feats, targets, domain=None, save_dir=plot_dir_name + args.model_name + '_tsne.png')
        mds(feats, targets, domain=None, save_dir=plot_dir_name + args.model_name + 'mds.png')
        sns_plot = sns.heatmap(torch.zeros(args.n_class, args.n_class), cmap='Blues', annot=True)
        sns_plot.get_figure().savefig(plot_dir_name + args.model_name + '_confmatrix.png')

    return acc_test, maF, correlation, final_consistency

def train_sup(args):
    train_loaders, val_loader, test_loader = setup_dataloaders(args)

    if args.backbone == 'TWaveNet': 
        part = [[1, 0], [1, 1], [1, 1], [1, 0], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0]]
        args.weight_decay = 1e-4
    else: part = None

    # Instantiate the training model 
    model = build_model(args, part=part)
    model = model.to(DEVICE)

    # Set up controller if required.
    model_c = FCN_controller(n_channels=args.n_feature, args=args).to(DEVICE) if args.controller else None

    # Instantiate canonicalizer if required.
    model_cano = ESCNN_translation_EquivariantNetwork(in_shape=(args.n_feature, args.len_sw),
                                                    out_channels=3).to(DEVICE) if args.cano else None

    # Print parameter count.
    if args.target_domain in ('17', 'a', '10', '0'):
        print('Number of parameters:', sum(p.numel() for p in model.parameters()))

    # Set a descriptive model name and ensure directories exist.
    args.model_name = f"{args.backbone}_{args.dataset}_cuda{args.cuda}_bs{args.batch_size}_sw{args.len_sw}"
    save_dir = 'results/'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()

    # Train the model.
    best_model = train(args, train_loaders, val_loader, model, DEVICE, criterion,
                    model_c=model_c, model_cano=model_cano)

    # Instantiate the test model using the same helper.
    model_test = build_model(args, part=part).to(DEVICE)
    model_test.load_state_dict(best_model)

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
        acc, mf1, correlation, const = test(test_loader, model_test, DEVICE, criterion, plot=False, model_c=model_c, model_cano=None)
    elif args.cano:
        acc, mf1, correlation, const = test(test_loader, model_test, DEVICE, criterion, plot=False, model_c=None, model_cano=model_cano)
    else:    
        acc, mf1, correlation, const = test(test_loader, model_test, DEVICE, criterion, plot=False)

    return acc, mf1, correlation, const

######################################## 

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

def set_domains(args):
    args = parser.parse_args()
    if args.dataset == 'usc':
        domain = [10, 11, 12, 13]        
    elif args.dataset == 'ucihar':
        domain = [0, 1, 2, 3, 4]
    elif args.dataset == 'ieee_small':
        domain = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    elif args.dataset == 'ieee_big':
        domain = [17, 18, 19, 20, 21]
    elif args.dataset == 'dalia':
        domain = [0, 1, 2, 3, 4]  
    elif args.dataset == 'ecg':
        domain = [1, 3]
    elif args.dataset == 'hhar':
        domain = ['a', 'b', 'c', 'd']
    elif args.dataset == 'clemson':
        domain = [i for i in range(0, 10)]
    elif args.dataset == 'respTR':
        domain = [i for i in range(0, 9)]
    elif args.dataset == 'chapman' or args.dataset == 'physio' or args.dataset == 'sleep':
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

            mif, maf, mac, const = train_sup(args)
            seed_metric.append([mif,maf,mac,const])

        
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
