import os 
import argparse
from main import main
import numpy as np


parser = argparse.ArgumentParser(description='argument setting of network')
parser.add_argument('--cuda', default=0, type=int, help='cuda device ID, 0/1')
# Hyperparameter
parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
parser.add_argument('--n_epoch', type=int, default=60, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_cls', type=float, default=1e-3, help='learning rate for linear classifier')
parser.add_argument('--scheduler', type=bool, default=True, help='if or not to use a scheduler')
parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--VAE', action='store_true', help='Finding Order in Chaos')
parser.add_argument('--robust_shift', action='store_true', help='Robust shift')
parser.add_argument('--NTX_mod', action='store_true', help='Modified NTXent')
parser.add_argument('--mean', type=float, default=1, help='Mean of Gaussian')
parser.add_argument('--std', type=float, default=0.1, help='std of Gaussian')
parser.add_argument('--low_limit', type=float, default=0.7, help='low limit of Gaussian')
parser.add_argument('--high_limit', type=float, default=1, help='high limit of Gaussian')
## Comparison with Baselines
parser.add_argument('--VanillaMixup', action='store_true', help='Prior work') # Linear mixup
parser.add_argument('--CutMix', action='store_true', help='Prior work')
parser.add_argument('--BinaryMix', action='store_true', help='Prior work')
parser.add_argument('--SpecMix', action='store_true', help='Prior work')
parser.add_argument('--BestGeoMix', action='store_true', help='Prior work') 
parser.add_argument('--GeoMix', action='store_true', help='Prior work')
parser.add_argument('--Randomfftmix', action='store_true', help='Prior work')
parser.add_argument('--ablation_2', action='store_true', help='Prior work')
parser.add_argument('--opposite_phase', action='store_true', help='Prior work') # Importance of phase interpolation
parser.add_argument('--ablation_tfc', action='store_true', help='Prior work')  # Aug. Bank
parser.add_argument('--STAug', action='store_true', help='Prior work')  # STAug 
parser.add_argument('--BestMixup', action='store_true', help='Prior work')
parser.add_argument('--DACL', action='store_true', help='Prior work')
parser.add_argument('--COPGEN', action='store_true', help='Prior work')
parser.add_argument('--IDAA', action='store_true', help='Prior work')
parser.add_argument('--alpha', default=1.0, type=float, help='weight for contrastive loss with adversarial example')
parser.add_argument('--GaussLatent', action='store_true', help='Prior work')
parser.add_argument('--dim_mixing', action='store_true', help='Prior work')
parser.add_argument('--InfoMin', action='store_true', help='Prior work')
parser.add_argument('--robust_check', action='store_true', help='Shift consistency')
parser.add_argument('--covariance', action='store_true', help='Training for shift consistency')
# Datasets
parser.add_argument('--dataset', type=str, default='ucihar', choices=['ucihar', 'shar', 'hhar', 'usc', 'ieee_small','ieee_big', 'dalia','ecg'], help='name of dataset')
parser.add_argument('--n_feature', type=int, default=77, help='name of feature dimension')
parser.add_argument('--len_sw', type=int, default=30, help='length of sliding window')
parser.add_argument('--n_class', type=int, default=18, help='number of class')
parser.add_argument('--cases', type=str, default='random', choices=['random', 'subject', 'subject_large', 'cross_device', 'joint_device'],
                    help='name of scenarios, cross_device and joint_device only applicable when hhar is used')
parser.add_argument('--split_ratio', type=float, default=0.2, help='split ratio of test/val: train(0.64), val(0.16), test(0.2)')
parser.add_argument('--target_domain', type=str, default='0', help='the target domain, [0 to 29] for ucihar, '
                                                                   '[1,2,3,5,6,9,11,13,14,15,16,17,19,20,21,22,23,24,25,29] for shar, '
                                                                   '[a-i] for hhar')

# Augmentations
parser.add_argument('--aug1', type=str, default='jit_scal',
                    choices=['na', 'noise', 'scale', 'negate', 'perm', 'shuffle', 'shift', 't_flip', 't_warp', 'resample', 'random_out','rotation', 'perm_jit', 'jit_scal', 'hfc', 'lfc', 'p_shift', 'ap_p', 'ap_f','order'],
                    help='the type of augmentation transformation')
parser.add_argument('--aug2', type=str, default='resample',
                    choices=['na', 'noise', 'scale', 'negate', 'perm', 'shuffle', 'shift', 't_flip', 't_warp', 'resample', 'random_out', 'rotation', 'perm_jit', 'jit_scal', 'hfc', 'lfc', 'p_shift', 'ap_p', 'ap_f', 'order'],
                    help='the type of augmentation transformation')

# Frameworks
parser.add_argument('--framework', type=str, default='byol', choices=['byol', 'simsiam', 'simclr', 'nnclr', 'tstcc', 'auditory'], help='name of framework')
parser.add_argument('--backbone', type=str, default='DCL', choices=['FCN', 'DCL', 'LSTM', 'AE', 'CNN_AE', 'Transformer', 'resnet', 'resnet_ssl'], help='name of backbone network')
parser.add_argument('--criterion', type=str, default='cos_sim', choices=['cos_sim', 'NTXent'],
                    help='type of loss function for contrastive learning')
parser.add_argument('--p', type=int, default=128,
                    help='byol: projector size, simsiam: projector output size, simclr: projector output size')
parser.add_argument('--phid', type=int, default=128,
                    help='byol: projector hidden size, simsiam: predictor hidden size, simclr: na')

# log
parser.add_argument('--logdir', type=str, default='log/', help='log directory')

# byol
parser.add_argument('--lr_mul', type=float, default=10.0,
                    help='lr multiplier for the second optimizer when training byol')
parser.add_argument('--EMA', type=float, default=0.996, help='exponential moving average parameter')

# nnclr
parser.add_argument('--mmb_size', type=int, default=1024, help='maximum size of NNCLR support set')

# TS-TCC
parser.add_argument('--lambda1', type=float, default=1.0, help='weight for temporal contrastive loss')
parser.add_argument('--lambda2', type=float, default=0.7, help='weight for contextual contrastive loss, also used as the weight for reconstruction loss when AE or CAE being backbone network')
parser.add_argument('--temp_unit', type=str, default='tsfm', choices=['tsfm', 'lstm', 'blstm', 'gru', 'bgru'], help='temporal unit in the TS-TCC')

# hhar
parser.add_argument('--device', type=str, default='Phones', choices=['Phones', 'Watch'], help='data of which device to use (random case); data of which device to be used as training data (cross-device case, data from the other device as test data)')

# plot
parser.add_argument('--plt', type=bool, default=False, help='if or not to plot results')
parser.add_argument('--plot_tsne', action='store_true', help='if or not to plot t-SNE')

# Example: python runner_function.py --framework 'nnclr' --backbone 'DCL' --dataset 'ieee_small' --aug1 'na' --aug2 'perm_jit' --n_epoch 120 --batch_size 256 --lr 3e-3 --lr_cls 0.03 --cuda 0 --cases 'subject_large' 
# Example: python runner_function.py --framework 'simclr' --backbone 'resnet' --dataset 'ieee_big' --aug1 'na' --aug2 'perm_jit' --n_epoch 120 --batch_size 256 --lr 3e-3 --lr_cls 0.03 --cuda 1 --cases 'subject_large' 

# VAE Settings 
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
# parser.add_argument('--visdom', action='store_true', help='whether plotting in visdom is desired')
parser.add_argument('--save', type=str, default='test3')  
parser.add_argument('--log_freq', default=200, type=int, help='num iterations per log')
############### Parser done ################


# Unique seed for each run
def get_unique_seed(used_seeds):
    while True:
        seed = np.random.randint(1, 1000)  # You can adjust the range based on your preference
        if seed not in used_seeds:
            used_seeds.append(seed)
            return seed

# Domains for each dataset
def set_domains(args):
    args = parser.parse_args()
    if args.dataset == 'shar':
        domain = [1, 2, 3, 5]
    elif args.dataset == 'ucihar':
        domain = [0, 1, 2, 3, 4]
        args.fs = 50 # sampling rate
    elif args.dataset == 'usc':
        domain = [10, 11, 12, 13]
    elif args.dataset == 'ieee_small':
        domain = [0, 1, 2, 3, 4]
    elif args.dataset == 'ieee_big':
        domain = [17, 18, 19, 20, 21]
    elif args.dataset == 'dalia':
        domain = [0, 1, 2, 3, 4]     
    elif args.dataset == 'ecg':
        domain = [1, 3]
    elif args.dataset == 'hhar':
        domain = ['a', 'b', 'c', 'd']
    return domain

if __name__ == '__main__':
    args = parser.parse_args()
    all_metrics, used_seeds = [], []
    for i in range(3): # 3 different seeds
        seed = i * 10 + 1
        domain, error = set_domains(args), []
        print('Loop: {}'.format(i))
        for k in domain: # For each domain
            setattr(args, 'target_domain', str(k))
            setattr(args, 'save', 'results/' + args.dataset + str(k))
            setattr(args, 'cases', 'subject_large')
            mif,maf,mac,consis = main(args, seed)
            error.append([mif,maf,mac, consis])
            print('Domain: {}, M1: {:.3f}, M2: {:.4f}, M3: {:.4f}, Consistency: {:.4f}'.format(k, mif,maf,mac, consis))
        values = np.array(error)
        all_metrics.append([np.mean(values[:,0]), np.mean(values[:,1]), np.mean(values[:,2]), np.mean(values[:,3])])
        print('Seed: {}, M1: {:.3f}, M2: {:.4f}, M3: {:.4f}, Consistency: {:.4f}'.format(seed, all_metrics[-1][0], all_metrics[-1][1], all_metrics[-1][2], all_metrics[-1][3]))

    valuesf = np.array(all_metrics)
    mean = np.mean(valuesf,0)
    std = np.std(valuesf,0)
    print('M1: {:.3f}, M2: {:.4f}, M3: {:.4f}, C: {:.4f}'.format(mean[0], mean[1], mean[2], mean[3]))
    print('Std1: {:.3f}, Std2: {:.4f}, Std3: {:.4f}, stdC: {:.4f}'.format(std[0], std[1], std[2], std[3]))