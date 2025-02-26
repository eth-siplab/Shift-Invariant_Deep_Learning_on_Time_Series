import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import random
import scipy.io
from data_preprocess.base_loader import base_loader


def load_domain_data(domain_idx):
    str_folder = 'data/respTR/'
    data_all = scipy.io.loadmat(str_folder + 'respTR.mat')
    data_all = data_all['data']
    domain_idx = int(domain_idx)
    X = data_all[domain_idx, 0]
    y = np.squeeze(data_all[domain_idx, 1])
    d = np.full(y.shape, domain_idx)  # Domain information for each sample
    y = np.full((X.shape[1],), y)  # Replicate y for all segments in X
    d = np.full((X.shape[1],), d)  # Replicate d for all segments in X
    return X, y, d


class data_loader_respTR(base_loader):
    def __init__(self, samples, labels, domains, args):
        super(data_loader_respTR, self).__init__(samples, labels, args)
        self.domains = domains

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        return np.squeeze(sample), target, domain


def prep_domains_respTR_subject_val(args):
    # Define source and target domains
    source_domain_list = [i for i in range(0, 42)]
    target_domain_list = [i for i in range(int(args.target_domain) * 4, int(args.target_domain) * 4 + 4)]
    source_domain_list = [x for x in source_domain_list if x not in target_domain_list]

    # Validation domains (1-fold for validation)
    val_domain_list = random.sample(source_domain_list, 4)
    source_domain_list = [x for x in source_domain_list if x not in val_domain_list]

    # Source domain data preparation
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        x, y, d = load_domain_data(source_domain)
        x = np.transpose(x.reshape((-1, 1, x.shape[0], x.shape[1])), (3, 2, 1, 0))  # (n_samples, 32000, 1, 12)
        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        d_win_all = np.concatenate((d_win_all, d), axis=0) if d_win_all.size else d

    data_set = data_loader_respTR(torch.tensor(x_win_all, dtype=torch.float32),
                                  torch.tensor(y_win_all, dtype=torch.long),
                                  torch.tensor(d_win_all, dtype=torch.long),
                                  args)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=True)
    source_loaders = [source_loader]

    # Validation domain data preparation
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for val_domain in val_domain_list:
        x, y, d = load_domain_data(val_domain)
        x = np.transpose(x.reshape((-1, 1, x.shape[0], x.shape[1])), (3, 2, 1, 0))
        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        d_win_all = np.concatenate((d_win_all, d), axis=0) if d_win_all.size else d

    data_set = data_loader_respTR(torch.tensor(x_win_all, dtype=torch.float32),
                                  torch.tensor(y_win_all, dtype=torch.long),
                                  torch.tensor(d_win_all, dtype=torch.long),
                                  args)
    val_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    # Target domain data preparation
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for target_domain in target_domain_list:
        x, y, d = load_domain_data(target_domain)
        x = np.transpose(x.reshape((-1, 1, x.shape[0], x.shape[1])), (3, 2, 1, 0))
        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        d_win_all = np.concatenate((d_win_all, d), axis=0) if d_win_all.size else d

    data_set = data_loader_respTR(torch.tensor(x_win_all, dtype=torch.float32),
                                  torch.tensor(y_win_all, dtype=torch.long),
                                  torch.tensor(d_win_all, dtype=torch.long),
                                  args)
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    return source_loaders, val_loader, target_loader


def prep_respTR(args):
    if args.cases == 'subject_large' or args.cases == 'subject_large_ssl_fn':
        return prep_domains_clemson_subject_large(args)
    elif args.cases == 'subject_val':
        return prep_domains_respTR_subject_val(args)
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'
