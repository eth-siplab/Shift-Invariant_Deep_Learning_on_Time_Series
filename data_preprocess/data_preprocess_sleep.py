'''
Data Pre-processing on sleep dataset.

'''

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import scipy.io
import pickle as cp
from data_preprocess.data_preprocess_utils import get_sample_weights, train_val_split
from data_preprocess.base_loader import base_loader


def load_domain_data():
    str_folder = './data/'
    data = torch.load(str_folder + 'sleep_combined.pt')
    train = data['train']
    val = data['val']
    test = data['test']
    return train, val, test

class data_loader_sleep(base_loader):
    def __init__(self, samples, labels, domains):
        super(data_loader_sleep, self).__init__(samples, labels, domains)

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        sample = np.reshape(sample, (sample.shape[1], 1, 1))
        return np.squeeze(np.transpose(sample, (1, 0, 2)),0), target, domain

def prep_domains_sleep_subject_sp(args):
    train, val, test = load_domain_data()

    data_set = data_loader_sleep(train['samples'], train['labels'], np.zeros(train['labels'].shape))
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=None)
    source_loaders = [source_loader]

    # 
    data_set_val = data_loader_sleep(val['samples'], val['labels'], np.zeros(val['labels'].shape))
    val_loader = DataLoader(data_set_val, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=None)
    val_loader = val_loader   

    # target domain data prep

    data_set_test = data_loader_sleep(test['samples'], test['labels'], np.zeros(test['labels'].shape))
    target_loader = DataLoader(data_set_test, batch_size=args.batch_size, shuffle=False)

    return source_loaders, val_loader, target_loader

def prep_sleep(args):
    if args.cases == 'subject_val':
        return prep_domains_sleep_subject_sp(args)
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'

