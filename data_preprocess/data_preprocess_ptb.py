'''
Data Pre-processing on ptb dataset.

'''

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import scipy.io
import pickle as cp
from sklearn.model_selection import train_test_split
from data_preprocess.base_loader import base_loader


def load_domain_data():
    str_folder = './data/'
    data = np.load(str_folder + 'ptb_np.npy', allow_pickle=True).item()
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
    return x_train, y_train, x_test, y_test

class data_loader_ptb(base_loader):
    def __init__(self, samples, labels, domains):
        super(data_loader_ptb, self).__init__(samples, labels, domains)

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        return sample, target, domain

def prep_domains_ptb_subject_sp(args):
    x_train, y_train, x_test, y_test = load_domain_data()
    #
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
    data_set = data_loader_ptb(x_train, y_train, np.zeros(y_train.shape))
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=None)
    source_loaders = [source_loader]
    # 
    data_set_val = data_loader_ptb(x_val, y_val, np.zeros(y_val.shape))
    val_loader = DataLoader(data_set_val, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=None)
    val_loader = val_loader   
    # target domain data prep
    data_set_test = data_loader_ptb(x_test, y_test, np.zeros(y_test.shape))
    target_loader = DataLoader(data_set_test, batch_size=args.batch_size, shuffle=False)

    return source_loaders, val_loader, target_loader

def prep_ptb(args):
    if args.cases == 'subject_val':
        return prep_domains_ptb_subject_sp(args)
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'

