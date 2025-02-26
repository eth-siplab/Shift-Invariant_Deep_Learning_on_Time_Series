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
from sklearn.model_selection import train_test_split


def load_domain_data():
    str_folder = './data/'
    data = scipy.io.loadmat(str_folder + 'physio17.mat')
    data = data['signals']
    return data[:,0], data[:,1]

class data_loader_physio(base_loader):
    def __init__(self, samples, labels, domains):
        super(data_loader_physio, self).__init__(samples, labels, domains)

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        return sample, target.item(), domain

def prep_domains_physio_subject_sp(args):
    data_x, data_y = load_domain_data()
    # Split the data into training (60%), validation (20%), and test (20%)
    train_ratio = 0.6
    val_ratio = 0.2
    # Calculate the sizes of training, validation, and test sets
    train_size = int(len(data_x) * train_ratio)
    val_size = int(len(data_x) * val_ratio)
    test_size = len(data_x) - train_size - val_size

    # Use train_test_split to perform the split
    arr1_train, arr1_temp, arr2_train, arr2_temp = train_test_split(data_x, data_y, test_size=(val_size + test_size),stratify=data_y)

    arr1_val, arr1_test, arr2_val, arr2_test = train_test_split(arr1_temp, arr2_temp, test_size=(test_size / (val_size + test_size)), stratify=arr2_temp)

    data_set = data_loader_physio(np.stack(arr1_train, axis=0), np.stack(arr2_train, axis=0), np.zeros(np.stack(arr2_train, axis=0).shape))
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=None)
    source_loaders = [source_loader]

    # 
    data_set_val = data_loader_physio(np.stack(arr1_val, axis=0), np.stack(arr2_val, axis=0), np.zeros(np.stack(arr2_val, axis=0).shape))
    val_loader = DataLoader(data_set_val, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=None)
    val_loader = val_loader   

    # target domain data prep

    data_set_test = data_loader_physio(np.stack(arr1_test, axis=0), np.stack(arr2_test, axis=0), np.zeros(np.stack(arr2_test, axis=0).shape))
    target_loader = DataLoader(data_set_test, batch_size=args.batch_size, shuffle=False)

    return source_loaders, val_loader, target_loader

def prep_domains_physio_subject_np(args):
    data_x, data_y = load_domain_data()
    # Split the data into training (60%), validation (20%), and test (20%)
    train_ratio = 0.6
    val_ratio = 0.2

    # Calculate the sizes of training, validation, and test sets
    train_size = int(len(data_x) * train_ratio)
    val_size = int(len(data_x) * val_ratio)
    test_size = len(data_x) - train_size - val_size

    # Use train_test_split to perform the split
    arr1_train, arr1_temp, arr2_train, arr2_temp = train_test_split(data_x, data_y, test_size=(val_size + test_size))

    arr1_val, arr1_test, arr2_val, arr2_test = train_test_split(arr1_temp, arr2_temp, test_size=(test_size / (val_size + test_size)))
    train_data = np.stack(arr1_train, axis=0), np.stack(arr2_train, axis=0)
    val_data = np.stack(arr1_val, axis=0), np.stack(arr2_val, axis=0)
    test_data = np.stack(arr1_test, axis=0), np.stack(arr2_test, axis=0)
    # 
    return train_data, val_data, test_data

def prep_physio(args):
    if args.cases == 'subject_val':
        return prep_domains_physio_subject_sp(args)
    if args.cases == 'subject_val_np':
        return prep_domains_physio_subject_np(args)    
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'

