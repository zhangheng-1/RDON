
"""
Load args and model from a directory
"""

import os
import numpy as np
import torch
import yaml
from timeit import default_timer
import sys

from torch.utils.data import DataLoader,Dataset
from utilities import *


def get_batch_data(TRAIN_PATH, TEST_PATH, ntrain, ntest, r_train,s_train,r_test,s_test, batch_size,n_out):
    '''
    get and format data for training and testing

    Parameters:
    ----------
        - TRAIN_PATH : path to training data, e.g. '../Data/data/train_random.mat
        - TEST_PATH  : path to training data, e.g. '../Data/data/test_random.mat'
        - ntrain     : number of training data
        - ntest      : number of testing data
        - r_train    : downsampling factor of training data, [fac_input_x, fac_input_y,fac_output_x,fac_output_y]
        - s_train    : resolution of training data, [s_input_x, s_input_y, s_output_x, s_output_y]
        - r_test     : downsampling factor of testing data, [fac_input_x, fac_input_y,fac_output_x,fac_output_y]
        - s_test     : resolution of testing data, [s_input_x, s_input_y, s_output_x, s_output_y]
        - batch_size : batch size for training
        - n_out      : number of output channels, here is 4: rhoxy, phasexy, rhoyx, phaseyx
    '''
    print("begin to read data")
    key_map0 = ['rhoxy','phsxy','rhoyx','phsyx']
    key_map = key_map0[:n_out] # number of output channels
    t_read0 = default_timer()

    # get training data
    reader = MatReader(TRAIN_PATH)
    x_train = reader.read_field('sig')
    x_train = np.abs(x_train) #### 将电导率变成电阻率
    x_train = x_train[:ntrain,::r_train[0],::r_train[1]][:,:s_train[0],:s_train[1]]
    y_train = torch.stack([reader.read_field(key_map[i])\
    [:ntrain,::r_train[2],::r_train[3]][:,:s_train[2],:s_train[3]] for i in range(len(key_map))]).permute(1,2,3,0)
    freq_base    = reader.read_field('freq')[0]
    obs_base     = reader.read_field('obs')[0]
    freq    = torch.log10(freq_base[::r_train[2]][:s_train[2]]) # normalization
    obs     = obs_base[::r_train[3]][:s_train[3]]/torch.max(obs_base) # normalization
    loc1,loc2     = torch.meshgrid(freq,obs)
    # loc is the input of trunck net
    loc_train = torch.cat((loc1.reshape(-1,1),loc2.reshape(-1,1)),-1)
    del reader

    # get test data
    reader_test = MatReader(TEST_PATH)
    x_test = reader_test.read_field('sig')
    x_test = np.abs(x_test) 
    x_test = x_test[:ntest,::r_test[0],::r_test[1]][:,:s_test[0],:s_test[1]]
    y_test = torch.stack([reader_test.read_field(key_map[i])\
    [:ntest,::r_test[2],::r_test[3]][:,:s_test[2],:s_test[3]] for i in range(len(key_map))]).permute(1,2,3,0)
    freq    = torch.log10(freq_base[::r_test[2]][:s_test[2]])
    obs     = obs_base[::r_test[3]][:s_test[3]]/torch.max(obs_base)
    loc1,loc2= torch.meshgrid(freq,obs)
    loc_test = torch.cat((loc1.reshape(-1,1),loc2.reshape(-1,1)),-1)
    del reader_test

    #data normalization
    x_normalizer = GaussianNormalizer(x_train)
    # x_train = x_normalizer.encode(x_train)
    # x_test = x_normalizer.encode(x_test)

    # y_normalizer = GaussianNormalizer(y_train)
    y_normalizer = GaussianNormalizer_out(y_train)
    # y_train = y_normalizer.encode(y_train)
    # y_test = y_normalizer.encode(y_test)

    x_train = x_train.reshape(ntrain,s_train[0],s_train[1],1)
    x_test = x_test.reshape(ntest,s_test[0],s_test[1],1)

    # # Convert data to dataloader 
    # train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    t_read1 = default_timer()
    print(f"reading finished in {t_read1-t_read0:.3f} s")

    return loc_train,loc_test,x_train,y_train,x_test,y_test,freq_base,obs_base,y_normalizer,x_normalizer




class Onet_dataset(Dataset):
    def __init__(self, y, u, Guy):
        self.y = y
        self.u = u
        self.Guy = Guy
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.y[idx], self.u[idx], self.Guy[idx]