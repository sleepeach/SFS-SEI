import os
import torch
import random
import numpy as np
from data_load import *
from optimizer_SGD_APGD import *


'''load dataset'''
class Dataset:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classes = 100
    dataset = getdata_ADS_B()
    batch_size = 64


'''CrossEntropyLoss'''
class HP_train:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lr = 0.001
    optimizer1 = SGD
    optimizer2 = APGD
    loss = torch.nn.CrossEntropyLoss(reduction = 'sum')
    epoch = 100
    log_path ="Log/ADS_B4800/logs_CNN_CE_batchsize%d_epoch%d_lr%d" %(Dataset.batch_size, epoch,lr)
    save_path ='Model_weights/ADS_B4800/CNN_CE_batchsize%d_epoch%d_lr%d.pth' %(Dataset.batch_size, epoch,lr)
    prune_save_path ='New_model_weights/ADS_B4800/CNN_CE_batchsize%d_epoch%d_lr%d.pth' %(Dataset.batch_size, epoch,lr)
    alpha = 0.001

class HP_val:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lr = 0.001
    loss = torch.nn.CrossEntropyLoss(reduction='sum')
    epoch = 100
    log_path ="Log/ADS_B4800/logs_CNN_CE_batchsize%d_epoch%d_lr%d" %(Dataset.batch_size, epoch,lr)

class HP_test:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_path = HP_train.save_path
    save_path = HP_train.prune_save_path
