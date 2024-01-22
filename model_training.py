import torch
import numpy as np
import pandas as pd 
from torch import nn
import os
from pathlib import Path
from tfrecord.tools import tfrecord2idx
import mymodel_cnn # CNN
import ood_genomics_dataset
from torch.utils.data import DataLoader

# use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# データの読み込み
train = ood_genomics_dataset.OODGenomicsDataset("data", "train") # before_2011_in_tr
in_valid = ood_genomics_dataset.OODGenomicsDataset("data", "val") # between_2011-2016_in_valid
ood_valid = ood_genomics_dataset.OODGenomicsDataset("data", "val_ood") #between_2011-2016_ood_valid
in_test = ood_genomics_dataset.OODGenomicsDataset("data", "test") # after_2016_in_test
ood_test = ood_genomics_dataset.OODGenomicsDataset("data", "test_ood") #after_2016_ood_test

train_iter = next(iter(train))
in_valid_iter = next(iter(in_valid))
ood_valid_iter = next(iter(ood_valid))
in_test_iter = next(iter(in_test))
ood_test_iter = next(iter(ood_test))
print("train: ", train_iter.figure())

train_label = train.label_dict
in_valid_label = in_valid.label_dict
ood_valid_label = ood_valid.label_dict
in_test_label = in_test.label_dict
ood_test_label = ood_test.label_dict

train_loder = DataLoader(train, batch_size = 32)
in_valid_loder = DataLoader(in_valid, batch_size = 32)
ood_valid_loder = DataLoader(ood_valid, batch_size = 32)



"""
# optimizer
optimizer = torch.optim.AdamW(params = model.parameters(),
                              lr = 0.001, weight_decay = 0.01)
"""

# loss function
loss_function = torch.nn.MSELoss()

"""
def run_model(model, train, in_valid, ood_valid):
    model = model
"""
 
def training_roop(train, in_valid, ood_valid): # 途中
    train = train_loder
    in_valid = in_valid
    ood_valid = ood_valid
    
    train_iter = next(iter(train))
    in_valid_iter = next(iter(in_valid))
    ood_valid_iter = next(iter(ood_valid))
    
    
training_roop(train, in_valid, ood_valid)