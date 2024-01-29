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

# ラベル
train_label = train.label_dict
in_valid_label = in_valid.label_dict
ood_valid_label = ood_valid.label_dict
in_test_label = in_test.label_dict
ood_test_label = ood_test.label_dict

train_dl = DataLoader(train, batch_size = 32)
in_valid_dl = DataLoader(in_valid, batch_size = 32)
ood_valid_dl = DataLoader(ood_valid, batch_size = 32)

# モデルと損失関数

# train用の損失関数
def train_loss_function(train_dl, model, loss_function, epochs, optimizer):
    a

def run_model(train_dl, in_valid_dl, ood_valid_dl, model, lr, epochs):
    # optimizer
    optimizer = torch.optim.AdamW(params = model.parameters(),
                              lr = lr, weight_decay = 0.01)
    # loss function
    loss_function = torch.nn.MSELoss()
    
    # 損失関数の配列
    train_losses = []
    in_valid_losses = []
    ood_valid_losses = []
    
    print("train_seq: ", next(iter(train))[0][0])
    
    for epoch in range(epochs):
        # train_loss_functionとvalid_loss_functionを作る
        train_loss = loss_function(train_dl)
        in_valid_loss = loss_function(in_valid_dl)
        ood_valid_loss = loss_function(ood_valid_dl)
        
        optimizer.zero_grad() # 勾配を初期化
        train_loss.backward() # 勾配を計算
        optimizer.step() # パラメータを更新
        
        train_losses.append(train_loss)
        in_valid_losses.append(in_valid_loss)
        ood_valid_losses.append(ood_valid_loss)
        print(f"E{epoch} | train loss: {train_loss:.3f} | in valid loss: {in_valid_loss:.3f} | ood valid loss: {ood_valid_loss:.3f}")
        
        return train_losses, in_valid_losses, ood_valid_losses

def training_roop(train_dl, in_valid_dl, ood_valid_dl): # 途中
    seq_len = len(next(iter(train_dl))[0][0])
    model = mymodel_cnn.model_CNN(seq_len)
    model.to(DEVICE)
    lr = 0.001 # 学習率
    epochs = 10 # エポック数
    
    train_losses, in_valid_losses, ood_valid_losses = run_model(train_dl, in_valid_dl, ood_valid_dl, model, lr, epochs)
    return train_losses, in_valid_losses, ood_valid_losses
    
    
train_losses, valid_losses = training_roop(train_dl, in_valid_dl, ood_valid_dl)