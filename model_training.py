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
from sklearn.preprocessing import OneHotEncoder

# use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# データの読み込み
train = ood_genomics_dataset.OODGenomicsDataset("data", "train") # before_2011_in_tr
in_valid = ood_genomics_dataset.OODGenomicsDataset("data", "val") # between_2011-2016_in_valid
ood_valid = ood_genomics_dataset.OODGenomicsDataset("data", "val_ood") #between_2011-2016_ood_valid
in_test = ood_genomics_dataset.OODGenomicsDataset("data", "test") # after_2016_in_test
ood_test = ood_genomics_dataset.OODGenomicsDataset("data", "test_ood") #after_2016_ood_test

# batch_size
BATCH_SIZE = 32

# Dataloder
train_dl = DataLoader(train, batch_size = BATCH_SIZE)
in_valid_dl = DataLoader(in_valid, batch_size = BATCH_SIZE)
ood_valid_dl = DataLoader(ood_valid, batch_size = BATCH_SIZE)

# モデルと損失関数

# one-hot-encoding
def one_hot_encoding(data_seq_minibatch):
    data_seq_minibatch = data_seq_minibatch.to(torch.int64) # int型に変換
    
    data_seq_one_hot = nn.functional.one_hot(data_seq_minibatch)
    data_seq_one_hot = torch.transpose(data_seq_one_hot, 1, 2)
    # print("one_hot: ", data_seq_one_hot.shape)
    data_seq_one_hot = data_seq_one_hot.to(torch.float32) # floatに戻す
    return data_seq_one_hot

def make_train_loss(train_dl, epochs, lr):
    train_losses = [] #trainの損失を格納するリスト
    
    for epoch, data in enumerate(train_dl):
        # Every data instance is an input + label pair
        train_seq_minibatch, train_label_minibatch = data # train
        train_seq_minibatch = train_seq_minibatch.to(DEVICE) # deviceを統一
        train_label_minibatch = train_label_minibatch.to(DEVICE)
        train_label_minibatch = train_label_minibatch.to(torch.float32)
        
        train_seq_minibatch = one_hot_encoding(train_seq_minibatch)
        
        model = mymodel_cnn.model_CNN(train_seq_minibatch.shape[2])
        model.to(DEVICE)
        # print("model: ", model)
        
        # optimizer
        optimizer = torch.optim.AdamW(params = model.parameters(),
                              lr = lr, weight_decay = 0.01)
        # loss function
        loss_function = torch.nn.MSELoss()
        
        # Zero your gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        train_outputs = model.forward(train_seq_minibatch)
        # print("outputs: ", train_outputs[0])

        # Compute the loss and its gradients
        train_loss = loss_function(train_outputs[0], train_label_minibatch)
        train_losses.append(train_loss)
        train_loss.backward()

        # Adjust learning weights
        optimizer.step()
        
        # print(f"E{epoch} | train loss: {train_loss:.3f}")
    
    return train_losses

def make_valid_loss(in_valid_dl, ood_valid_dl, epochs, lr):
    in_valid_losses = []
    ood_valid_losses = []
    
    for epoch, data in enumerate(in_valid_dl): # in_valid
        # Every data instance is an input + label pair
        in_valid_seq_minibatch, in_valid_label_minibatch = data 
        in_valid_seq_minibatch = in_valid_seq_minibatch.to(DEVICE) # deviceを統一
        in_valid_label_minibatch = in_valid_label_minibatch.to(DEVICE)
        in_valid_label_minibatch = in_valid_label_minibatch.to(torch.float32)
        
        in_valid_seq_minibatch = one_hot_encoding(in_valid_seq_minibatch) # one_hot
        
        model = mymodel_cnn.model_CNN(in_valid_seq_minibatch.shape[2])
        model.to(DEVICE)
        
        # optimizer
        optimizer = torch.optim.AdamW(params = model.parameters(),
                              lr = lr, weight_decay = 0.01)
        # loss function
        loss_function = torch.nn.MSELoss()
        
        # Zero your gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        in_valid_outputs = model.forward(in_valid_seq_minibatch)

        # Compute the loss and its gradients
        in_valid_loss = loss_function(in_valid_outputs[0], in_valid_label_minibatch)
        
        in_valid_losses.append(in_valid_loss)
        in_valid_loss.backward()
        
        # Adjust learning weights
        optimizer.step()
        
        # print(f"E{epoch} | in valid loss: {in_valid_loss:.3f}")

    
    for epoch, data in enumerate(ood_valid_dl): # ood_valid
        # Every data instance is an input + label pair
        ood_valid_seq_minibatch, ood_valid_label_minibatch = data 
        ood_valid_seq_minibatch = ood_valid_seq_minibatch.to(DEVICE) # deviceを統一
        ood_valid_label_minibatch = ood_valid_label_minibatch.to(DEVICE)
        ood_valid_label_minibatch = ood_valid_label_minibatch.to(torch.float32)
        
        ood_valid_seq_minibatch = one_hot_encoding(ood_valid_seq_minibatch) # one_hot
        
        model = mymodel_cnn.model_CNN(ood_valid_seq_minibatch.shape[2])
        model.to(DEVICE)
        
        # optimizer
        optimizer = torch.optim.AdamW(params = model.parameters(),
                              lr = lr, weight_decay = 0.01)
        # loss function
        loss_function = torch.nn.MSELoss()
        
        # Zero your gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        ood_valid_outputs = model.forward(ood_valid_seq_minibatch)

        # Compute the loss and its gradients
        ood_valid_loss = loss_function(ood_valid_outputs, ood_valid_label_minibatch)
        
        ood_valid_losses.append(ood_valid_loss)
        ood_valid_loss.backward()
        
        # Adjust learning weights
        optimizer.step()
        
        # print(f"E{epoch} | ood valid loss: {ood_valid_loss:.3f}")


    return in_valid_losses, ood_valid_losses    

def run_model(train_dl, in_valid_dl, ood_valid_dl, epochs, lr):
    # 損失の配列
    train_losses = []
    in_valid_losses = []
    ood_valid_losses = []
    
    # train_loss
    train_losses = make_train_loss(train_dl, epochs, lr)
    # in_valid_loss, ood_valid_loss
    in_valid_losses, ood_valid_losses = make_valid_loss(in_valid_dl, ood_valid_dl, epochs, lr)
    
    return train_losses, in_valid_losses, ood_valid_losses
    
    assert False
        
    # for epoch in range(epochs):
        # train_seq = next(iter(train_dl))
    
    for epoch, data in enumerate(train_dl):
        # Every data instance is an input + label pair
        train_seq_minibatch, train_label_minibatch = data # train

        
        train_seq_minibatch = one_hot_encoding(train_seq_minibatch)
        # print("mini: ", train_seq_minibatch)
        # print("ここまでOK")
        
        assert False
        
        # print("train_seq1: ", train_seq.shape)
        # print("train_label: ", train_label)
        # train_loss_functionとvalid_loss_functionを作る
        train_seq_pred = model(train_seq_minibatch)
        train_loss = loss_function(train_seq_pred, train_label_minibatch)
        
        # in_valid_loss = loss_function(in_valid_dl)
        # ood_valid_loss = loss_function(ood_valid_dl)
        
        optimizer.zero_grad() # 勾配を初期化
        train_loss.backward() # 勾配を計算
        optimizer.step() # パラメータを更新
        
        train_losses.append(train_loss)
        in_valid_losses.append(in_valid_loss)
        ood_valid_losses.append(ood_valid_loss)
        print(f"E{epoch} | train loss: {train_loss:.3f} | in valid loss: {in_valid_loss:.3f} | ood valid loss: {ood_valid_loss:.3f}")
        
    return train_losses, in_valid_losses, ood_valid_losses

def training_loop(train_dl, in_valid_dl, ood_valid_dl): # 途中
    lr = 0.001 # 学習率
    epochs = 10 # エポック数
    
    train_losses, in_valid_losses, ood_valid_losses = run_model(train_dl, in_valid_dl, ood_valid_dl, epochs, lr)
    return train_losses, in_valid_losses, ood_valid_losses
    
    
train_losses, valid_losses = training_loop(train_dl, in_valid_dl, ood_valid_dl)