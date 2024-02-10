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
import matplotlib.pyplot as plt
import GPUtil

def get_most_freemem_gpu(): # 一番空いているGPUを使うようにする
    max_mem_ = 0
    max_id_ = 0
    i = 0 
    for g in GPUtil.getGPUs():
        if g.memoryFree > max_mem_ :
            max_mem_ = g.memoryFree
            max_id_ = i
        i += 1
    return(max_id_)

DEVICE = get_most_freemem_gpu()
print("device = ", DEVICE)

torch.cuda.set_device(DEVICE)

# データの読み込み
train = ood_genomics_dataset.OODGenomicsDataset("data", "train") # before_2011_in_tr
in_valid = ood_genomics_dataset.OODGenomicsDataset("data", "val") # between_2011-2016_in_valid
ood_valid = ood_genomics_dataset.OODGenomicsDataset("data", "val_ood") #between_2011-2016_ood_valid
in_test = ood_genomics_dataset.OODGenomicsDataset("data", "test") # after_2016_in_test
ood_test = ood_genomics_dataset.OODGenomicsDataset("data", "test_ood") #after_2016_ood_test

# batch_size
BATCH_SIZE = 32

# epochs
EPOCHS = 10

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

def make_train_loss_one_epoch(train_dl, lr):
    train_losses = [] #trainの損失を格納するリスト
    
    model = mymodel_cnn.model_CNN(250) # 250 = train_seq_minibatch.shape[2]
    model.to(DEVICE)
    
    # optimizer
    optimizer = torch.optim.AdamW(params = model.parameters(),
                                   lr = lr, weight_decay = 0.01)
    # loss function
    loss_function = torch.nn.MSELoss()
    
    for i, data in enumerate(train_dl):
        # Every data instance is an input + label pair
        train_seq_minibatch, train_label_minibatch = data # train
        train_seq_minibatch = train_seq_minibatch.to(DEVICE) # deviceを統一
        train_label_minibatch = train_label_minibatch.to(DEVICE)
        train_label_minibatch = train_label_minibatch.to(torch.float32)
        
        train_seq_minibatch = one_hot_encoding(train_seq_minibatch)
        # print("seq[2]", train_seq_minibatch.shape[2])
        
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
        
        if i % 5000 == 0:
            print(f"E{i} | train loss: {train_loss:.3f}")
    
    print("train_loss is OK!")
    return train_losses

def make_valid_loss_one_epoch(in_valid_dl, ood_valid_dl, lr):
    in_valid_losses = []
    ood_valid_losses = []
    
    model = mymodel_cnn.model_CNN(250)
    model.to(DEVICE)
    
    # optimizer
    optimizer = torch.optim.AdamW(params = model.parameters(),
                            lr = lr, weight_decay = 0.01)
    # loss function
    loss_function = torch.nn.MSELoss()
    
    for i, data in enumerate(in_valid_dl): # in_valid
        # Every data instance is an input + label pair
        in_valid_seq_minibatch, in_valid_label_minibatch = data 
        in_valid_seq_minibatch = in_valid_seq_minibatch.to(DEVICE) # deviceを統一
        in_valid_label_minibatch = in_valid_label_minibatch.to(DEVICE)
        in_valid_label_minibatch = in_valid_label_minibatch.to(torch.float32)
        
        in_valid_seq_minibatch = one_hot_encoding(in_valid_seq_minibatch) # one_hot
        
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
        
        if i % 5000 == 0:
            print(f"E{i} | in valid loss: {in_valid_loss:.3f}")
        
    print("in_valid_loss is OK!")


    # ood_valid
    model = mymodel_cnn.model_CNN(ood_valid_seq_minibatch.shape[2])
    model.to(DEVICE)
    
    # optimizer
    optimizer = torch.optim.AdamW(params = model.parameters(),
                            lr = lr, weight_decay = 0.01)
    # loss function
    loss_function = torch.nn.MSELoss()
    
    for i, data in enumerate(ood_valid_dl): # ood_valid
        # Every data instance is an input + label pair
        ood_valid_seq_minibatch, ood_valid_label_minibatch = data 
        ood_valid_seq_minibatch = ood_valid_seq_minibatch.to(DEVICE) # deviceを統一
        ood_valid_label_minibatch = ood_valid_label_minibatch.to(DEVICE)
        ood_valid_label_minibatch = ood_valid_label_minibatch.to(torch.float32)
        
        ood_valid_seq_minibatch = one_hot_encoding(ood_valid_seq_minibatch) # one_hot
        
        
        
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
        
        if i % 5000 == 0:
            print(f"E{i} | ood valid loss: {ood_valid_loss:.3f}")

    print("ood_valid_loss is OK!")
    return in_valid_losses, ood_valid_losses    

def run_model(train_dl, in_valid_dl, ood_valid_dl, lr):
    # 損失の配列
    train_losses = []
    in_valid_losses = []
    ood_valid_losses = []
    
    # train_loss
    train_losses = make_train_loss_one_epoch(train_dl, lr)
    # in_valid_loss, ood_valid_loss
    in_valid_losses, ood_valid_losses = make_valid_loss_one_epoch(in_valid_dl, ood_valid_dl, lr)
    
    return train_losses, in_valid_losses, ood_valid_losses
    
    


    

def loss_plot(train_loss, in_valid_loss, ood_valid_loss, epochs):
    x = list(range(train_loss))
    y1 = train_loss
    y2 = in_valid_loss
    y3 = ood_valid_loss
    
    # plot
    fig, ax = plt.subplots()
    
    ax.grid(True)
    ax.set(xlim = (0, epochs), ylim = (0, 0.001)) 
    ax.set_xlabel('step_size')
    ax.set_ylabel('loss')
    ax.plot(x, y1, linewidth = 1.0, color = 'r', label = 'train_loss')
    ax.plot(x, y2, linewidth = 1.0, color = 'b', label = 'in_valid_loss')
    ax.plot(x, y3, linewidth = 1.0, color = 'g', label = 'ood_valid_loss')

    
    ax.legend()
    
    plt.savefig('loss_plot{}.png'.format(epochs), dpi=200, bbox_inches="tight", pad_inches=0.1)
    

def training_loop(train_dl, in_valid_dl, ood_valid_dl): # 途中
    lr = 0.001 # 学習率
    
    for epoch in range(EPOCHS):
        print("epoch: ", epoch + 1)
        train_losses, in_valid_losses, ood_valid_losses = run_model(train_dl, in_valid_dl, ood_valid_dl, lr)
        loss_plot(train_losses, in_valid_losses, ood_valid_losses, epoch)
    
    print("trainig conplete!")
    
training_loop(train_dl, in_valid_dl, ood_valid_dl)


