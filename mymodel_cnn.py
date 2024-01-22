import torch 
import numpy as np
import pandas as pd 
from torch import nn

class model_CNN(nn.Module): # 途中
    def __init__(self,
                 seq_len,
                 num_filters = 1000,
                 kernel_size = 20):
        super().__init__()
        self.seq_len = seq_len
    
        self.conv_net = nn.Sequential(
            nn.Conv1d(4, num_filters, kernel_size = kernel_size), # 畳み込み(長さ20の畳み込みフィルタを1000枚), ゼロパディング, stride = 1
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size = kernel_size), # 最大プーリング(kernel_sizeは一緒じゃなくてもいい)
            nn.Flatten(),
            nn.Linear(num_filters * (seq_len - kernel_size + 1), 1000) # 全結合(入力: , 出力: 1000ニューロン)
        )
    
    def forward(self, xb):
        xb = xb.permute(0, 2, 1) # ? (今度決める)
        out = self.conv_net(xb)
        
        return out
    