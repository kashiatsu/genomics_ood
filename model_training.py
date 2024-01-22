import torch
import numpy as np
import pandas as pd 
from torch import nn
import os
from pathlib import Path
from tfrecord.tools import tfrecord2idx
import mymodel_cnn

# use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# データセットの読み込み
DATASET_ROOT = Path('data/llr_ood_genomics')
id_train_path = DATASET_ROOT / "before_2011_in_tr"
id_val_path = DATASET_ROOT / "between_2011-2016_in_val"
id_test_path = DATASET_ROOT / "after_2016_in_test"
ood_val_path = DATASET_ROOT / "between_2011-2016_ood_val"
ood_test_path = DATASET_ROOT / "after_2016_ood_test"

splits = {
    "train": id_train_path,
    "val": id_val_path,
    "test": id_test_path,
    "val_ood": ood_val_path,
    "test_ood": ood_test_path
}

for i in range (0, 1):
    train = open(id_train_path / 'before_2011_in_tr-0000{}-of-00010.tfrecord'.format(i))
    id_val = open(id_val_path / 'between_2011-2016_in_val-0000{}-of-00010.index'.format(i))
    ood_val = open(ood_val_path / 'between_2011-2016_ood_val-0000{}-of-00010.index'.format(i))
    id_test = open(id_test_path / 'after_2016_in_test-0000{}-of-00010.index'.format(i))
    ood_test = open(ood_test_path / 'after_2016_ood_test-0000{}-of-00010.index'.format(i))
    # どのようにデータを抜き取るか?
    l = train.readlines()
    print(type(l))
    # print(l)
    
    
    
    
        