import torch 
from torch import nn

class model_CNN(nn.Module): # 途中
    def __init__(self,
                 seq_len,
                 num_filters = 1000,
                 kernel_size = 20):
        super().__init__()
        self.seq_len = seq_len

        # 畳み込み層
        self.conv_net = nn.Sequential(
            nn.Conv1d(4, num_filters, kernel_size = kernel_size), # 畳み込み(長さ20の畳み込みフィルタを1000枚), ゼロパディング, stride = 1
            nn.ReLU(inplace = True)
        )
        
        # Flatten + Linear(全結合層)
        self.Flatten_Linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters * (seq_len - kernel_size + 1), 1000) # 全結合(入力: , 出力: 1000ニューロン)
        )

            # nn.MaxPool1d(kernel_size = kernel_size), # 最大プーリング(kernel_sizeは一緒じゃなくてもいい), 最大じゃなくていい
            # torch.max()
            # nn.Flatten(),
            # nn.Linear(num_filters * (seq_len - kernel_size + 1), 1000) # 全結合(入力: , 出力: 1000ニューロン)
            
        
    
    def forward(self, xb):
        xb = xb.permute(0, 2, 1) # 次元の並び替え(今度決める)
        out = self.conv_net(xb)
        print("out: ", out.shape())
        out = torch.max(out, dim = 1) # 最大プーリング(各配列ごとの最大値をとる)
        # この後にFlatten_Linearにoutを送る
        out = self.Flatten_Linear(out)
        print("out: ", out.shape())
        return out
    