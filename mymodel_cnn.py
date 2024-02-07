import torch 
from torch import nn

# use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1000, 1000), #(1000, 1000)に
            nn.ReLU(),
            nn.Linear(1000, 32), # (1000, (クラスの数 = 32))
        )
            
        
    
    def forward(self, xb):
        out = self.conv_net(xb)
        out = torch.max(out, dim = 2) # 最大プーリング(各配列ごとの最大値をとる)
        out = out.values
        # この後にlinear_relu_stackにoutを送る
        out = self.linear_relu_stack(out)
        # print("out3: ", out)
        return out
    