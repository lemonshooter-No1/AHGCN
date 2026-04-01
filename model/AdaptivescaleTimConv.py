import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *

#此类对应论文中的Selective-frame Temporal Convolution
class AdaptivescaleTimConv(nn.Module):
    '''Selective-frame Temporal Convolution (STC).'''

    def __init__(self, in_spa, in_tem, ratio=0.5):
        super(AdaptivescaleTimConv, self).__init__()
        self.in_spa = in_spa            #pool_channels[in_spa,in_tem]
        self.in_tem = in_tem
        self.ratio = ratio  # ratio=0.5 前50%
        self.ch_reduce = nn.Sequential(         #对应convs 降低时间维度
            nn.Conv2d(in_spa, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.spa_reduce = nn.Sequential(        #对应convc 降低特征维度
            nn.Conv2d(25, 1, 1),                #KIMORE
            #nn.Conv2d(39, 1, 1),               #UI_PRMD
            #nn.Conv2d(18, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

        self.fc_tem = nn.Sequential(                #多层感知机
            nn.Linear(in_tem, in_tem),              #线性层
            nn.Tanh(),
            nn.Linear(in_tem, in_tem),
            nn.Tanh(),                              #激活函数
            nn.Linear(in_tem, in_tem),
        )
        #self.drop = nn.Dropout(p=0.8)
        self.drop = nn.Dropout(p=0.25)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.Linear):
                fc_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x_in, N):  # forward函数 model(data)直接调用
        N, C, T, V = x_in.size()
        x_in = x_in.view(N, C, T, V)
        x = x_in
        #x = x.mean(dim=1)  # N C T V                #x(72,128,240,39)             #在第一维上取均值

        x = self.ch_reduce(x)                       #降低时间维度    convc
        x = x.permute(0, 3, 2, 1).contiguous()  # permute交换维度  contiguous相当于深拷贝
        x = self.spa_reduce(x)                      #降低特征维度    convs
        x = x.view(N, -1)  # 维度变成(N,总维度/N)                  #(72,240)

        # attention.
        x = self.fc_tem(x)    #三个线性 两个激活                      MLP

        #x = self.drop(x)

        scores = self.sigmoid(x)  # N T                         #式(9)

        # Top-k selection.
        T_new = int(T * self.ratio)
        top_scores, indices = scores.topk(T_new, dim=1, largest=True, sorted=False)
        top_scores = top_scores.unsqueeze(1).unsqueeze(-1)
        if T_new < T:
            x_out = torch.empty((N, C, T_new, V), device=x_in.device)
            for i in range(N):
                x_out[i] = x_in[i, :, indices[i], :]                        #x_out(32,128,50,25)
        else:
            x_out = x_in  # x_out是所选特征

        x_out = (x_out * top_scores).view(N, C, T_new, V)

        return x_out, indices


