import torch
import torch.nn as nn
import math
import numpy as np

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')        #使用正态分布对输入张量进行赋值  fan_out 通过创建随机矩阵显示创建权重 fan_in 通过线性层(卷积或全连接)隐形创建
    nn.init.constant_(conv.bias, 0)

def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    if fc.bias is not None:
        nn.init.constant_(fc.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)             #用scale填充bn.weight
    nn.init.constant_(bn.bias, 0)                   #用0填充bn.bias


class PositionalEncoding(nn.Module):

    def __init__(self, channel, joint_num, time_len, domain):                   #in_channels, num_point, num_frame, 'spatial'
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        self.domain = domain

        if domain == "temporal":                        #时间
            # temporal embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(t)
        elif domain == "spatial":                       #空间
            # spatial embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        # pe = position/position.max()*2 -1
        # pe = pe.view(time_len, joint_num).unsqueeze(0).unsqueeze(0)
        # Compute the positional encodings once in log space.

        #Positional Encoding 位置编码器
        pe = torch.zeros(self.time_len * self.joint_num, channel)

        div_term = torch.exp(torch.arange(0, channel, 2).float() * -(math.log(10000.0) / channel))  # channel//2   arange(start,end,step)返回[(end-start)/step]的一维张量，以step为步长等间隔取值
        pe[:, 0::2] = torch.sin(position * div_term)                                #PE(pos,2i)
        pe[:, 1::2] = torch.cos(position * div_term)                                #PE(pos,2i+1)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)    #(channel,time_len,joint_num)
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = x + self.pe[:, :, :x.size(2)]
        return x