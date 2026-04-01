import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import graph.GraphUIPRMD as GUI
import graph.GraphKIMORE as GKI
import graph.GraphSkate  as GSI
from Processing.data_processing_UIPRMD import UI_Data_Loader
from model.AdaptivescaleTimConv import  AdaptivescaleTimConv
from model.AdaptivescaleHyperConv import  AdaptivescaleHyperConv
from model.utils import conv_init, bn_init
from model.agc_layer import AGCLayer as InputEncoding, unit_tcn
from tqdm import  tqdm
import os









class ASTHCBlock(nn.Module):
    '''Selective Spatial Temporal Hypergraph Convolution Block.  图5b'''

    def __init__(self, in_channels, out_channels, A, G_part, G_body, num_point, num_frame, stride=1, pool_channels=None,residual=True, tcn_dropout=0.2, device='cpu'):
        super(ASTHCBlock, self).__init__()
        self.stride = stride
        self.gcn1 = AdaptivescaleHyperConv(in_channels, out_channels, A, G_part, G_body, num_point, num_frame, device=device)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=1,dropout=tcn_dropout)
        self.relu = nn.ReLU()
        if self.stride == 2:
            self.pool = AdaptivescaleTimConv(*pool_channels)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=1)



    def forward(self, x,matrix_flag = False):
        res = self.residual(x)
        if matrix_flag:
            x = self.gcn1(x,matrix_flag)
        else:
            x = self.gcn1(x)
        x = self.tcn1(x)


        N, C, T, V = x.size()
        if self.stride == 2:
            x, indices = self.pool(x, N)

            T_new = (T+1)//2
            pre_res = res.view(N, C, T, V)
            res = torch.empty((N, C, T_new, V), device=x.device)
            for i in range(N):
                res[i] = pre_res[i,:,indices[i],:]
            res = res.view(N, C, T_new, V)



        x = x + res
        return self.relu(x)


class Module(nn.Module):
    '''Selective-HCN Model.'''

    def __init__(self, dataset_name='Kimore_ex2', in_channels=3,tcn_dropout=0.2, model_dropout=0.25, device='cpu'):
        super(Module, self).__init__()
        self.device = device

        if 'UI_PRMD' in dataset_name:
            self.graph = GUI.GraphUIPRMD()
            self.hypergraph = GUI.Hypergraph()
            num_point = 39
            num_frame = 240
        elif 'Kimore' in dataset_name:
            self.graph = GKI.GraphKIMORE()
            self.hypergraph = GKI.Hypergraph()
            num_point = 25
            num_frame = 100
        elif 'MIT_Skate' in dataset_name or 'MITSkate' in dataset_name:
            self.graph = GSI.GraphMITSkate()
            self.hypergraph = GSI.Hypergraph()
            num_point = 25
            num_frame = 4500

        A = self.graph.A
        #bias_mat = self.graph.bias_mat
        G_part = self.hypergraph.G_part
        G_body = self.hypergraph.G_body
        self.data_bn = nn.BatchNorm1d(in_channels * num_point).to(device)
        #self.data_bn.cuda()
        self.drop = nn.Dropout(model_dropout)
        self.relu = nn.ReLU()




        # #KIMORE数据集
        self.l1 = InputEncoding(3, 64, A, residual=True, device=device)
        self.l2 = ASTHCBlock(64, 64, A, G_part, G_body, num_point, num_frame,tcn_dropout=tcn_dropout, device=device)
        self.l3 = ASTHCBlock(64, 64, A, G_part, G_body, num_point, num_frame,tcn_dropout=tcn_dropout, device=device)
        self.l4 = ASTHCBlock(64, 64, A, G_part, G_body, num_point, num_frame,tcn_dropout=tcn_dropout, device=device)
        #self.l5 = ASTHCBlock(64, 128, A, G_part, G_body, num_point, 100, stride=2, pool_channels=[128, 100])  # 选帧  这个函数调用后会进入SeleframeTemConv
        self.l5 = ASTHCBlock(64, 128, A,  G_part, G_body, num_point, num_frame,tcn_dropout=tcn_dropout, device=device)
        self.l6 = ASTHCBlock(128, 128, A, G_part, G_body, num_point, num_frame,tcn_dropout=tcn_dropout, device=device)
        self.l7 = ASTHCBlock(128, 128, A,  G_part, G_body, num_point, num_frame,tcn_dropout=tcn_dropout, device=device)
        #self.l8 = SelectSTHCBlock(128, 256, A, G_part, G_body, num_point, 50, stride=2, pool_channels=[256, 50])  # 选帧
        self.l8 = ASTHCBlock(128, 256, A, G_part, G_body, num_point, num_frame,tcn_dropout=tcn_dropout, device=device)
        self.l9 = ASTHCBlock(256, 256, A, G_part, G_body, num_point, num_frame,tcn_dropout=tcn_dropout, device=device)
        self.l10 = ASTHCBlock(256, 256, A, G_part, G_body, num_point, num_frame,tcn_dropout=tcn_dropout, device=device)


        #TCN
        self.conv2d1 = nn.Conv2d(256, 16, (9, 1), padding=(4, 0))
        self.conv2d2 = nn.Conv2d(16, 16, (15, 1), padding=(7, 0))
        self.conv2d3 = nn.Conv2d(16, 16, (15, 1), padding=(7, 0))

        #LSTM
        self.lstm = nn.LSTM(input_size=6400,hidden_size=256,num_layers=1,batch_first=True,bias=True).to(device)
        self.gru = nn.GRU(input_size=6400,hidden_size=256,num_layers=1,batch_first=True,bias=True).to(device)
        self.actv = nn.ReLU()
        self.fc2 = nn.Linear(48,256)
        self.fc3 = nn.Linear(128,128)
        self.fc = nn.Linear(256, 1)

        bn_init(self.data_bn, 1)


    def Lstm(self, x):
        x = x.contiguous().view(x.shape[0],x.shape[1], x.shape[2]*x.shape[3])
        x = nn.functional.normalize(x,p=1,dim=2)


        rec,(h0,c0)= nn.LSTM(input_size=75,hidden_size=256,num_layers=1,batch_first=True,bias=True).to(self.device)(x)
        h0 = torch.as_tensor(h0).to(torch.float32)
        c0 = torch.as_tensor(c0).to(torch.float32)
        h0 = h0.permute(1, 0, 2)
        c0 = c0.permute(1, 0, 2)
        rec3 = rec[:,-1,:]
        h0 = h0[:,-1,:]
        c0 = c0[:,-1,:]
        return rec3,h0,c0



    def forward(self, x, deteact_flag=False,matrix_flag = False):
        """
        Args:
            x: 输入张量，形状为 (N, T, V, C)
               N - 批次大小
               T - 时间步数/帧数
               V - 关节数量
               C - 通道数/坐标维度
            detect_flag: 是否输出特征可视化
            matrix_flag: 是否使用矩阵标志
        Returns:
            output: 预测分数，形状为 (N, 1)
        """

        x = torch.as_tensor(x).to(torch.float32)
        N, T, V, C = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N,  V * C, T).to(self.device)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 1, 3, 2).contiguous().view(N, C, T, V).to(self.device)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)


        x = x.permute(0,2,3,1)
        x = x.reshape(x.shape[0],x.shape[1], x.shape[2]*x.shape[3])
        x = nn.functional.normalize(x, p=1, dim=2)
        x,h0 = self.gru(x)
        x = x.sum(1)



        c_new = x.size(1)
        x = x.view(N, c_new, -1)
        x = x.mean(2)

        return self.fc(x)



if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data_loaderUI = UI_Data_Loader("UI_PRMD_ex1")
    train_x, test_x, train_y, test_y = train_test_split(data_loaderUI.scaled_x, data_loaderUI.scaled_y, test_size=0.2,random_state=42)
    #process = tqdm(train_x)
    train = Module()
    #print(train.parameters())
    train.forward(train_x)

