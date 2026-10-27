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
from model.AdaptivescaleTimConv import AdaptivescaleTimConv, TemporalHyperConv
from graph.Hypergraph import Hypergraph, prepare_temporal
from model.AdaptivescaleHyperConv import  AdaptivescaleHyperConv
from model.utils import conv_init, bn_init
from model.agc_layer import AGCLayer as InputEncoding, unit_tcn
from tqdm import  tqdm
import os









# Spatial-temporal hypergraph block with optional frame pooling.
class ASTHCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, G_part, G_body, num_point, num_frame,
                 stride=1, pool_channels=None, residual=True, tcn_dropout=0.0, device='cpu', topk_ratio=0.5):
        super(ASTHCBlock, self).__init__()
        self.stride = stride
        self.gcn1 = AdaptivescaleHyperConv(in_channels, out_channels, A, G_part, G_body,
                                          num_point, num_frame, device=device)
        self.tcn1 = TemporalHyperConv(out_channels, out_channels, dropout=tcn_dropout)
        self.relu = nn.ReLU()
        self.residual_enabled = residual
        if self.stride == 2:
            self.pool = AdaptivescaleTimConv(out_channels, num_frame, ratio=topk_ratio, num_point=num_point)

    def forward(self, x, H, G_interjoint, matrix_flag=False, prepared=None):
        spatial = self.gcn1(x, matrix_flag, G_interjoint)
        temporal = self.tcn1(spatial, H, prepared)
        residual = spatial if self.residual_enabled else torch.zeros_like(spatial)
        if self.stride == 2:
            temporal, indices = self.pool(spatial, features=temporal)
            selected = indices[:, None, :, None].expand(-1, residual.size(1), -1, residual.size(3))
            residual = residual.gather(2, selected)
            H = Hypergraph.restrict(H, indices, spatial.size(3))
        return self.relu(temporal + residual), H


# AHGCN backbone and sequence regression head.
class Module(nn.Module):
    '''Selective-HCN Model.'''

    def __init__(self, dataset_name='Kimore_ex2', in_channels=3,tcn_dropout=0.0, model_dropout=0.25, device='cpu', num_center=5, sparse_lambda=0.01,
                 sparse_threshold=0.001, sparse_iter=1000, sparse_tol=1e-6, temporal_k=4,
                 topk_ratio=0.5, num_frame=None, gru_norm=1, init_variant='original', use_topk=False):
        super(Module, self).__init__()
        self.device = device
        if gru_norm not in (1, 2):
            raise ValueError('gru_norm must be 1 or 2.')
        self.gru_norm = gru_norm
        if init_variant not in ('original', 'spatial_open', 'temporal_identity', 'gru_memory', 'spatial_gru', 'all'):
            raise ValueError('Unsupported init_variant.')
        frame_override = num_frame
        self.hypergraph_config = dict(num_center=num_center, sparse_lambda=sparse_lambda,
                                     sparse_threshold=sparse_threshold, sparse_iter=sparse_iter,
                                     sparse_tol=sparse_tol, temporal_k=temporal_k,
                                     topk_ratio=topk_ratio, num_frame=num_frame, gru_norm=gru_norm, init_variant=init_variant, use_topk=use_topk)
        self.structure = Hypergraph(num_center, sparse_lambda, sparse_threshold,
                                    sparse_iter, sparse_tol, temporal_k)

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
            num_point = 18
            num_frame = 4500

        else:
            raise ValueError(f'Unsupported dataset: {dataset_name}')
        if frame_override is not None:
            num_frame = frame_override
        if num_frame < 2 or not 0 < topk_ratio <= 1:
            raise ValueError('Invalid sequence length or frame retention ratio.')
        A = self.graph.A
        #bias_mat = self.graph.bias_mat
        G_part = self.hypergraph.G_part
        G_body = self.hypergraph.G_body
        self.data_bn = nn.BatchNorm1d(in_channels * num_point).to(device)
        #self.data_bn.cuda()
        self.drop = nn.Dropout(model_dropout)
        self.relu = nn.ReLU()




        self.l1 = InputEncoding(in_channels, 64, A, residual=True, device=device)
        channels = [(64, 64), (64, 64), (64, 64), (64, 128), (128, 128),
                    (128, 128), (128, 256), (256, 256), (256, 256)]
        for block, (input_channels, output_channels) in enumerate(channels, 1):
            stride = 2 if use_topk and block in (4, 7) else 1
            layer = ASTHCBlock(input_channels, output_channels, A, G_part, G_body,
                              num_point, num_frame, stride=stride, tcn_dropout=tcn_dropout,
                              device=device, topk_ratio=topk_ratio)
            setattr(self, f'l{block + 1}', layer)
            if stride == 2:
                num_frame = max(1, int(num_frame * topk_ratio))

        #TCN
        self.conv2d1 = nn.Conv2d(256, 16, (9, 1), padding=(4, 0))
        self.conv2d2 = nn.Conv2d(16, 16, (15, 1), padding=(7, 0))
        self.conv2d3 = nn.Conv2d(16, 16, (15, 1), padding=(7, 0))

        #LSTM
        self.lstm = nn.LSTM(input_size=6400,hidden_size=256,num_layers=1,batch_first=True,bias=True).to(device)
        self.gru = nn.GRU(input_size=256 * num_point,hidden_size=256,num_layers=1,batch_first=True,bias=True).to(device)
        self.actv = nn.ReLU()
        self.fc2 = nn.Linear(48,256)
        self.fc3 = nn.Linear(128,128)
        self.fc = nn.Linear(256, 1)

        bn_init(self.data_bn, 1)
        if init_variant in ('spatial_open', 'spatial_gru', 'all'):
            for block in range(2, 11):
                bn_init(getattr(self, f'l{block}').gcn1.bn, 1)
        if init_variant in ('temporal_identity', 'all'):
            for block in range(2, 11):
                bn_init(getattr(self, f'l{block}').tcn1.bn, 0)
        if init_variant in ('gru_memory', 'spatial_gru', 'all'):
            for weight in self.gru.weight_ih_l0.chunk(3, dim=0):
                nn.init.xavier_uniform_(weight)
            for weight in self.gru.weight_hh_l0.chunk(3, dim=0):
                nn.init.orthogonal_(weight)
            nn.init.zeros_(self.gru.bias_ih_l0)
            nn.init.zeros_(self.gru.bias_hh_l0)
            with torch.no_grad():
                self.gru.bias_ih_l0[self.gru.hidden_size:2 * self.gru.hidden_size].fill_(1)


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



    def forward(self, x, deteact_flag=False,matrix_flag = False, hypergraphs=None):


        x = torch.as_tensor(x).to(torch.float32)
        N, T, V, C = x.size()
        x_input = x.to(self.data_bn.weight.device)
        x = x_input.permute(0, 2, 3, 1).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        coordinates = x_input
        if hypergraphs is None:
            G_interjoint = self.structure.spatial(coordinates)
            H = self.structure.temporal(coordinates)
        else:
            G_interjoint, H = hypergraphs
            if G_interjoint.shape != (N, V, V) or len(H) != N:
                raise ValueError('Cached hypergraphs do not match the input batch.')
            if any(graph.shape != (T * V, T * V) for graph in H):
                raise ValueError('Cached temporal hypergraphs do not match the input length.')
        x = self.l1(x)
        prepared = prepare_temporal(H)
        for block in range(2, 11):
            layer = getattr(self, f'l{block}')
            x, H = layer(x, H, G_interjoint, matrix_flag, prepared)
            if layer.stride == 2:
                prepared = prepare_temporal(H)

        x = x.permute(0,2,3,1)
        x = x.reshape(x.shape[0],x.shape[1], x.shape[2]*x.shape[3])
        x = nn.functional.normalize(x, p=self.gru_norm, dim=2)
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

