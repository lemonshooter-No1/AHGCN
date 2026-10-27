import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *


# Frame weighting and chronological Top-k selection.
class AdaptivescaleTimConv(nn.Module):
    '''Selective-frame Temporal Convolution (STC).'''

    def __init__(self, in_spa, in_tem, ratio=0.5, num_point=25):
        super(AdaptivescaleTimConv, self).__init__()
        if not 0 < ratio <= 1:
            raise ValueError('Frame retention ratio must be in (0, 1].')
        self.in_spa = in_spa            #pool_channels[in_spa,in_tem]
        self.in_tem = in_tem
        self.ratio = ratio
        self.ch_reduce = nn.Sequential(
            nn.Conv2d(in_spa, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.spa_reduce = nn.Sequential(
            nn.Conv2d(num_point, 1, 1),                #KIMORE
            #nn.Conv2d(39, 1, 1),               #UI_PRMD
            #nn.Conv2d(18, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

        self.fc_tem = nn.Sequential(
            nn.Linear(in_tem, in_tem),
            nn.Tanh(),
            nn.Linear(in_tem, in_tem),
            nn.Tanh(),
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

    def forward(self, x_in, N=None, features=None):
        N, C, T, V = x_in.size()
        x_in = x_in.view(N, C, T, V)
        x = x_in


        x = self.ch_reduce(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.spa_reduce(x)
        x = x.view(N, -1)

        # attention.
        if T != self.in_tem:
            x = F.interpolate(x.unsqueeze(1), size=self.in_tem, mode='linear', align_corners=False).squeeze(1)
        x = self.fc_tem(x)
        if T != self.in_tem:
            x = F.interpolate(x.unsqueeze(1), size=T, mode='linear', align_corners=False).squeeze(1)
        scores = self.sigmoid(x)
        T_new = max(1, int(T * self.ratio))
        indices = scores.topk(T_new, dim=1, largest=True, sorted=False).indices.sort(dim=1).values
        top_scores = scores.gather(1, indices).unsqueeze(1).unsqueeze(-1)
        source = x_in if features is None else features
        selected = indices[:, None, :, None].expand(-1, source.size(1), -1, V)
        x_out = source.gather(2, selected) * top_scores

        return x_out, indices




# Temporal hypergraph aggregation and channel projection.
class TemporalHyperConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(TemporalHyperConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x, H, prepared=None):
        from graph.Hypergraph import temporal_aggregate
        return self.drop(self.bn(self.conv(temporal_aggregate(x, H, prepared))))
