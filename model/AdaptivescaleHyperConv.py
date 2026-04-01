import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from model.utils import PositionalEncoding, conv_init, bn_init, conv_branch_init, fc_init


#这个类对应论文中的selective-scale hypergraph convolution
class AdaptivescaleHyperConv(nn.Module):
    '''Select-scale Hypergraph Convolution (SHC).   论文中的3.2'''

    def __init__(self, in_channels, out_channels, A, G_part, G_body, num_point, num_frame, coff_embedding=4,num_subset=3,device='cpu'): #coff_embeding一般是4或8
        super(AdaptivescaleHyperConv, self).__init__()
        self.device = device
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        #self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32))).cuda()  # 将不可训练的Tensor转换成可训练的parameter,经过类型转换这个变量就变成了模型的一部分，成为模型中根据训练可以改动的参数
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))  # 将不可训练的Tensor转换成可训练的parameter,经过类型转换这个变量就变成了模型的一部分，成为模型中根据训练可以改动的参数
        nn.init.constant_(self.PA, 1e-6)  # 10的负6次方

        self.G_part = nn.Parameter(torch.from_numpy(G_part.astype(np.float32))).to(device)
        self.G_body = nn.Parameter(torch.from_numpy(G_body.astype(np.float32))).to(device)
        #邻接矩阵
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False).to(device)
        self.num_subset = num_subset  # num_subset=3
        self.num_point = num_point

        self.conv_joint = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_joint.append(nn.Conv2d(in_channels, out_channels, 1))        #原代码
            #self.conv_joint.append(nn.Conv2d(in_channels, in_channels, 1))
            #self.conv_joint.append(nn.Conv2d(in_channels, 25, 1))

        #self.convmatch = nn.Conv2d(25, out_channels, 1)

        #self.PE = PositionalEncoding(in_channels, num_point, num_frame, 'spatial')        #原代码是spatial
        self.PE = PositionalEncoding(in_channels, num_point, num_frame, 'temporal')        #第二个修改
        self.conv_theta = nn.Conv2d(in_channels, inter_channels, 1)
        self.conv_phi = nn.Conv2d(in_channels, inter_channels, 1)
        self.conv_nonlocal = nn.Conv2d(in_channels, out_channels, 1)                      #原代码



        self.conv_part = nn.Conv2d(in_channels, out_channels, 1)                         #原代码
        self.conv_body = nn.Conv2d(in_channels, out_channels, 1)
        # self.conv_part = nn.Conv2d(in_channels, in_channels, 1)
        # self.conv_body = nn.Conv2d(in_channels, in_channels, 1)
        '''self.conv_avg = nn.Conv2d(in_channels, out_channels, 1)'''

        # selective-scale.
        self.num_branch = 3
        d = int(out_channels / 2)
        self.fc = nn.Linear(out_channels, d)
        #self.fc = nn.Linear(in_channels, out_channels)
        self.fc_branch = nn.ModuleList([])
        for i in range(3):                                                  #可以减少全连接的层数 之前是3    第一个修改
            self.fc_branch.append(
                nn.Linear(d, out_channels)
                #nn.Linear(out_channels, out_channels)
            )
        self.softmax = nn.Softmax(dim=1)
        #self.leaky_relu = nn.LeakyReLU()

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():  # 对不同的函数使用不同的初始化方式
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            # elif isinstance(m, nn.Linear):
            #     fc_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)  # m.weight=1,m.bias=0
        #bn_init(self.bn, 1e-6)  #原代码
        bn_init(self.bn, 0)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_joint[i], self.num_subset)

    def norm(self, A):
        D_list = torch.sum(A, 0).view(1, self.num_point)                        #对第0维求和之后变成(1,39)维矩阵 按行求和
        D_list_12 = (D_list + 0.001) ** (-1)                                    #平滑处理，平滑系数0.001，防止分母为0
        D_12 = torch.eye(self.num_point).to(device=A.device) * D_list_12        #eye对角矩阵 对角线上元素为D_list_l2
        A = torch.matmul(A, D_12)                                               #得到标准化邻接矩阵
        return A

    def forward(self, x, matrix_flag=False):
        device = x.device
        N, C, T, V = x.size()  # N视频数量 C关节特征 T视频帧数量 V关节数量
        #A = self.A.cuda(x.get_device())

        A = self.A
        A = A + self.PA


        # G_part = self.G_part.cuda(x.get_device())
        # G_body = self.G_body.cuda(x.get_device())
        # Avg = self.GloAvgMat.cuda(x.get_device())
        G_part = self.G_part.to(device)                 #这里导出邻接矩阵
        G_body = self.G_body.to(device)
        #Avg = self.GloAvgMat


        if matrix_flag:
            np_Gpart = G_part.cpu().detach().numpy()
            np_Gbody = G_body.cpu().detach().numpy()
            np_A = A[1].cpu().detach().numpy()
            np.savetxt('G_part.txt', np_Gpart)
            np.savetxt('G_body.txt', np_Gbody)
            np.savetxt('np_A.txt', np_A)
            # np.savetxt("Gpart.txt",np_Gpart)
            # np.savetxt("Gbody.txt",np_Gbody)



        x_joint = None
        for i in range(self.num_subset):
            x_temp = x.view(N, C * T, V)                        #将每个通道的时间序列拼接在一起得到更长的时间序列  x.view(N,T*C,V)将每个时间步的所有通道拼接在一起，得到更宽的特征图 只是拼接没有相乘 结果一样
            x_temp = self.conv_joint[i](torch.matmul(x_temp, self.norm(A[i])).view(N, C, T, V))
            #x_temp = self.conv_joint[i](torch.matmul(x_temp, A[i]).view(N, C, T, V))               #不使用norm试一下 效果一般且轻微震荡
            x_joint = x_temp + x_joint if x_joint is not None else x_temp

        # for i in range(self.num_subset):
        #     x_temp = x.view(N, C, T, V)
        #     x_temp = self.conv_joint[i](x_temp)
        #     x_temp = x_temp.permute(0, 2, 3, 1)  # N,T,V,C
        #     coefs = self.softmax(self.leaky_relu(x_temp) + self.bias_mat[i])
        #     h_prim = torch.matmul(coefs, x_temp)
        #     x_joint = h_prim + x_joint if x_joint is not None else h_prim
        # x_joint = x_joint.permute(0, 3, 1, 2)
        # x_joint = self.convmatch(x_joint)


        # PositionalEncoding 神经网络常用技巧
        x_withPE = self.PE(x)
        theta = self.conv_theta(x_withPE).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
        phi = self.conv_phi(x_withPE).view(N, self.inter_c * T, V)
        att_map = self.soft(torch.matmul(theta, phi) / theta.size(-1))  # N V V

        #非局部块
        x_nonlocal = x.view(N, C * T, V)
        x_nonlocal = torch.matmul(x_nonlocal, att_map).view(N, C, T, V)
        x_nonlocal = self.conv_nonlocal(x_nonlocal)

        #论文中的Xpart和Xbody  式(3)
        x_part = x.view(N, C * T, V)
        x_part = torch.matmul(x_part, self.norm(G_part)).view(N, C, T, V)
        x_part = self.conv_part(x_part)
        x_body = x.view(N, C * T, V)
        x_body = torch.matmul(x_body, self.norm(G_body)).view(N, C, T, V)
        x_body = self.conv_body(x_body)
        """
        x_avg = x.view(N, C * T, V)
        x_avg = torch.matmul(x_avg, Avg).view(N, C, T, V)
        x_avg = self.conv_avg(x_avg)
        y = x_joint + x_part + x_body + x_nonlocal + x_avg
        """

        # selective-scale.     [batch_size,64,100,25]
        x_joint = x_joint.unsqueeze_(dim=1)
        x_part = x_part.unsqueeze_(dim=1)
        x_body = x_body.unsqueeze_(dim=1)




        #x_joint x_part x_body = [22,1,64,100,25]
        x_local = torch.cat([x_joint, x_part, x_body], dim=1)      #论文中的U 式(4)
        x_sum = torch.sum(x_local, dim=1)
        glo_avg = x_sum.mean(-1).mean(-1)
        feature_z = self.fc(glo_avg)  # 式(5)
        #feature_z = glo_avg

        attention_vectors = None
        for i, fc in enumerate(self.fc_branch):
            vector = fc(feature_z).unsqueeze_(dim=1)
            #vector = feature_z.unsqueeze_(dim=1)                               #不加全连接层试一下 效果不好
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)  # 式6
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)  # unsqueeze 在指定dim扩展维度，并且扩展的维度为1
        x_local_selected = (x_local * attention_vectors).sum(dim=1)  # 式7

        y = x_local_selected + x_nonlocal  # x_nonlocal 论文中提到的非局部块
        y = self.bn(y)                                                    #不要bn 效果好
        y += self.down(x)
        #print("daozhelilemei")

        return self.relu(y)