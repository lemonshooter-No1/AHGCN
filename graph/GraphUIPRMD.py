import sys
import numpy as np

from graph import tools

sys.path.extend(['../'])

# Joint index:
# {0,  "Left head front"}
# {1,  "Right head front"},
# {2,  "Left back head"},
# {3,  "Right back head"},
# {4,  "cervical vertebra"},
# {5,  "thoracic vertabra"},
# {6,  "Clavicle"},
# {7,  "Sternum"},
# {8,  "Right back"},
# {9,  "Left shoulder"},
# {10, "Left upper arm"},
# {11, "Left elbow"},
# {12, "Left forearm"},
# {13, "Left wrist A"},
# {14, "Left wrist B"},
# {15, "Left finger"},
# {16, "Right shoulder"},
# {17, "Right upper arm"}
# {18, "Right elbow"}
# {19, "Right forearm"}
# {20, "Right wrist A"}
# {21, "Right wrist B"}
# {22, "Right finger"}
# {23, "Left ASIS"}
# {24, "Right ASIS"}
# {25, "Left PSIS"}
# {26, "Right PSIS"}
# {27, "Left thigh"}
# {28, "Left knee"}
# {29, "Left tibia"}
# {30, "Left ankle"}
# {31, "Left heel"}
# {32, "Left toe"}
# {33, "Right thigh"}
# {34, "Right knee"}
# {35, "Right tibia"}
# {36, "Right ankle"}
# {37, "Right heel"}
# {38, "Right toe"}






num_node = 39
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (1, 3), (1, 4), (1, 5),(4,5),(6,7),(4,6),(5,7),(6,8),(7,8),(8,9),(8,10),(6,9),(7,10),(11,12),(11,13),
                    (12,14),(13,15),(11,14),(11,15),(11,16),(11,17),(16,18),(17,19),(18,20),(19,21),(20,22),(21,23),(22,24),(23,25),
                    (24,26),(25,27),(6,28),(7,29),(28,30),(29,31),(16,32),(20,32),(17,33),(21,33),(30,34),(31,35),(34,26),(35,37),(34,38),(35,39),(36,38),(37,39)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

inward_ori_index_B = [(1, 2), (2, 21), (3, 21), (4, 3), (6, 5), (7, 6),
                     (8, 7), (10, 9), (11, 10), (12, 11),
                     (14, 13), (15, 14), (16, 15), (18, 17), (19, 18),
                     (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward_B = [(i - 1, j - 1) for (i, j) in inward_ori_index_B]
outward_B = [(j, i) for (i, j) in inward_B]
neighbor_B = inward_B + outward_B

class GraphUIPRMD:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.B = self.get_adjacency_matrix_B(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.neighbor_B = neighbor_B

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A

    def get_adjacency_matrix_B(self, labeling_mode=None):
        if labeling_mode is None:
            return self.B
        if labeling_mode == 'spatial':
            B = tools.get_spatial_graph(num_node, self_link, inward_B, outward_B)
        else:
            raise ValueError()
        return B


class Hypergraph:
    def __init__(self):
        self.G_part = self.generate_G_part()
        self.G_body = self.generate_G_body()

    def generate_G_part(self, variable_weight=False):
        H = np.zeros((39, 15))
        H[0][0], H[1][0], H[2][0] = 1, 1, 1
        H[1][1], H[3][1], H[5][1] = 1, 1, 1
        H[2][2], H[4][2], H[6][2] = 1, 1, 1
        H[7][3], H[8][3], H[9][3] = 1, 1, 1
        H[10][4], H[11][4], H[12][4] = 1, 1, 1
        H[10][5], H[13][5], H[15][5] = 1, 1, 1
        H[11][6], H[14][6], H[16][6] = 1, 1, 1
        H[15][7], H[17][7], H[31][7], H[19][7] = 1, 1, 1, 1
        H[16][8], H[18][8], H[32][8], H[20][8] = 1, 1, 1, 1
        H[21][9], H[23][9], H[25][9] = 1, 1, 1
        H[22][10], H[24][10], H[26][10] = 1, 1, 1
        H[5][11], H[27][11], H[29][11] = 1, 1, 1
        H[6][12], H[28][12], H[30][12] = 1, 1, 1
        H[33][13],H[35][13],H[37][13] = 1, 1, 1
        H[34][14],H[36][14],H[38][14] = 1, 1, 1


        H = np.array(H)
        n_edge = H.shape[1]
        # the weight of the hyperedge
        W = np.ones(n_edge)
        # the degree of the node
        DV = np.sum(H * W, axis=1)
        # the degree of the hyperedge
        DE = np.sum(H, axis=0)
        #print(DE)

        invDE = np.mat(np.diag(np.power(DE, -1)))
        DV2 = np.mat(np.diag(np.power(DV, -0.5)))
        W = np.mat(np.diag(W))
        H = np.mat(H)
        HT = H.T

        if variable_weight:
            DV2_H = DV2 * H
            invDE_HT_DV2 = invDE * HT * DV2
            return DV2_H, W, invDE_HT_DV2
        else:
            G = DV2 * H * W * invDE * HT * DV2
            return G

    def generate_G_body(self, variable_weight=False):
        H = np.zeros((39, 5))
        H[0][0], H[3][0], H[4][0], H[5][0], H[6][0], H[7][0], H[10][0], H[13][0], H[14][0], H[15][0], H[16][0] = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1   #头 脖子 锁骨 胸 骨盆 臀 股骨
        H[11][1], H[13][1], H[15][1], H[17][1], H[19][1], H[31][1],H[21][1],H[23][1],H[25][1] = 1, 1, 1, 1, 1, 1 ,1, 1, 1                          #左侧 骨盆 臀 股骨 膝 胫骨 半月板 脚踝 足 脚趾头
        H[12][2], H[14][2], H[16][2], H[18][2], H[20][2], H[32][2],H[22][2],H[24][2],H[26][2] = 1, 1, 1, 1, 1, 1 ,1, 1, 1                          #右侧 骨盆 臀 股骨 膝 胫骨 半月板 脚踝 足 脚趾头
        H[1][3], H[3][3], H[5][3], H[27][3], H[8][3],H[29][3],H[33][3],H[35][3],H[37][3] = 1, 1, 1, 1, 1, 1, 1, 1 ,1                               #左侧 头 脖子 锁骨 肩 胸 肘 手腕 上手 手
        H[2][4], H[4][4], H[6][4], H[28][4], H[9][4],H[30][4],H[34][4],H[36][4],H[38][4] = 1, 1, 1, 1, 1, 1, 1, 1 ,1                               #右侧 头 脖子 锁骨 肩 胸 肘 手腕 上手 手


        H = np.array(H)
        n_edge = H.shape[1]
        W = np.ones(n_edge)
        DV = np.sum(H * W, axis=1)
        DE = np.sum(H, axis=0)

        invDE = np.mat(np.diag(np.power(DE, -1)))
        DV2 = np.mat(np.diag(np.power(DV, -0.5)))
        W = np.mat(np.diag(W))
        H = np.mat(H)
        HT = H.T

        if variable_weight:
            DV2_H = DV2 * H
            invDE_HT_DV2 = invDE * HT * DV2
            return DV2_H, W, invDE_HT_DV2
        else:
            G = DV2 * H * W * invDE * HT * DV2
            return G

if __name__ == '__main__':
    HP = Hypergraph()