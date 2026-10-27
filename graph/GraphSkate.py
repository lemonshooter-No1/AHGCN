import sys
import numpy as np
import torch

from graph import tools

sys.path.extend(['../'])





num_node = 18
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8),
                    (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
                    (0, 14), (14, 16), (0, 15), (15, 17)]
neighbor_link = list(inward_ori_index)
self_edge = self_link + neighbor_link


inward = list(inward_ori_index)
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

inward_ori_index_B = [(1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8),
                    (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
                    (0, 14), (14, 16), (0, 15), (15, 17)]
inward_B = list(inward_ori_index_B)
outward_B = [(j, i) for (i, j) in inward_B]
neighbor_B = inward_B + outward_B


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1


    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

self_hop_dis = get_hop_distance(num_node, self_edge, max_hop=1)




self_edge = self_link + neighbor_link
A1 = np.zeros((num_node, num_node))
for i, j in self_edge:
    A1[j, i] = 1
    A1[i, j] = 1

A2 = np.zeros((num_node, num_node))
for root in range(A1.shape[1]):
    for neighbour in range(A1.shape[0]):
        if A1[root, neighbour] == 1:
            for neighbour_of_neigbour in range(A1.shape[0]):
                if A1[neighbour, neighbour_of_neigbour] == 1:
                    A2[root, neighbour_of_neigbour] = 1





class GraphMITSkate:
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

        bias_mat = []
        bias_mat_1 = np.where(A1[0] != 0, A1[0], -1e9)
        bias_mat_2 = np.where(A2[0] != 0, A2[0], -1e9)
        bias_mat_1 = torch.as_tensor(bias_mat_1)
        bias_mat_2 = torch.as_tensor(bias_mat_2)
        bias_mat.append(bias_mat_1)
        bias_mat.append(bias_mat_2)
        bias_mat = np.stack(bias_mat)

        return A

    def get_adjacency_matrix_B(self, labeling_mode=None):
        if labeling_mode is None:
            return self.B
        if labeling_mode == 'spatial':
            B = tools.get_spatial_graph(num_node, self_link, inward_B, outward_B)
        else:
            raise ValueError()
        return B



# Spatial and temporal hypergraph construction.
class Hypergraph:
    def __init__(self):
        self.G_part = self.generate_G_part()
        self.G_body = self.generate_G_body()

    def generate_G_part(self, variable_weight=False):
        H = np.zeros((18, 8))
        H[0][0], H[14][0], H[16][0] = 1, 1, 1
        H[0][1], H[15][1], H[17][1] = 1, 1, 1
        H[2][2], H[1][2], H[5][2]  = 1, 1, 1
        H[2][3], H[3][3], H[4][3] = 1, 1, 1
        H[5][4], H[6][4], H[7][4] = 1, 1, 1
        H[0][5], H[1][5], H[8][5] = 1, 1, 1
        H[0][5], H[1][5], H[11][5] = 1, 1, 1
        H[8][6], H[9][6], H[10][6] = 1, 1, 1
        H[11][7], H[12][7], H[13][7] = 1, 1, 1



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



        H = np.zeros((18, 5))
        H[16][0], H[14][0], H[0][0], H[15][0], H[17][0] = 1, 1, 1, 1, 1
        H[1][1], H[2][1], H[3][1], H[4][1] = 1, 1, 1, 1
        H[1][2], H[5][2], H[6][2], H[7][2] = 1, 1, 1, 1
        H[1][3], H[8][3], H[9][3], H[10][3] = 1, 1, 1, 1
        H[1][4], H[11][4], H[12][4], H[13][4] = 1, 1, 1, 1





        H = np.array(H)
        n_edge = H.shape[1]
        # the weight of the hyperedge
        W = np.ones(n_edge)
        # the degree of the node
        DV = np.sum(H * W, axis=1)
        # the degree of the hyperedge
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

GraphOlySkate = GraphMITSkate


if __name__ == '__main__':
    HP = Hypergraph()
