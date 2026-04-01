import numpy as np


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD



def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


def get_spatial_graph_with_no_init(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = edge2mat(inward, num_node)
    Out = edge2mat(outward, num_node)
    A = np.stack((I, In, Out))
    return A

def get_hop_distance(num_node,edge,max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def get_spatial_graphA(num_node, edge, center, hop_dis, max_hop):
    valid_hop = range(0, 2, 1)
    adjacency = np.zeros((num_node, num_node))
    for hop in valid_hop:
        adjacency[hop_dis == hop] = 1
    normalize_adjacency = normalize_digraph(adjacency)
    hop_dis = get_hop_distance(num_node, edge, max_hop=max_hop)
    A = []
    for hop in valid_hop:
        a_root = np.zeros((num_node, num_node))
        a_close = np.zeros((num_node, num_node))
        a_further = np.zeros((num_node, num_node))
        for i in range(num_node):
            for j in range(num_node):
                if hop_dis[j, i] == hop:
                    if hop_dis[j, center] == hop_dis[i, center]:
                        a_root[j, i] = normalize_adjacency[j, i]
                    elif hop_dis[j, center] > hop_dis[i, center]:
                        a_close[j, i] = normalize_adjacency[j, i]
                    else:
                        a_further[j, i] = normalize_adjacency[j, i]
        if hop == 0:
            A.append(a_root)
        else:
            A.append(a_root + a_close)
            A.append(a_further)
    A = np.stack(A)
    return A