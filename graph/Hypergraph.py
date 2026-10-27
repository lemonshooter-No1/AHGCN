import numpy as np
import torch
from scipy.spatial import cKDTree


# Spatial and temporal hypergraph construction.
class Hypergraph:
    def __init__(self, num_center=5, sparse_lambda=0.01, sparse_threshold=0.001,
                 sparse_iter=1000, sparse_tol=1e-6, temporal_k=4):
        if num_center < 1 or sparse_lambda <= 0 or sparse_threshold < 0:
            raise ValueError('Invalid spatial hypergraph parameters.')
        if sparse_iter < 1 or sparse_tol <= 0 or temporal_k < 1:
            raise ValueError('Invalid hypergraph solver parameters.')
        self.num_center = num_center
        self.sparse_lambda = sparse_lambda
        self.sparse_threshold = sparse_threshold
        self.sparse_iter = sparse_iter
        self.sparse_tol = sparse_tol
        self.temporal_k = temporal_k

    @torch.no_grad()
    def spatial(self, x):
        N, T, V, C = x.shape
        if self.num_center > V or T < 2:
            raise ValueError('Spatial hypergraphs require at least two frames and num_center <= joints.')
        motion = x[:, 1:] - x[:, :-1]
        distance = motion.norm(dim=-1).sum(dim=1)
        centers = distance.topk(self.num_center, dim=1).indices
        z = motion.permute(0, 2, 1, 3).reshape(N, V, -1)
        z = torch.nn.functional.normalize(z, dim=-1)
        gram = torch.bmm(z, z.transpose(1, 2))
        target = gram.gather(1, centers.unsqueeze(-1).expand(-1, -1, V))
        mask = torch.ones_like(target).scatter_(2, centers.unsqueeze(-1), 0)
        step = gram.abs().sum(dim=-1).amax(dim=-1).clamp_min(1).reciprocal().view(N, 1, 1)
        coef = torch.zeros_like(target)
        search = coef.clone()
        momentum = 1.0
        active = torch.ones(N, 1, 1, dtype=torch.bool, device=x.device)
        for _ in range(self.sparse_iter):
            value = search - step * (torch.bmm(search, gram) - target)
            updated = value.sign() * (value.abs() - step * self.sparse_lambda).clamp_min(0) * mask
            updated = torch.where(active, updated, coef)
            converged = (updated - coef).abs().flatten(1).amax(dim=1).view(N, 1, 1) < self.sparse_tol
            active = active & ~converged
            if not active.any().item():
                coef = updated
                break
            next_momentum = (1 + (1 + 4 * momentum * momentum) ** 0.5) / 2
            search = updated + (momentum - 1) / next_momentum * (updated - coef)
            coef = updated
            momentum = next_momentum
        H = (coef.abs() > self.sparse_threshold).to(x.dtype)
        H.scatter_(2, centers.unsqueeze(-1), 1)
        H = H.transpose(1, 2)
        degree_node = H.sum(dim=2)
        degree_edge = H.sum(dim=1)
        scale = degree_node.clamp_min(1).rsqrt()
        kernel = torch.bmm(H / degree_edge.unsqueeze(1), H.transpose(1, 2))
        return kernel * scale.unsqueeze(2) * scale.unsqueeze(1)

    @torch.no_grad()
    def temporal(self, x):
        N, T, V, C = x.shape
        if self.temporal_k > (T - 1) * V:
            raise ValueError('temporal_k exceeds the number of nodes in other frames.')
        count = T * V
        result = []
        query_count = min(count, self.temporal_k + V)
        for sample in x.detach().cpu().numpy():
            points = sample.reshape(count, C)
            tree = cKDTree(points)
            _, neighbors = tree.query(points, k=query_count)
            valid = neighbors // V != np.arange(count)[:, None] // V
            order = np.argsort(~valid, axis=1, kind='stable')[:, :self.temporal_k]
            neighbors = np.take_along_axis(neighbors, order, axis=1)
            members = np.concatenate((np.arange(count)[:, None], neighbors), axis=1)
            rows = torch.as_tensor(members.reshape(-1), dtype=torch.long, device=x.device)
            columns = torch.arange(count, device=x.device).repeat_interleave(self.temporal_k + 1)
            indices = torch.stack((rows, columns))
            values = torch.ones(rows.numel(), dtype=x.dtype, device=x.device)
            result.append(torch.sparse_coo_tensor(indices, values, (count, count)).coalesce())
        return result

    @staticmethod
    def restrict(graphs, frames, num_point):
        result = []
        for H, selected in zip(graphs, frames):
            nodes = (selected.unsqueeze(1) * num_point + torch.arange(num_point, device=selected.device)).reshape(-1)
            mapping = torch.full((H.size(0),), -1, dtype=torch.long, device=selected.device)
            mapping[nodes] = torch.arange(nodes.numel(), device=selected.device)
            rows, columns = H.coalesce().indices()
            keep = (mapping[rows] >= 0) & (mapping[columns] >= 0)
            indices = torch.stack((mapping[rows[keep]], mapping[columns[keep]]))
            values = H.values()[keep]
            result.append(torch.sparse_coo_tensor(indices, values, (nodes.numel(), nodes.numel())).coalesce())
        return result


def prepare_temporal(graphs):
    count = graphs[0].size(0)
    indices = torch.cat([H.coalesce().indices() + i * count for i, H in enumerate(graphs)], dim=1)
    values = torch.cat([H.coalesce().values() for H in graphs])
    size = len(graphs) * count
    H = torch.sparse_coo_tensor(indices, values, (size, size)).coalesce()
    rows, columns = H.indices()
    degree_node = torch.bincount(rows, minlength=size).to(values.dtype)
    degree_edge = torch.bincount(columns, minlength=size).to(values.dtype)
    scale = degree_node.clamp_min(1).rsqrt().unsqueeze(1)
    return H, H.transpose(0, 1).coalesce(), scale, degree_edge.clamp_min(1).unsqueeze(1)


def temporal_aggregate(x, graphs, prepared=None):
    N, C, T, V = x.shape
    H, transpose, scale, degree_edge = prepare_temporal(graphs) if prepared is None else prepared
    feature = x.permute(0, 2, 3, 1).reshape(N * T * V, C)
    edges = torch.sparse.mm(transpose, feature * scale) / degree_edge
    output = torch.sparse.mm(H, edges) * scale
    return output.reshape(N, T, V, C).permute(0, 3, 1, 2).contiguous()


# Cache sequence hypergraphs for indexed minibatch access.
class HypergraphCache:
    def __init__(self, structure, data, device='cpu', batch_size=20, build_device='cpu'):
        if batch_size < 1:
            raise ValueError('Cache batch size must be positive.')
        self.device = torch.device(device)
        self.spatial = []
        self.temporal = []
        with torch.no_grad():
            for start in range(0, len(data), batch_size):
                x = torch.as_tensor(data[start:start + batch_size], dtype=torch.float32, device=build_device)
                self.spatial.append(structure.spatial(x).to(self.device))
                self.temporal.extend(H.to(self.device) for H in structure.temporal(x))
        self.spatial = torch.cat(self.spatial, dim=0)

    def batch(self, indices, device):
        indices = torch.as_tensor(indices, dtype=torch.long).tolist()
        G = self.spatial[torch.as_tensor(indices, dtype=torch.long, device=self.device)].to(device)
        H = [self.temporal[i].to(device) for i in indices]
        return G, H

    @staticmethod
    def select_device(mode, device, data, temporal_k):
        device = torch.device(device)
        if mode == 'none':
            return None
        if mode == 'cpu' or device.type == 'cpu':
            if mode == 'cuda':
                raise ValueError('CUDA caching requires CUDA training or testing.')
            return torch.device('cpu')
        if mode == 'cuda':
            return device
        size = 0
        for x in data:
            N, T, V, C = x.shape
            size += N * (4 * V * V + 20 * T * V * (temporal_k + 1))
        free, _ = torch.cuda.mem_get_info(device)
        return device if size < free // 10 else torch.device('cpu')
