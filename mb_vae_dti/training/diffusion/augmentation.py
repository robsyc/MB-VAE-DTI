"""
This file contains the augmentation utilities for the discrete diffusion decoder of DTITree.

Extra augmentation features are computed using the discretized noisy graph G^t.
    - Cycles & spectral features (graph descriptors)
    - Molecular features (charge, valency, weight)
"""

import torch

from .utils import PlaceHolder

class Augmentation:
    def __init__(
            self,
            valencies,
            max_weight,
            atom_weights,
            max_n_nodes = 64
    ):
        self.extra_graph_features = ExtraFeatures(max_n_nodes=max_n_nodes)
        self.extra_molecular_features = ExtraMolecularFeatures(
            valencies=valencies,
            max_weight=max_weight,
            atom_weights=atom_weights
        )
    
    def __call__(self, G_t, node_mask):
        """
        Compute both graph-structural features and molecular features,
        then combine them into a single PlaceHolder.
        """
        graph_features = self.extra_graph_features(G_t.E, node_mask)
        molecular_features = self.extra_molecular_features(G_t)
        
        # Combine the features
        combined_X = torch.cat((graph_features.X, molecular_features.X), dim=-1)
        combined_E = torch.cat((graph_features.E, molecular_features.E), dim=-1)
        combined_y = torch.cat((graph_features.y, molecular_features.y), dim=-1)
        
        return PlaceHolder(X=combined_X, E=combined_E, y=combined_y)

##################################################################################
# Descriptive graph features (cycles, spectral features)

class ExtraFeatures:
    def __init__(self, max_n_nodes = 64):
        self.max_n_nodes = max_n_nodes
        self.ncycles = NodeCycleFeatures()
        self.eigenfeatures = EigenFeatures()

    def __call__(self, E_t, node_mask):
        n = node_mask.sum(dim=1).unsqueeze(1) / self.max_n_nodes
        x_cycles, y_cycles = self.ncycles(E_t, node_mask)       # (bs, n_cycles)

        eigenfeatures = self.eigenfeatures(E_t, node_mask)
        extra_edge_attr = torch.zeros((*E_t.shape[:-1], 0)).type_as(E_t)
        n_components, batched_eigenvalues, nonlcc_indicator, k_lowest_eigvec = eigenfeatures
        # (bs, 1), (bs, 10), (bs, n, 1), (bs, n, 2)

        return PlaceHolder(
            X=torch.cat((x_cycles, nonlcc_indicator, k_lowest_eigvec), dim=-1),
            E=extra_edge_attr,
            y=torch.hstack((n, y_cycles, n_components, batched_eigenvalues)))


class NodeCycleFeatures:
    def __init__(self):
        self.kcycles = KNodeCycles()

    def __call__(self, E_t, node_mask):
        adj_matrix = E_t[..., 1:].sum(dim=-1).float()
        x_cycles, y_cycles = self.kcycles.k_cycles(adj_matrix=adj_matrix)   # (bs, n_cycles)
        x_cycles = x_cycles.type_as(adj_matrix) * node_mask.unsqueeze(-1)
        # Avoid large values when the graph is dense
        x_cycles = x_cycles / 10
        y_cycles = y_cycles / 10
        x_cycles[x_cycles > 1] = 1
        y_cycles[y_cycles > 1] = 1
        return x_cycles, y_cycles


class EigenFeatures:
    """
    Code taken from : https://github.com/Saro00/DGN/blob/master/models/pytorch/eigen_agg.py
    """
    def __init__(self):
        pass

    def __call__(self, E_t, node_mask):
        A = E_t[..., 1:].sum(dim=-1).float() * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        L = compute_laplacian(A, normalize=False)
        mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1]).type_as(L).unsqueeze(0)
        mask_diag = mask_diag * (~node_mask.unsqueeze(1)) * (~node_mask.unsqueeze(2))
        L = L * node_mask.unsqueeze(1) * node_mask.unsqueeze(2) + mask_diag

        eigvals, eigvectors = torch.linalg.eigh(L)
        eigvals = eigvals.type_as(A) / torch.sum(node_mask, dim=1, keepdim=True)
        eigvectors = eigvectors * node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
        # Retrieve eigenvalues features
        n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)

        # Retrieve eigenvectors features
        nonlcc_indicator, k_lowest_eigenvector = get_eigenvectors_features(
            vectors=eigvectors,
            node_mask=node_mask,
            n_connected=n_connected_comp
        )
        return n_connected_comp, batch_eigenvalues, nonlcc_indicator, k_lowest_eigenvector


def compute_laplacian(adjacency, normalize: bool):
    """
    adjacency : batched adjacency matrix (bs, n, n)
    normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
    Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    diag = torch.sum(adjacency, dim=-1)     # (bs, n)
    n = diag.shape[-1]
    D = torch.diag_embed(diag)      # Degree matrix      # (bs, n, n)
    combinatorial = D - adjacency                        # (bs, n, n)

    if not normalize:
        return (combinatorial + combinatorial.transpose(1, 2)) / 2

    diag0 = diag.clone()
    diag[diag == 0] = 1e-12

    diag_norm = 1 / torch.sqrt(diag)            # (bs, n)
    D_norm = torch.diag_embed(diag_norm)        # (bs, n, n)
    L = torch.eye(n).unsqueeze(0) - D_norm @ adjacency @ D_norm
    L[diag0 == 0] = 0
    return (L + L.transpose(1, 2)) / 2


def get_eigenvalues_features(eigenvalues, k=5):
    """
    values : eigenvalues -- (bs, n)
    node_mask: (bs, n)
    k: num of non zero eigenvalues to keep
    """
    ev = eigenvalues
    bs, n = ev.shape
    n_connected_components = (ev < 1e-5).sum(dim=-1)
    assert (n_connected_components > 0).all(), (n_connected_components, ev)

    to_extend = max(n_connected_components) + k - n
    if to_extend > 0:
        eigenvalues = torch.hstack((eigenvalues, 2 * torch.ones(bs, to_extend).type_as(eigenvalues)))
    indices = torch.arange(k).type_as(eigenvalues).long().unsqueeze(0) + n_connected_components.unsqueeze(1)
    first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
    return n_connected_components.unsqueeze(-1), first_k_ev


def get_eigenvectors_features(vectors, node_mask, n_connected, k=2):
    """
    vectors (bs, n, n) : eigenvectors of Laplacian IN COLUMNS
    returns:
        not_lcc_indicator : indicator vectors of largest connected component (lcc) for each graph  -- (bs, n, 1)
        k_lowest_eigvec : k first eigenvectors for the largest connected component   -- (bs, n, k)
    """
    bs, n = vectors.size(0), vectors.size(1)

    # Create an indicator for the nodes outside the largest connected components
    first_ev = torch.round(vectors[:, :, 0], decimals=3) * node_mask                        # bs, n
    # Add random value to the mask to prevent 0 from becoming the mode
    random = torch.randn(bs, n, device=node_mask.device) * (~node_mask)                                   # bs, n
    first_ev = first_ev + random
    most_common = torch.mode(first_ev, dim=1).values                                    # values: bs -- indices: bs
    mask = ~ (first_ev == most_common.unsqueeze(1))
    not_lcc_indicator = (mask * node_mask).unsqueeze(-1).float()

    # Get the eigenvectors corresponding to the first nonzero eigenvalues
    to_extend = max(n_connected) + k - n
    if to_extend > 0:
        vectors = torch.cat((vectors, torch.zeros(bs, n, to_extend).type_as(vectors)), dim=2)   # bs, n , n + to_extend
    indices = torch.arange(k).type_as(vectors).long().unsqueeze(0).unsqueeze(0) + n_connected.unsqueeze(2)    # bs, 1, k
    indices = indices.expand(-1, n, -1)                                               # bs, n, k
    first_k_ev = torch.gather(vectors, dim=2, index=indices)       # bs, n, k
    first_k_ev = first_k_ev * node_mask.unsqueeze(2)

    return not_lcc_indicator, first_k_ev


def batch_trace(X):
    """
    Expect a matrix of shape B N N, returns the trace in shape B
    :param X:
    :return:
    """
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    return trace


def batch_diagonal(X):
    """
    Extracts the diagonal from the last two dims of a tensor
    :param X:
    :return:
    """
    return torch.diagonal(X, dim1=-2, dim2=-1)


class KNodeCycles:
    """ Builds cycle counts for each node in a graph.
    """

    def __init__(self):
        super().__init__()

    def calculate_kpowers(self):
        self.k1_matrix = self.adj_matrix.float()
        self.d = self.adj_matrix.sum(dim=-1)
        self.k2_matrix = self.k1_matrix @ self.adj_matrix.float()
        self.k3_matrix = self.k2_matrix @ self.adj_matrix.float()
        self.k4_matrix = self.k3_matrix @ self.adj_matrix.float()
        self.k5_matrix = self.k4_matrix @ self.adj_matrix.float()
        self.k6_matrix = self.k5_matrix @ self.adj_matrix.float()

    def k3_cycle(self):
        """ tr(A ** 3). """
        c3 = batch_diagonal(self.k3_matrix)
        return (c3 / 2).unsqueeze(-1).float(), (torch.sum(c3, dim=-1) / 6).unsqueeze(-1).float()

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4_matrix)
        c4 = diag_a4 - self.d * (self.d - 1) - (self.adj_matrix @ self.d.unsqueeze(-1)).sum(dim=-1)
        return (c4 / 2).unsqueeze(-1).float(), (torch.sum(c4, dim=-1) / 8).unsqueeze(-1).float()

    def k5_cycle(self):
        diag_a5 = batch_diagonal(self.k5_matrix)
        triangles = batch_diagonal(self.k3_matrix) / 2

        # Triangle count matrix (indicates for each node i how many triangles it shares with node j)
        joint_cycles = self.k2_matrix * self.adj_matrix
        # c5 = diag_a5 - 2 * triangles * self.d - (self.adj_matrix @ triangles.unsqueeze(-1)).sum(dim=-1) + triangles
        prod = 2 * (joint_cycles @ self.d.unsqueeze(-1)).squeeze(-1)
        prod2 = 2 * (self.adj_matrix @ triangles.unsqueeze(-1)).squeeze(-1)
        c5 = diag_a5 - prod - 4 * self.d * triangles - prod2 + 10 * triangles
        return (c5 / 2).unsqueeze(-1).float(), (c5.sum(dim=-1) / 10).unsqueeze(-1).float()

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6_matrix)
        term_2_t = batch_trace(self.k3_matrix ** 2)
        term3_t = torch.sum(self.adj_matrix * self.k2_matrix.pow(2), dim=[-2, -1])
        d_t4 = batch_diagonal(self.k2_matrix)
        a_4_t = batch_diagonal(self.k4_matrix)
        term_4_t = (d_t4 * a_4_t).sum(dim=-1)
        term_5_t = batch_trace(self.k4_matrix)
        term_6_t = batch_trace(self.k3_matrix)
        term_7_t = batch_diagonal(self.k2_matrix).pow(3).sum(-1)
        term8_t = torch.sum(self.k3_matrix, dim=[-2, -1])
        term9_t = batch_diagonal(self.k2_matrix).pow(2).sum(-1)
        term10_t = batch_trace(self.k2_matrix)

        c6_t = (term_1_t - 3 * term_2_t + 9 * term3_t - 6 * term_4_t + 6 * term_5_t - 4 * term_6_t + 4 * term_7_t +
                3 * term8_t - 12 * term9_t + 4 * term10_t)
        return None, (c6_t / 12).unsqueeze(-1).float()

    def k_cycles(self, adj_matrix, verbose=False):
        self.adj_matrix = adj_matrix
        self.calculate_kpowers()

        k3x, k3y = self.k3_cycle()
        assert (k3x >= -0.1).all()

        k4x, k4y = self.k4_cycle()
        assert (k4x >= -0.1).all()

        k5x, k5y = self.k5_cycle()
        assert (k5x >= -0.1).all(), k5x

        _, k6y = self.k6_cycle()
        assert (k6y >= -0.1).all()

        kcyclesx = torch.cat([k3x, k4x, k5x], dim=-1)
        kcyclesy = torch.cat([k3y, k4y, k5y, k6y], dim=-1)
        return kcyclesx, kcyclesy

################################################################################## 
# Molecular features (charge, valency, weight)

class ExtraMolecularFeatures:
    def __init__(
            self, 
            valencies, 
            max_weight, 
            atom_weights):
        self.charge = ChargeFeature(valencies=valencies)
        self.valency = ValencyFeature()
        self.weight = WeightFeature(max_weight=max_weight, atom_weights=atom_weights)

    def __call__(self, G_t):
        charge = self.charge(G_t).unsqueeze(-1)      # (bs, n, 1)
        valency = self.valency(G_t).unsqueeze(-1)    # (bs, n, 1)
        weight = self.weight(G_t)                    # (bs, 1)

        extra_edge_attr = torch.zeros((*G_t.E.shape[:-1], 0)).type_as(G_t.E)

        return PlaceHolder(X=torch.cat((charge, valency), dim=-1), E=extra_edge_attr, y=weight)


class ChargeFeature:
    def __init__(self, valencies):
        self.valencies = valencies

    def __call__(self, G_t):
        bond_orders = torch.tensor([0, 1, 2, 3, 1.5], device=G_t.E.device).reshape(1, 1, 1, -1)
        weighted_E = G_t.E * bond_orders      # (bs, n, n, de)
        current_valencies = weighted_E.argmax(dim=-1).sum(dim=-1)   # (bs, n)

        valencies = torch.tensor(self.valencies, device=G_t.X.device).reshape(1, 1, -1)
        X = G_t.X * valencies  # (bs, n, dx)
        normal_valencies = torch.argmax(X, dim=-1)               # (bs, n)

        return (normal_valencies - current_valencies).type_as(G_t.X)


class ValencyFeature:
    def __init__(self):
        pass

    def __call__(self, G_t):
        orders = torch.tensor([0, 1, 2, 3, 1.5], device=G_t.E.device).reshape(1, 1, 1, -1)
        E = G_t.E * orders      # (bs, n, n, de)
        valencies = E.argmax(dim=-1).sum(dim=-1)    # (bs, n)
        return valencies.type_as(G_t.X)


class WeightFeature:
    def __init__(self, max_weight, atom_weights):
        self.max_weight = max_weight
        self.atom_weight_list = torch.tensor(atom_weights)

    def __call__(self, G_t):
        X = torch.argmax(G_t.X, dim=-1)         # (bs, n)
        X_weights = self.atom_weight_list.to(X.device)[X]   # (bs, n)
        return X_weights.sum(dim=-1).unsqueeze(-1).type_as(G_t.X) / self.max_weight     # (bs, 1)
