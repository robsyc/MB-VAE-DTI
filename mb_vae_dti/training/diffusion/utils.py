"""
Utility functions/classes used in the discrete diffusion decoder of the DTITree model.
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import BondType as BT
import wandb
import os
import rdkit
from typing import Dict, List

from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, to_dense_batch, remove_self_loops

class PlaceHolder:
    """
    This is basically a wrapper around objects X (nodes), E (edges), and y (global features).
    Used to
    - Store limit distribution graph (G^T) with marginal distributions of nodes & edges
    - 
    """
    def __init__(self, X, E, y = None):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        try:
            self.y = self.y.type_as(x)
        except:
            pass
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self

##################################################################################

def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    """
    Used in
    - GraphTransformer
    """
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ.
    Used in
    - PredefinedNoiseScheduleDiscrete
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    return betas.squeeze()

##################################################################################

def to_dense(x, edge_index, edge_attr, batch):
    """
    Used to produce dense_data & node_mask from batch data (x, edge_index, edge_attr, batch)
    - in validation_step()
    """
    X, node_mask = to_dense_batch(x=x, batch=batch)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)
    return PlaceHolder(X=X, E=E, y=None), node_mask

def encode_no_edge(E):
    # Create a new tensor with an additional dimension for no-edge type as the FIRST feature
    bs, n, n, num_edge_types = E.shape
    E_new = torch.zeros(bs, n, n, num_edge_types + 1, device=E.device, dtype=E.dtype)
    
    # Identify positions with no edges (sum of all edge types is 0)
    no_edge = torch.sum(E, dim=3) == 0
    
    # Set the first dimension (no-edge type) to 1 where there are no edges
    E_new[:, :, :, 0] = no_edge.float()
    
    # Copy existing edge features to the remaining dimensions
    E_new[:, :, :, 1:] = E
    
    # Set diagonal elements to 0 (no self-loops)
    diag = torch.eye(n, dtype=torch.bool, device=E.device).unsqueeze(0).expand(bs, -1, -1)
    E_new[diag] = 0
    
    return E_new

##################################################################################

def sample_discrete_features(probX, probE, node_mask):
    ''' Sample features from multinomial distribution with given probabilities (probX, probE, proby)
        :param probX: bs, n, dx_out        node features
        :param probE: bs, n, n, de_out     edge features
        :param proby: bs, dy_out           global features.
    
    Bascially, once a forward noise is applied, we get probabilities for nodes & edges
    (probX, probE) and we sample from these distributions to get back a *discrete* graph.

    Same goes for predictions made by the reverse diffusion model, we also need to sample to get a discrete graph.
    '''
    bs, n, _ = probX.shape
    # Noise X
    # The masked rows should define probability distributions as well
    probX[~node_mask] = 1 / probX.shape[-1]

    # Flatten the probability tensor to sample with multinomial
    probX = probX.reshape(bs * n, -1)       # (bs * n, dx_out)

    # Sample X
    X_t = probX.multinomial(1)                                  # (bs * n, 1)
    X_t = X_t.reshape(bs, n)     # (bs, n)

    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]

    probE = probE.reshape(bs * n * n, -1)    # (bs * n * n, de_out)

    # Sample E
    E_t = probE.multinomial(1).reshape(bs, n, n)   # (bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = (E_t + torch.transpose(E_t, 1, 2))

    return PlaceHolder(X=X_t, E=E_t, y=torch.zeros(bs, 0).type_as(X_t))

##################################################################################

class DistributionNodes:
    def __init__(self, histogram: dict):
        """
        Compute the distribution of the number of nodes in the dataset (heavy atoms), 
        and sample from this distribution.
        
        historgram: The keys are num_nodes, the values are counts
        """
        # histogram should be dict of int_keys, int_values
        # convert keys to int if they are strings
        histogram = {int(k): int(v) for k, v in histogram.items()}
        max_n_nodes = max(histogram.keys())
        prob = torch.zeros(max_n_nodes + 1)
        for num_nodes, count in histogram.items():
            prob[num_nodes] = count

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(prob)

    def sample_n(self, n_samples, device):
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def log_prob(self, batch_n_nodes):
        # NOTE: this is used in the validation step to compute the log probability of the number of nodes in the graph
        assert len(batch_n_nodes.size()) == 1
        p = self.prob.to(batch_n_nodes.device)

        probas = p[batch_n_nodes]
        log_p = torch.log(probas + 1e-6)
        return log_p

##################################################################################

def mask_distributions(true_X, true_E, pred_X, pred_E, node_mask):
    """
    Set masked rows to arbitrary distributions, so it doesn't contribute to loss
    :param true_X: bs, n, dx_out
    :param true_E: bs, n, n, de_out
    :param pred_X: bs, n, dx_out
    :param pred_E: bs, n, n, de_out
    :param node_mask: bs, n
    :return: same sizes as input

    Used in val loss kl_prior() & compute_Lt()

    Originally defined in DiffMS/src/diffusion/diffusion_utils.py
    """

    row_X = torch.zeros(true_X.size(-1), dtype=torch.float, device=true_X.device)
    row_X[0] = 1.
    row_E = torch.zeros(true_E.size(-1), dtype=torch.float, device=true_E.device)
    row_E[0] = 1.

    diag_mask = ~torch.eye(node_mask.size(1), device=node_mask.device, dtype=torch.bool).unsqueeze(0)
    # true_X[~node_mask] = row_X # RuntimeError ?
    # pred_X[~node_mask] = row_X
    # true_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E
    # pred_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E
    # Fix for GPU indexing issue - explicitly handle broadcasting
    node_mask_expanded = ~node_mask
    if node_mask_expanded.any():
        num_masked_nodes = node_mask_expanded.sum()
        true_X[node_mask_expanded] = row_X.unsqueeze(0).expand(num_masked_nodes, -1)
        pred_X[node_mask_expanded] = row_X.unsqueeze(0).expand(num_masked_nodes, -1)
    
    edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask)
    if edge_mask.any():
        num_masked_edges = edge_mask.sum()
        true_E[edge_mask] = row_E.unsqueeze(0).expand(num_masked_edges, -1)
        pred_E[edge_mask] = row_E.unsqueeze(0).expand(num_masked_edges, -1)
    # END OF FIX

    true_X = true_X + 1e-7
    pred_X = pred_X + 1e-7
    true_E = true_E + 1e-7
    pred_E = pred_E + 1e-7

    true_X = true_X / torch.sum(true_X, dim=-1, keepdim=True)
    pred_X = pred_X / torch.sum(pred_X, dim=-1, keepdim=True)
    true_E = true_E / torch.sum(true_E, dim=-1, keepdim=True)
    pred_E = pred_E / torch.sum(pred_E, dim=-1, keepdim=True)

    return true_X, true_E, pred_X, pred_E

def posterior_distributions(X, E, y, X_t, E_t, y_t, Qt, Qsb, Qtb):
    # NOTE: this is used in the validation step in compute_Lt()
    prob_X = compute_posterior_distribution(M=X, M_t=X_t, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X)   # (bs, n, dx)
    prob_E = compute_posterior_distribution(M=E, M_t=E_t, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E)   # (bs, n * n, de)

    return PlaceHolder(X=prob_X, E=prob_E, y=y_t)

def compute_posterior_distribution(M, M_t, Qt_M, Qsb_M, Qtb_M):
    ''' M: X or E
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T
    '''
    # Flatten feature tensors
    M = M.flatten(start_dim=1, end_dim=-2).to(torch.float32)        # (bs, N, d) with N = n or n * n
    M_t = M_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)    # same

    Qt_M_T = torch.transpose(Qt_M, -2, -1)      # (bs, d, d)

    left_term = M_t @ Qt_M_T   # (bs, N, d)
    right_term = M @ Qsb_M     # (bs, N, d)
    product = left_term * right_term    # (bs, N, d)

    denom = M @ Qtb_M     # (bs, N, d) @ (bs, d, d) = (bs, N, d)
    denom = (denom * M_t).sum(dim=-1)   # (bs, N, d) * (bs, N, d) + sum = (bs, N)
    # denom = product.sum(dim=-1)
    # denom[denom == 0.] = 1

    prob = product / denom.unsqueeze(-1)    # (bs, N, d)

    return prob

##################################################################################

def sample_discrete_feature_noise(limit_dist, node_mask):
    """
    Sample from the limit distribution of the diffusion process

    Used in sample_batch() for initial noisy graph G^T (with limit distribution)
    which is iteratively updated to get the clean graph G
    """
    bs, n_max = node_mask.shape
    x_limit = limit_dist.X[None, None, :].expand(bs, n_max, -1)
    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)
    # y_limit = limit_dist.y[None, :].expand(bs, -1)
    U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
    U_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max)
    U_y = torch.empty((bs, 0))

    long_mask = node_mask.long()
    U_X = U_X.type_as(long_mask)
    U_E = U_E.type_as(long_mask)
    U_y = U_y.type_as(long_mask)

    U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()
    U_E = F.one_hot(U_E, num_classes=e_limit.shape[-1]).float()

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = torch.zeros_like(U_E)
    indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    U_E = U_E * upper_triangular_mask
    U_E = (U_E + torch.transpose(U_E, 1, 2))

    assert (U_E == torch.transpose(U_E, 1, 2)).all()

    return PlaceHolder(X=U_X, E=U_E, y=U_y).mask(node_mask)

def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb):
    """ M: X or E
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
        X_t: bs, n, dt          or bs, n, n, dt
        Qt: bs, d_t-1, dt
        Qsb: bs, d0, d_t-1
        Qtb: bs, d0, dt.
    
        Used in sample_p_zs_given_zt(), originally defined in DiffMS/src/diffusion/diffusion_utils.py
    """
    # Flatten feature tensors
    # Careful with this line. It does nothing if X is a node feature. If X is an edge features it maps to
    # bs x (n ** 2) x d
    X_t = X_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)            # bs x N x dt

    Qt_T = Qt.transpose(-1, -2)                 # bs, dt, d_t-1
    left_term = X_t @ Qt_T                      # bs, N, d_t-1
    left_term = left_term.unsqueeze(dim=2)      # bs, N, 1, d_t-1

    right_term = Qsb.unsqueeze(1)               # bs, 1, d0, d_t-1
    numerator = left_term * right_term          # bs, N, d0, d_t-1

    X_t_transposed = X_t.transpose(-1, -2)      # bs, dt, N

    prod = Qtb @ X_t_transposed                 # bs, d0, N
    prod = prod.transpose(-1, -2)               # bs, N, d0
    denominator = prod.unsqueeze(-1)            # bs, N, d0, 1
    denominator[denominator == 0] = 1e-6

    out = numerator / denominator
    return out

##################################################################################

class MolecularVisualization:
    def __init__(self, atom_types: list[str]):
        self.atom_types = atom_types

    def mol_from_graphs(self, node_list, adjacency_matrix):
        """
        Convert graphs to rdkit molecules
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        # dictionary to map integer value to the char of atom
        atom_decoder = self.atom_types

        # create empty editable mol object
        mol = Chem.RWMol()

        # add atoms to mol and keep track of index
        node_to_idx = {}
        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            a = Chem.Atom(atom_decoder[int(node_list[i])])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx

        # add bonds to mol (w/ logic to add single/double/triple/aromatic bonds)
        for ix, row in enumerate(adjacency_matrix):
            for iy, bond in enumerate(row):
                # only traverse half the symmetric matrix
                if iy <= ix:
                    continue
                if bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                elif bond == 4:
                    bond_type = Chem.rdchem.BondType.AROMATIC
                else:
                    continue
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

        try:
            # NOTE: supposedly, this is where the hydrogen atoms are inferred by RDKit
            mol = mol.GetMol()
        except rdkit.Chem.KekulizeException:
            print("Can't kekulize molecule")
            mol = None
        return mol

    def visualize(self, path: str, molecules: list, num_molecules_to_visualize: int, log='graph'):
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        print(f"Visualizing {num_molecules_to_visualize} of {len(molecules)}")
        if num_molecules_to_visualize > len(molecules):
            print(f"Shortening to {len(molecules)}")
            num_molecules_to_visualize = len(molecules)
        
        for i in range(num_molecules_to_visualize):
            file_path = os.path.join(path, 'molecule_{}.png'.format(i))
            mol = self.mol_from_graphs(molecules[i][0].numpy(), molecules[i][1].numpy())
            try:
                Draw.MolToFile(mol, file_path)
                if wandb.run and log is not None:
                    print(f"Saving {file_path} to wandb")
                    wandb.log({log: wandb.Image(file_path)}, commit=True)
            except rdkit.Chem.KekulizeException:
                print("Can't kekulize molecule")