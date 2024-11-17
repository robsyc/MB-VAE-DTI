import logging

import networkx as nx
import numpy as np
import torch
from fuse.data.ops.op_base import OpBase
from fuse.utils.ndict import NDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

import bmfm_sm.core.data_modules.namespace as ns

# from https://github.com/snap-stanford/ogb/blob/master/ogb/utils/mol.py


def _safe_index(l, e):
    """Return index of element e in list l. If e is not present, return the last index."""
    try:
        return l.index(e)
    except:
        return len(l) - 1


def identity(x):
    return x


class OpSmilesToGraph(OpBase):
    # allowable node (atom) and edge (bond) features
    # Added 'misc' to end to handle out of bound cases (following https://github.com/snap-stanford/ogb/blob/master/ogb/utils/mol.py)
    allowable_features = {
        # ADDED 0 FOR CLINTOX
        "possible_atomic_num_list": list(range(0, 119)) + ["misc"],
        "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
        "possible_chirality_list": [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER,
            "misc",
        ],
        "possible_hybridization_list": [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED,
            "misc",
        ],
        "_ChiralityPossible": [0, 1],
        # Total number of explicit and implicit Hs
        "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
        "possible_implicit_valence_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
        # ADDED 10, 12 FOR HIV
        "possible_total_valence_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, "misc"],
        "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
        "possible_IsAromatic_list": [0, 1],
        "possible_radical_electron": list(np.arange(0, 20)) + ["misc"],
        "possible_GasteigerCharge": list(np.arange(0, 10)) + ["misc"],
        "possible_betti_0": list(np.arange(0, 128)),
        "possible_betti_1": list(np.arange(0, 128)),
        "possible_bonds": [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
            "misc",
        ],
        "possible_bond_dirs": [  # only for double bond stereo information
            Chem.rdchem.BondDir.NONE,
            Chem.rdchem.BondDir.ENDUPRIGHT,
            Chem.rdchem.BondDir.ENDDOWNRIGHT,
            "misc",
        ],
        "bond_is_conjugeated": [0, 1],
        "bond_is_ring": [0, 1],
        "bond_stereo": [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS,
            "misc",
        ],
    }

    def __init__(
        self,
        atom_features: list[str]
        | None = (
            "possible_atomic_num_list",
            "possible_chirality_list",
            "possible_degree_list",
            "possible_formal_charge_list",
            "possible_radical_electron",
            "possible_hybridization_list",
            "possible_IsAromatic_list",
            "possible_numH_list",
            "_ChiralityPossible",
        ),
        bond_features: list[str]
        | None = (
            "possible_bonds",
            "possible_bond_dirs",
            "bond_stereo",
            "bond_is_conjugeated",
            "bond_is_ring",
        ),
        embed_node_feats=None,
        embed_edge_feats=None,
        fixed_feat_embedding=None,
        node_feature_type=None,
        fixed_feat_embed_categories_max=None,
    ):
        """
        :param atom_features: atom features
        :param bond_features: bond features

        The default features are the TrimNet based 5 atom features and 2 bond feats
        ref: https://github.com/yvquanli/TrimNet/blob/2770dca2394d32af92819980215e0a211660a167/trimnet_drug/source/dataset.py#L143
        """
        super().__init__()

        # store input
        self.atom_feats = atom_features
        self.bond_feats = bond_features
        self.allowable_features = OpSmilesToGraph.allowable_features

        self.embed_node_feats = embed_node_feats
        self.embed_edge_feats = embed_edge_feats
        self.node_feature_type = node_feature_type
        self.fixed_feat_embedding = fixed_feat_embedding
        self.fixed_feat_embed_categories_max = fixed_feat_embed_categories_max
        self.x_norm_func = identity

    def __call__(self, sample_dict: NDict, key_in, key_out) -> NDict:
        """Converts smiles code to a graph."""
        # Get the smile code of interest
        smiles_codes = sample_dict[key_in]

        # Generate the graph representation (smiles -> molecule -> graph)
        try:
            mol = MolFromSmiles(smiles_codes)
            graph = self._mol2graph(mol)
        except Exception as e:
            # If there is a problem, use Data = None
            graph = None
            logging.warning(
                'Problem creating graph for "'
                + str(smiles_codes)
                + '"; '
                + str(e)
                + "; skipping."
            )

        # Store the graph representation
        sample_dict[key_out] = graph

        # Return dictionary
        return sample_dict

    def _mol_to_nx(self, mol):
        atoms_list = []
        edges_list = []
        for atom in mol.GetAtoms():
            atoms_list.append(atom.GetIdx())
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges_list.append((i, j))
            edges_list.append((j, i))
        g = nx.Graph()
        g.add_nodes_from(atoms_list)
        g.add_edges_from(edges_list)
        return g

    def _getHomology(self, mol, atom):
        node = atom.GetIdx()
        g = self._mol_to_nx(mol)
        g.remove_node(node)
        betti_0 = nx.number_connected_components(g)
        betti_1 = len(nx.cycle_basis(g))
        return betti_0, betti_1

    def _mol2graph(self, mol):
        """
        from TrimNet
        ref: https://github.com/yvquanli/TrimNet/blob/2770dca2394d32af92819980215e0a211660a167/trimnet_drug/source/dataset.py#L354
        returns a dictionary with
            - x: node features
            - edge_attr: edge features
            - edge_index: indices for the edges
            - y: label.
        """
        if mol is None:
            return None
        node_attr = self._atom_attr(mol)
        x = torch.FloatTensor(node_attr)

        if self.embed_node_feats == True:
            if self.node_feature_type == "cate":  # true
                if self.fixed_feat_embedding == True:
                    x = convert_to_single_emb(
                        x, offset=self.fixed_feat_embed_categories_max
                    )
                else:  # Do things for compact embedding...
                    indices = [0]
                    for feat in self.bond_feats:  # node_feat
                        assert feat in self.allowable_features.keys()  # allowable_feats
                        indices.append(len(self.allowable_features[feat]))
                    # what does this do? The code won't go in
                    x = convert_to_single_emb(x, feature_offset=indices[:-1])
            elif self.node_feature_type == "dense":
                x = self.x_norm_func(x)
            else:
                raise ValueError("node feature type error")

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_index, edge_attr = self._bond_attr(mol)
        edge_index = torch.LongTensor(edge_index).t()
        edge_attr = torch.FloatTensor(edge_attr)
        if self.embed_edge_feats == True:
            if len(edge_attr.size()) == 1:
                edge_attr = edge_attr[:, None]
            edge_attr = convert_to_single_emb(
                edge_attr, offset=self.fixed_feat_embed_categories_max
            )

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=None,  # None as a placeholder
        )
        return data

    def _atom_attr(self, mol, explicit_H=True, use_chirality=True):
        """
        Converts rdkit mol object to graph Data object required by the pytorch geometric package.
        :param mol: rdkit mol object
        :return: graph data object with the attributes: x, edge_index, edge_attr
        The default features are the trimNet based 5 atom features and 2 bond feats.
        """
        if explicit_H == False:
            self.atom_feats = (
                self.atom_feats.remove("possible_numH_list")
                if "possible_numH_list" in self.atom_feats
                else self.atom_feats
            )
        if use_chirality == False:
            self.atom_feats = (
                self.atom_feats.remove("possible_chirality_list")
                if "possible_chirality_list" in self.atom_feats
                else self.atom_feats
            )
            self.atom_feats = (
                self.atom_feats.remove("_ChiralityPossible")
                if "_ChiralityPossible" in self.atom_feats
                else self.atom_feats
            )

        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features = []
            for feat in self.atom_feats:
                if feat not in self.allowable_features.keys():
                    atom_features.append(np.nan)
                elif feat == "possible_atomic_num_list":
                    atom_features.append(
                        _safe_index(self.allowable_features[feat], atom.GetAtomicNum())
                    )
                elif feat == "possible_chirality_list":
                    atom_features.append(
                        _safe_index(self.allowable_features[feat], atom.GetChiralTag())
                    )
                elif feat == "_ChiralityPossible":
                    atom_features.append(
                        self.allowable_features[feat].index(
                            int(atom.HasProp("_ChiralityPossible"))
                        )
                    )
                elif feat == "possible_formal_charge_list":
                    atom_features.append(
                        _safe_index(
                            self.allowable_features[feat], atom.GetFormalCharge()
                        )
                    )
                elif feat == "possible_hybridization_list":
                    atom_features.append(
                        _safe_index(
                            self.allowable_features[feat], atom.GetHybridization()
                        )
                    )
                elif feat == "possible_numH_list":
                    atom_features.append(
                        _safe_index(self.allowable_features[feat], atom.GetTotalNumHs())
                    )
                elif feat == "possible_implicit_valence_list":
                    atom_features.append(
                        _safe_index(
                            self.allowable_features[feat], atom.GetImplicitValence()
                        )
                    )
                elif feat == "possible_total_valence_list":
                    atom_features.append(
                        _safe_index(
                            self.allowable_features[feat], atom.GetTotalValence()
                        )
                    )
                elif feat == "possible_degree_list":
                    atom_features.append(
                        _safe_index(
                            self.allowable_features[feat], atom.GetTotalDegree()
                        )
                    )
                elif feat == "possible_IsAromatic_list":
                    atom_features.append(
                        self.allowable_features[feat].index(int(atom.GetIsAromatic()))
                    )
                elif feat == "possible_radical_electron":
                    atom_features.append(
                        _safe_index(
                            self.allowable_features[feat], atom.GetNumRadicalElectrons()
                        )
                    )
                # elif feat == 'possible_GasteigerCharge':
                #     atom_features.append(allowable_features[feat].index(float(atom.GetProp('_GasteigerCharge'))))
                elif feat == "possible_betti_0" or feat == "possible_betti_1":
                    [betti_0, betti_1] = self._getHomology(mol, atom)
                    betti = betti_0 if feat == "possible_betti_0" else betti_1
                    value = (
                        self.allowable_features[feat].index(betti)
                        if betti <= self.allowable_features[feat][-1]
                        else self.allowable_features[feat].index(
                            self.allowable_features[feat][-1]
                        )
                    )
                    atom_features.append(value)
                else:
                    atom_features.append(np.nan)
                    raise NotImplementedError

            atom_features_list.append(atom_features)

        return np.array(atom_features_list)

    def _bond_attr(self, mol, use_chirality=True):
        """
        Converts rdkit mol object to graph Data object required by the pytorch geometric package.
        :param mol: rdkit mol object
        :return: graph data object with the attributes: x, edge_index, edge_attr
        The default features are the trimNet based 5 atom features and 2 bond feats
        ref: https://github.com/yvquanli/TrimNet/blob/2770dca2394d32af92819980215e0a211660a167/trimnet_drug/source/dataset.py#L143
        ref: https://github.com/yvquanli/TrimNet/blob/2770dca2394d32af92819980215e0a211660a167/trimnet_drug/source/dataset.py#L177.
        """
        # Initialize results
        edges_list = []
        edge_features_list = []

        if use_chirality == False:
            self.bond_feats = (
                self.bond_feats.remove("bond_stereo")
                if "bond_stereo" in self.bond_feats
                else self.bond_feats
            )

        num_bond_features = len(self.bond_feats)  # bond type, bond direction

        if len(mol.GetBonds()) > 0:  # mol has bonds
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_features = []
                for feat in self.bond_feats:
                    if feat not in self.allowable_features.keys():
                        edge_features.append(np.nan)
                    elif feat == "possible_bonds":
                        edge_features.append(
                            _safe_index(
                                self.allowable_features[feat], bond.GetBondType()
                            )
                        )
                    elif feat == "possible_bond_dirs":
                        edge_features.append(
                            _safe_index(
                                self.allowable_features[feat], bond.GetBondDir()
                            )
                        )
                    elif feat == "bond_is_conjugeated":
                        edge_features.append(
                            self.allowable_features[feat].index(
                                int(bond.GetIsConjugated())
                            )
                        )
                    elif feat == "bond_is_ring":
                        edge_features.append(
                            self.allowable_features[feat].index(int(bond.IsInRing()))
                        )
                    elif feat == "bond_stereo":
                        edge_features.append(
                            _safe_index(self.allowable_features[feat], bond.GetStereo())
                        )

                edges_list.append((i, j))
                edge_features_list.append(edge_features)
                edges_list.append((j, i))
                edge_features_list.append(edge_features)
                edge_index = np.array(edges_list)
                edge_attr = np.array(edge_features_list)

        else:  # mol has no bonds
            edge_index = np.empty((0, 2))
            edge_attr = np.empty((0, num_bond_features))

        return edge_index, edge_attr


# Generate graph laplacian (for Graph2d use)
class OpGraphToGraphLaplacian(OpBase):
    def __init__(self, use_corrupt=True):
        super().__init__()
        self.use_corrupt = use_corrupt

    def __call__(self, sample_dict: NDict, key_in, key_out) -> NDict:
        """Converts molecule graph to a graph laplacian."""
        # Get the graph representation
        data = sample_dict[key_in]

        # Generate the graph laplacian
        try:
            # Get graph info: size, edge index, adjancy matrix, in_degree
            N = data.x.size(0)
            edge_index = data.edge_index if self.use_corrupt else data.orig_edge_index
            dense_adj = self._get_adj_mat(N, edge_index)
            in_degree = dense_adj.long().sum(dim=1).view(-1)

            # Compute laplacian eigenvector and eigenvalue
            lap_eigvec, lap_eigval = lap_eig(dense_adj, N, in_degree)  # [N, N], [N,]
            lap_eigval = lap_eigval[None, :].expand_as(lap_eigvec)

            # Store the laplacian eigenvector and eigenvalue
            data.lap_eigvec = lap_eigvec
            data.lap_eigval = lap_eigval

        except Exception as e:
            # If there is a problem, use None (to skip the entire data item)
            logging.warning(
                f"Problem creating graph laplacian for {sample_dict}; {str(e)} ; skipping."
            )
            data = None

        # Store the eigenvector and eigenvalue
        sample_dict[key_out] = data

        # Return dictionary
        return sample_dict

    def _get_adj_mat(self, N, edge_index, self_loop=False):
        # node adj matrix [N, N] bool
        edge_index = to_undirected(edge_index)
        adj = torch.zeros([N, N], dtype=torch.bool)
        adj[edge_index[0, :], edge_index[1, :]] = True

        if self_loop == True:
            adj_w_sl = adj.clone()  # adj with self loop
            adj_w_sl[torch.arange(N), torch.arange(N)] = 1
            return adj_w_sl
        else:
            return adj


def eig(sym_mat):
    # (sorted) eigenvectors with numpy
    # ref: https://github.com/jw9730/tokengt/blob/d2aba6d0998ab276e2f6cac9b09c4d6feccb7d0f/large-scale-regression/tokengt/data/algos.py#L17
    EigVal, EigVec = np.linalg.eigh(sym_mat)

    # for eigval, take abs because numpy sometimes computes the first eigenvalue approaching 0 from the negative
    eigvec = torch.from_numpy(EigVec).float()  # [N, N (channels)]
    eigval = torch.from_numpy(
        np.sort(np.abs(np.real(EigVal)))
    ).float()  # [N (channels),]
    return eigvec, eigval  # [N, N (channels)]  [N (channels),]


def lap_eig(dense_adj, number_of_nodes, in_degree):
    """
    ref: https://github.com/jw9730/tokengt/blob/d2aba6d0998ab276e2f6cac9b09c4d6feccb7d0f/large-scale-regression/tokengt/data/algos.py#L17
    Graph positional encoding v/ Laplacian eigenvectors
    https://github.com/DevinKreuzer/SAN/blob/main/data/molecules.py.
    """
    dense_adj = dense_adj.detach().float().numpy()
    in_degree = in_degree.detach().float().numpy()

    # Laplacian
    A = dense_adj
    N = np.diag(in_degree.clip(1) ** -0.5)
    L = np.eye(number_of_nodes) - N @ A @ N

    eigvec, eigval = eig(L)
    return eigvec, eigval  # [N, N (channels)]  [N (channels),]


# SD: UPdated to make feature offset to make smarter embedding... JIT does not work
# @torch.jit.script


# CHANGE TO 128 if you want to reduce vocab size. But you should also change the num_atoms arguments as well.
def convert_to_single_emb(x, offset: int = 128, feature_offset=None):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    if feature_offset == None:
        feature_offset = 1 + torch.arange(
            0, feature_num * offset, offset, dtype=torch.long
        )
    else:
        assert len(feature_offset) == x.shape[-1]
        feature_offset = 1 + torch.tensor(feature_offset)
    x = x + feature_offset
    return x


class OpMaskGraphEdges(OpBase):
    def __init__(self, mask_prob=0.15):
        super().__init__()
        self.mask_prob = mask_prob

    def __call__(
        self,
        sample_dict: NDict,
        key_in=ns.FIELD_LABELS_GRAPH_DATA,
        key_out=ns.FIELD_LABELS_GRAPH_DATA,
    ) -> NDict:
        data = sample_dict[key_in]

        # Saving the original edge index
        data.orig_edge_index = torch.clone(data.edge_index)

        # Which edges will be corrupted
        mask = torch.rand(data.edge_index.shape[1]) < self.mask_prob
        random_sources = torch.randint(
            0, torch.max(data.edge_index) + 1, (torch.sum(mask),)
        )
        random_targets = torch.randint(
            0, torch.max(data.edge_index) + 1, (torch.sum(mask),)
        )

        # Ensure that we don't create self-loops
        while torch.any(random_sources == random_targets):
            overlap = random_targets == random_sources
            random_targets[overlap] = torch.randint(
                0, torch.max(data.edge_index) + 1, (torch.sum(overlap),)
            )

        # Set the new, corrupted sources and targets for the chosen edges
        data.edge_index[0][mask] = random_sources
        data.edge_index[1][mask] = random_targets

        data.edge_corrupt = mask.to(torch.int)

        sample_dict[key_out] = data
        return sample_dict


# Generate rank homology (for Graph2d pretraining)
class OpGraphToRankHomology(OpBase):
    def __init__(self):
        pass

    def _mol_to_nx(self, mol):
        atoms_list = []
        edges_list = []
        for atom in mol.GetAtoms():
            atoms_list.append(atom.GetIdx())
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges_list.append((i, j))
            edges_list.append((j, i))
        g = nx.Graph()
        g.add_nodes_from(atoms_list)
        g.add_edges_from(edges_list)
        return g

    def _getHomology(self, mol, atom):
        node = atom.GetIdx()
        g = self._mol_to_nx(mol)
        g.remove_node(node)
        betti_0 = nx.number_connected_components(g)
        betti_1 = len(nx.cycle_basis(g))
        return betti_0, betti_1

    def __call__(self, sample_dict: NDict, key_in, key_out) -> NDict:
        """Converts molecule graph to a rank homology (betti numbers)."""
        # Initialize result
        rank_homology_list = []

        # Get the smile code of interest
        smiles_codes = sample_dict[key_in]

        # Get the molecule
        mol = MolFromSmiles(smiles_codes)

        # Loop over atoms and generate betti numbers (pair) for each atom
        for atom in mol.GetAtoms():
            betti_numbers = self._getHomology(mol, atom)
            rank_homology_list.append(betti_numbers)

        # Store the result in new field
        sample_dict[key_out] = rank_homology_list

        # Return dictionary
        return sample_dict
