"""
Discrete diffusion decoder for molecular graph generation.

This part of the code is based on 
- `diffusion_model_fp2mol.py` from the DiffMS codebase, and
- `diffusion_model_discrete.py` and `guidance_diffusion_model_discrete.py` from the DiGress codebase.

We still need to implement the following:
- validation metric methods
- sampling methods for iterative denoising
- ...

We also need to implement the dataset statistics-related logic elsewhere (e.g. into Lightning DataModules)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from training.models.graph_transformer import GraphTransformer

from training.models.temp import TrainLossDiscrete
from training.diffusion import MarginalUniformTransition

from training.diffusion.discrete_noise import PredefinedNoiseScheduleDiscrete
from training.diffusion.augmentation import Augmentation

from training.diffusion.utils import PlaceHolder, sample_discrete_features

############################################################################
############################################################################
# Get dataset 'statistics'
if True:
    from rdkit import Chem
    from rdkit.Chem.rdchem import BondType as BT

    from collections import Counter

    from training.diffusion.utils import ATOM_TO_VALENCY, ATOM_TO_WEIGHT

    from training.diffusion.utils import DistributionNodes

    from training.diffusion.augmentation import ExtraFeatures, ExtraMolecularFeatures

    with open('smiles-data.txt', 'r') as file:
        smiles_strings = file.readlines()

    # convert SMILES strings to rdkit molecules
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_strings]

    # histogram of num_nodes (heavy atoms only) & their counts -> used to instantiate DistributionNodes
    num_nodes = [len(mol.GetAtoms()) for mol in molecules]
    hist_dict = Counter(num_nodes)
    nodes_dist = DistributionNodes(hist_dict)

    # max number of nodes (heavy atoms) -> used to instantiate ExtraFeatures
    max_n_nodes = max(num_nodes) # 31 (though actual model will use 64)

    # valencies, atom_weights and max_weight -> used to instantiate ExtraMolecularFeatures
    atom_decoder = ['C', 'O', 'P', 'N', 'S', 'Cl', 'F', 'H']
    atom_encoder = {atom: i for i, atom in enumerate(atom_decoder)}
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    valencies = [ATOM_TO_VALENCY.get(atom, 0) for atom in atom_decoder]
    atom_weights = {i: ATOM_TO_WEIGHT.get(atom, 0) for i, atom in enumerate(atom_decoder)}
    max_weight = max(atom_weights.values())

    def compute_marginals_from_molecules(molecules, atom_decoder, bonds):
        """
        Compute node and edge type marginals from RDKit molecules.
        
        Returns:
            node_marginals: torch.Tensor of shape (num_atom_types,) - probability of each atom type
            edge_marginals: torch.Tensor of shape (num_edge_types,) - probability of each edge type
        """
        # Initialize counters
        node_counts = torch.zeros(len(atom_decoder))  # 8 atom types
        edge_counts = torch.zeros(len(bonds) + 1)     # 5 edge types: [no_bond, single, double, triple, aromatic]
            
        for mol in molecules:
            n_atoms = mol.GetNumAtoms()
            
            # Count node types
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol in atom_encoder:
                    node_counts[atom_encoder[symbol]] += 1
            
            # Count edge types
            # First, count all possible edges for this molecule
            all_possible_pairs = n_atoms * (n_atoms - 1)  # All possible directed edges (excluding self-loops)
            
            # Count actual bonds
            actual_bonds = 0
            for bond in mol.GetBonds():
                bond_type = bond.GetBondType()
                if bond_type in bonds:
                    edge_counts[bonds[bond_type] + 1] += 2  # +1 because index 0 is reserved for no_bond, +2 for both directions
                    actual_bonds += 2
            
            # Count non-bonds (no_bond = 0)
            edge_counts[0] += all_possible_pairs - actual_bonds
        
        # Normalize to get marginals (probabilities)
        node_marginals = node_counts / node_counts.sum()
        edge_marginals = edge_counts / edge_counts.sum()
        
        return node_marginals, edge_marginals

    def compute_valency_distribution_from_molecules(molecules, max_n_nodes):
        """
        Compute empirical valency distribution from RDKit molecules.
        This is equivalent to the valency_count method in the original implementation.
        
        Returns:
            valency_distribution: torch.Tensor - probability distribution of valencies
        """
        # Initialize valency counts - max possible valency is 3 * max_n_nodes - 2 (if everything is connected)
        valency_counts = torch.zeros(3 * max_n_nodes - 2)
        
        # Bond order multipliers: [no_bond, single, double, triple, aromatic]
        bond_multipliers = {
            BT.SINGLE: 1.0,
            BT.DOUBLE: 2.0, 
            BT.TRIPLE: 3.0,
            BT.AROMATIC: 1.5
        }
        
        for mol in molecules:
            n_atoms = mol.GetNumAtoms()
            
            # For each atom, compute its valency
            for atom_idx in range(n_atoms):
                atom_valency = 0.0
                
                # Get all bonds connected to this atom
                atom = mol.GetAtomWithIdx(atom_idx)
                for bond in atom.GetBonds():
                    bond_type = bond.GetBondType()
                    if bond_type in bond_multipliers:
                        atom_valency += bond_multipliers[bond_type]
                
                # Add to valency counts (following original implementation)
                valency_idx = int(atom_valency)  # This truncates fractional valencies like the original
                if valency_idx < len(valency_counts):
                    valency_counts[valency_idx] += 1
        
        # Normalize to get probability distribution
        valency_distribution = valency_counts / valency_counts.sum()
        
        return valency_distribution

    x_marginals, e_marginals = compute_marginals_from_molecules(molecules, atom_decoder, bonds)
    valency_distribution = compute_valency_distribution_from_molecules(molecules, max_n_nodes)
    
    # Debug output
    print("=== DATASET STATISTICS ===")
    print(f"Number of molecules: {len(molecules)}")
    print(f"Max number of nodes: {max_n_nodes}")
    print(f"Max molecular weight: {max_weight}")
    
    print("\n=== NODE TYPE MARGINALS ===")
    for i, atom in enumerate(atom_decoder):
        print(f"  {atom}: {x_marginals[i]:.4f}")
    
    print("\n=== EDGE TYPE MARGINALS ===")
    bond_names = ['No bond', 'Single', 'Double', 'Triple', 'Aromatic']
    for i, bond_name in enumerate(bond_names):
        print(f"  {bond_name}: {e_marginals[i]:.4f}")
    
    print("\n=== VALENCY DISTRIBUTION ===")
    print(f"Shape: {valency_distribution.shape}")
    print(f"Non-zero valencies:")
    for i, prob in enumerate(valency_distribution):
        if prob > 0:
            print(f"  Valency {i}: {prob:.4f}")
    
    print("\n=== THEORETICAL VALENCIES ===")
    for i, atom in enumerate(atom_decoder):
        print(f"  {atom}: {valencies[i]}")
    
    print("="*50)
############################################################################
############################################################################

class DiscreteDiffusionDecoder(nn.Module):
    def __init__(
            self,
            graph_transformer_params = {
                "n_layers": 5,
                "input_dims": {"X": 16, "E": 5, "y": 1037},
                "output_dims": {"X": 8, "E": 5, "y": 1024},
                "hidden_mlp_dims": {"X": 256, "E": 128, "y": 1024},
                "hidden_dims": {'dx': 256, 'de': 64, 'dy': 512, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 1024}
            },
            dataset_infos = {
                "heavy_atom_counts": hist_dict,
                "max_heavy_atoms": max_n_nodes,
                "max_mol_weight": max_weight,
                "atom_weights": atom_weights,
                "valencies": valencies,
                # "valency_distribution": valency_distribution, # TODO: not sure if we need this here
                "x_marginals": x_marginals,
                "e_marginals": e_marginals
            },
            lambda_train = [1, 5, 0],
            diffusion_steps = 500
    ):
        super().__init__()
        self.model_dtype = torch.float32
        self.T = diffusion_steps
        self.best_val_nll = 1e8

        self.Xdim_input, self.Edim_input, self.ydim_input = graph_transformer_params["input_dims"]["X"], graph_transformer_params["input_dims"]["E"], graph_transformer_params["input_dims"]["y"]
        self.Xdim_output, self.Edim_output, self.ydim_output = graph_transformer_params["output_dims"]["X"], graph_transformer_params["output_dims"]["E"], graph_transformer_params["output_dims"]["y"]

        # TODO: we can possibly merge these two into a single class (Augmentation) ? Since we always use both
        # self.extra_graph_features = ExtraFeatures(max_n_nodes=dataset_infos["max_heavy_atoms"])
        # self.extra_molecular_features = ExtraMolecularFeatures(
        #     valencies=dataset_infos["valencies"],
        #     max_weight=dataset_infos["max_mol_weight"],
        #     atom_weights=dataset_infos["atom_weights"]
        # )
        self.augmentation = Augmentation(
            valencies=dataset_infos["valencies"],
            max_weight=dataset_infos["max_mol_weight"],
            atom_weights=dataset_infos["atom_weights"],
            max_n_nodes=dataset_infos["max_heavy_atoms"]
        )

        self.train_loss = TrainLossDiscrete(lambda_train)
        # TODO: validation / test metrics
        # there are many of them ... ELBO-related and reconstruction-related

        self.model = GraphTransformer(
            n_layers=graph_transformer_params['n_layers'],
            input_dims=graph_transformer_params['input_dims'],
            output_dims=graph_transformer_params['output_dims'],
            hidden_mlp_dims=graph_transformer_params['hidden_mlp_dims'],
            hidden_dims=graph_transformer_params['hidden_dims'],
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU()
        )
        
        # TODO: possibility to load pretrained model?? -> perhaps something to do at DTITree or Branch model-level (rather than decoder)

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(timesteps=diffusion_steps)

        self.transition_model = MarginalUniformTransition(
            x_marginals=dataset_infos["x_marginals"],
            e_marginals=dataset_infos["e_marginals"],
            y_classes=graph_transformer_params["output_dims"]["y"]
        )

        self.nodes_dist = DistributionNodes(dataset_infos["heavy_atom_counts"])

        self.limit_dist = PlaceHolder(
            X=dataset_infos["x_marginals"], 
            E=dataset_infos["e_marginals"], 
            y=torch.ones(graph_transformer_params["output_dims"]["y"]) / graph_transformer_params["output_dims"]["y"]
        )
    
    # TODO: add methods
    # - forward
    # - sample_batch
    # - sample_p_zs_given_zt
    # - compute_extra_data
    # - ...


    # PYTORCH LIGHTNING METHODS

    # GENERAL UTILITY METHODS
    def apply_noise(self, X, E, y, node_mask):
        """
        Sample noise and apply it to the data
        Args:
            X: Clean node features (bs, n, dx)
            E: Clean edge features (bs, n, n, de)  
            y: Graph-level features (fingerprints) (bs, dy)
            node_mask: Node mask (bs, n)
        Returns:
            dict: Noisy data with ...
        """
        # Sample timestep t (uniformly from [0, T])
        t_int = torch.randint( # (bs, 1)
            0, self.T + 1, size=(X.size(0), 1), device=X.device).float()
        s_int = t_int - 1

        # Normalize timesteps to [0, 1] for noise scheduler
        t_float = t_int / self.T
        s_float = s_int / self.T

        # Get noise schedule parameters 
        beta_t = self.noise_schedule(t_normalized=t_float)                    # (bs, 1) controls amount of noise at timestep t
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float) # (bs, 1) cumulative noise factor from 0 to s
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float) # (bs, 1) cumulative noise factor from 0 to t

        # Get transition matrices Q_t_bar for forward diffusion
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Forward diffusion: Compute transition probabilities
        probX = X @ Qtb.X               # (bs, n, dx_out)    - node transition probabilities p(X_t | X_0)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out) - edge transition probabilities p(E_t | E_0)

        # Sample discrete features from the transition probabilities
        sampled_t = sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)
        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask) # NOTE: no noise is applied to y!
        noisy_data = { # TODO: check if we need all these fields...
            't_int': t_int, 't': t_float, 'beta_t': beta_t, 
            'alpha_s_bar': alpha_s_bar, 'alpha_t_bar': alpha_t_bar, 
            'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data


    def compute_extra_data(self, noisy_data):
        """
        At every training step (after adding noise & sampling discrete features)
        extra features are computed (graph-structural features & molecular features)
        and appended to the network input.
        + timestep info t is added to graph-level features y to inform the model about the noise level
        """
        # TODO: may want to simplify this by using a single class (Augmentation) ? Since we will always use both
        # extra_graph_features = self.extra_graph_features(noisy_data)
        # extra_molecular_features = self.extra_molecular_features(noisy_data)

        # extra_X = torch.cat((extra_graph_features.X, extra_molecular_features.X), dim=-1)
        # extra_E = torch.cat((extra_graph_features.E, extra_molecular_features.E), dim=-1)
        # extra_y = torch.cat((extra_graph_features.y, extra_molecular_features.y), dim=-1)

        extra_features = self.augmentation(noisy_data)
        extra_X = extra_features.X
        extra_E = extra_features.E  
        extra_y = extra_features.y

        t = noisy_data['t'] # normalized timestep
        extra_y = torch.cat((extra_y, t), dim=1)

        return PlaceHolder(X=extra_X, E=extra_E, y=extra_y)


    def forward(self, noisy_data, extra_data, node_mask):
        """
        Forward pass through graph transformer
        Args:
            noisy_data: ...
            extra_data: ...
            node_mask: tensor of shape (batch_size, n_nodes)
        Returns:
            ...
        """
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)
    
