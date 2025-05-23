import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_params(model):
    return 0 if model is None else sum(p.numel() for p in model.parameters() if p.requires_grad)

@dataclass
class DrugTargetOutput:
    """Container for DrugTargetTree outputs.
    
    Args:
        predictions: Tensor of interaction predictions
        kl_loss: Optional KL divergence loss for variational models
    """
    predictions: torch.Tensor
    kl_loss: Optional[torch.Tensor] = None


class MoleculeGenerator(nn.Module):
    """
    Generator block for molecule generation.

    Args:
        - latent_dim: int, the dimensionality of the input tensor
        - hidden_dim: int, the dimensionality of the hidden layer(s)
        - depth: int, the number of hidden layers & complexity of the model (residual connections)
        - dropout_prob: float, the dropout probability
        - n_nodes: int, the number of nodes in the graph
        - n_iters: int, the number of iterations for the Gumbel-Softmax sampling
    """
    # TODO, See: 
    # - https://pygmtools.readthedocs.io/en/latest/guide/numerical_backends.html#example-matching-isomorphic-graphs-with-pytorch-backend
    # - https://pygmtools.readthedocs.io/en/latest/api/_autosummary/pygmtools.utils.build_aff_mat.html#pygmtools.utils.build_aff_mat
    def __init__(self, latent_dim, hidden_dim, depth, dropout_prob, n_nodes, n_iters):
        super(MoleculeGenerator, self).__init__()
        self.depth = depth
        self.n_nodes = n_nodes
        self.n_iters = n_iters

        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.ELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(depth)
        ])
        self.hidden2topology = nn.Linear(hidden_dim, 2 * n_nodes * (n_nodes - 1) // 2)
        self.gumbel2hidden = nn.Sequential(
            nn.Linear(n_nodes * (n_nodes - 1) // 2, hidden_dim),
            nn.ELU(),
        )
        # TODO node and edge features
        # self.node_feature_gen = nn.Linear(hidden_dim, n_nodes)
        # self.edge_feature_gen = nn.Linear(hidden_dim, n_nodes * (n_nodes - 1) // 2)

    def forward(self, x):
        """
        Args:
            - x: torch.Tensor, the input tensor of shape (batch_size, latent_dim)
        Returns:
            - adj: torch.Tensor, the output tensor of shape (batch_size, n_nodes, n_nodes)
        """
        x = self.latent2hidden(x)
        for _ in range(self.n_iters):
            for residual in self.residual_blocks:
                x = x + residual(x)  # Residual connection            
            gb = self._gumbel_softmax(self.hidden2topology(x))  # Sample topology
            x = x + self.gumbel2hidden(gb)  # Integrate topology feedback
        return self._generate_adj(self.hidden2topology(x))

    def _gumbel_softmax(self, x):
        x = torch.reshape(x, (x.size(0), -1, 2)) # 2 halves of symmetric adjacency matrix
        return F.gumbel_softmax(x, tau=1, hard=True)[:,:,0] # mash both halves together & sample binary values

    def _generate_adj(self, x):
        """Convert sampled topology to adjacency matrix."""
        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:, idx[0], idx[1]] = self._gumbel_softmax(x)  # Fill upper triangular part
        adj = adj + adj.transpose(1, 2)  # Symmetrize
        return adj # (batch_size, n_nodes, n_nodes)


class ResidualMLP(nn.Module):
    """
    MLP with residual connections, LayerNorm, SiLU/ELU activation, and dropout.

    Args:
        - input_dim: int, the dimension of the input tensor
        - hidden_dim: int, the dimension of the hidden layer(s)
        - output_dim: int, the dimension of the output tensor
        - depth: int, the number of hidden residual layers
        - dropout_prob: float, the dropout probability (default: 0.1)
        - activation: str, the activation function (default: 'SiLU')
        - layer_scale: float, the value by which residual outputs are initially scaled
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            depth: int,
            dropout_prob: float = 0.1,
            activation: str = 'GELU',
            layer_scale: float = 1e-4,
    ):
        super(ResidualMLP, self).__init__()
        activation_map = {
            'GELU': nn.GELU(),
            'SiLU': nn.SiLU(),
            'ELU': nn.ELU(),
            'ReLU': nn.ReLU()
        }
        if activation not in activation_map:
            raise ValueError(f"Activation '{activation}' not implemented. Available options: {list(activation_map.keys())}")
        self.activation = activation_map[activation]
        
        self.input2hidden = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim, bias=False),
            self.activation,
            nn.Dropout(dropout_prob),
        )

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 4 * hidden_dim, bias=False),
                self.activation,
                nn.Dropout(dropout_prob),
                nn.Linear(4 * hidden_dim, hidden_dim, bias=False),
            ) for _ in range(depth)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(depth)])
        self.layer_scales = nn.ParameterList([
            nn.Parameter(torch.ones(hidden_dim) * layer_scale) for _ in range(depth)])
        
        self.hidden2output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: torch.Tensor, the input tensor of shape (batch_size, input_dim)
        Returns:
            - x: torch.Tensor, the output tensor of shape (batch_size, output_dim)
        """
        x = self.input2hidden(x)

        for layer, scale, norm in zip(self.hidden_layers, self.layer_scales, self.layer_norms):
            residual = layer(x)
            residual = residual * scale.view(1, -1)
            x = norm(x + residual)

        return self.hidden2output(x)


class DrugTargetTree(nn.Module):
    """
    Drug-target interaction prediction model using a tree-based architecture.

    Args:
        - drug_dims: list, the dimension(s) of the drug input tensor(s)
        - target_dims: list, the dimension(s) of the target input tensor(s)
        - hidden_dim: int, the dimension of the hidden layer(s)
        - latent_dim: int, the dimension of the latent space
        - depth: int, the number of hidden layers & complexity of the model (residual connections)
        - dropout_prob: float, the dropout probability (default: 0.1)
        - variational: bool, whether to use variational inference (default: False)
    """
    def __init__(
            self,
            drug_dims: list,
            target_dims: list,
            hidden_dim: int,
            latent_dim: int,
            depth: int,
            dropout_prob: float,
            variational: bool
    ):
        super(DrugTargetTree, self).__init__()
        self.drug_multiview = True if len(drug_dims) > 1 else False
        self.target_multiview = True if len(target_dims) > 1 else False

        self.variational = variational
        self.output_dim = 2*latent_dim if variational else latent_dim

        # Initialize drug and target leafs
        #   Each leaf is a ResidualMLP that encodes a view of the drug or target
        #   from their respective input dimensions into a hidden space of dimension hidden_dim
        self.drug_leafs = nn.ModuleList([
            ResidualMLP(drug_dim, hidden_dim, hidden_dim, depth, dropout_prob)
            for drug_dim in drug_dims
        ]) if self.drug_multiview else ResidualMLP(drug_dims[0], hidden_dim, self.output_dim, depth, dropout_prob)
        self.target_leafs = nn.ModuleList([
            ResidualMLP(target_dim, hidden_dim, hidden_dim, depth, dropout_prob)
            for target_dim in target_dims
        ]) if self.target_multiview else ResidualMLP(target_dims[0], hidden_dim, self.output_dim, depth, dropout_prob)

        # Initialize branches
        #   Each branch is a ResidualMLP that encodes the concatenation of all views of the drug or target
        #   into a single latent space of dimension self.output_dim
        self.drug_branch = ResidualMLP(
            hidden_dim * len(drug_dims), hidden_dim, self.output_dim, depth, dropout_prob
            ) if self.drug_multiview else None
        self.target_branch = ResidualMLP(
            hidden_dim * len(target_dims), hidden_dim, self.output_dim, depth, dropout_prob
            ) if self.target_multiview else None

        p_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"""Number of parameters
- Drug   leafs & branch: {get_model_params(self.drug_leafs):,}, {get_model_params(self.drug_branch):,}
- Target leafs & branch: {get_model_params(self.target_leafs):,}, {get_model_params(self.target_branch):,}
- Total: {p_trainable:,}""")

    def forward(self, drug_inputs, target_inputs, compute_kl_loss = False):
        """
        Args:
            - drug_inputs: list of torch.Tensor, the input tensor(s) of the drug-branch
            - target_inputs: list of torch.Tensor, the input tensor(s) of the target-branch
            - compute_kl_loss: bool, whether to compute the KL divergence loss (default: False)
        Returns:
            - y_hat: torch.Tensor, the output tensor of shape (batch_size, 1)
        """
        # Encode drug and target inputs to same hidden space
        #   Both: n_views x (batch_size, view_dim) -> (batch_size, n_views * hidden_dim)
        #   Concatenate ensures that the latent spaces are aligned along the last dimension
        drug_latents = torch.cat(
            [leaf(drug_input) for leaf, drug_input in zip(self.drug_leafs, drug_inputs)], dim=-1
            ) if self.drug_multiview else self.drug_leafs(drug_inputs[0])
        target_latents = torch.cat(
            [leaf(target_input) for leaf, target_input in zip(self.target_leafs, target_inputs)], dim=-1
            ) if self.target_multiview else self.target_leafs(target_inputs[0])

        # Encode drug and target latents to output space
        #   Both: (batch_size, n_views * hidden_dim) -> (batch_size, output_dim)
        drug_latents = self.drug_branch(drug_latents) if self.drug_multiview else drug_latents
        target_latents = self.target_branch(target_latents) if self.target_multiview else target_latents

        # In case of variational inference, reparameterize the latent space
        #   Both: (batch_size, 2*latent_dim) -> (batch_size, latent_dim)
        if self.variational:
            drug_mean, drug_logvar = torch.chunk(drug_latents, 2, dim=1)
            target_mean, target_logvar = torch.chunk(target_latents, 2, dim=1)
            drug_latents = self._reparameterize(drug_mean, drug_logvar)
            target_latents = self._reparameterize(target_mean, target_logvar)

        # Compute interaction output (dot product)
        #   (batch_size, latent_dim) * (batch_size, latent_dim) -> (batch_size, 1)
        #   Dot product gives a single interaction score for each drug and target
        y_hat = torch.sum(drug_latents * target_latents, dim=1)
        
        kl_loss = self._kl_divergence(drug_mean, drug_logvar) + self._kl_divergence(target_mean, target_logvar) if compute_kl_loss and self.variational else None

        return DrugTargetOutput(
            predictions=y_hat,
            kl_loss=kl_loss
        )

    def _reparameterize(self, mean, logvar):
        """
        Args:
            - mean: torch.Tensor, the mean tensor of shape (batch_size, latent_dim)
            - std: torch.Tensor, the standard-deviations of shape (batch_size, latent_dim)
        Returns:
            - z: torch.Tensor, the output tensor of shape (batch_size, latent_dim) after reparameterization
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # sample epsilon from N(0, 1)
        return mean + eps * std

    def _kl_divergence(self, mean, logvar):
        """
        Args:
            - mean: torch.Tensor, the mean tensor of shape (batch_size, latent_dim)
            - logvar: torch.Tensor, the log-variance tensor of shape (batch_size, latent_dim)
        Returns:
            - kl_loss: torch.Tensor, the KL divergence loss regularizing the latent space into a standard normal distribution
        """
        return - ( 
            0.5 * torch.sum(1 + logvar - mean**2 - logvar.exp()) 
            ) / mean.size(1)




# class VariationalDrugTargetTree(nn.Module):
#     """
#     Variational drug-target interaction prediction model using a tree-based architecture.

#     Args:
#         - drug_dims: list, the dimension(s) of the drug input tensor(s)
#         - target_dims: list, the dimension(s) of the target input tensor(s)
#         - hidden_dim: int, the dimension of the hidden layer(s)
#         - latent_dim: int, the dimension of the latent space
#         - depth: int, the number of hidden layers & complexity of the model (residual connections)
#         - dropout_prob: float, the dropout probability (default: 0.1)
#     """
#     def __init__(
#             self,
#             drug_dims: list,
#             target_dims: list,
#             hidden_dim: int,
#             latent_dim: int,
#             depth: int = 0,
#             dropout_prob: float = 0.1,
#     ):
#         super(VariationalDrugTargetTree, self).__init__()
#         self.drug_multiview = True if len(drug_dims) > 1 else False
#         self.target_multiview = True if len(target_dims) > 1 else False

#         # Initialize drug and target leafs
#         #   Same as before, but we project to means & logvars of multivariate Gaussian
#         self.drug_leafs = nn.ModuleList([
#             ResidualMLP(drug_dim, hidden_dim, 2 * latent_dim, depth, dropout_prob)
#             for drug_dim in drug_dims
#         ])
#         self.target_leafs = nn.ModuleList([
#             ResidualMLP(target_dim, hidden_dim, 2 * latent_dim, depth, dropout_prob)
#             for target_dim in target_dims
#         ])

#         # Initialize attentive branches
#         #   Similar as before, but we incorporate uncertainty into the attn mechanism
#         #   by treating the means and logvars seperately
#         p_attn = 0
#         if self.drug_multiview:
#             self.drug_means_to_attn_logits = nn.Conv1d(latent_dim, 1, 1)
#             self.drug_logvars_to_attn_logits = nn.Conv1d(latent_dim, 1, 1)
#             p_attn += get_model_params(self.drug_means_to_attn_logits) + get_model_params(self.drug_logvars_to_attn_logits)
#         if self.target_multiview:
#             self.target_means_to_attn_logits = nn.Conv1d(latent_dim, 1, 1)
#             self.target_logvars_to_attn_logits = nn.Conv1d(latent_dim, 1, 1)
#             p_attn += get_model_params(self.target_means_to_attn_logits) + get_model_params(self.target_logvars_to_attn_logits)
#         self.softplus = nn.Softplus()

#         p_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         print(f"Number of parameters\n - Drug Leafs: {get_model_params(self.drug_leafs):,}\n - Target Leafs: {get_model_params(self.target_leafs):,}\n - Attention: {p_attn:,}\n - Total: {p_trainable:,}")

#     def forward(self, drug_inputs, target_inputs, compute_kl_loss = False):
#         """
#         Args:
#             - drug_inputs: list of torch.Tensor, the input tensor(s) of the drug-branch
#             - target_inputs: list of torch.Tensor, the input tensor(s) of the target-branch
#             - compute_kl_loss: bool, whether to compute the KL divergence loss (default: False)
#         Returns:
#             - y_hat: torch.Tensor, the output tensor of shape (batch_size, 1)
#             - kl_loss: torch.Tensor, the KL divergence loss
#         """
#         # Encode drug and target inputs to same hidden space and chunk in means & logvars
#         #   Both: n_views x (batch_size, view_dim) 
#         #           -> (batch_size, 2*latent_dim, n_views) 
#         #           -> 2 * (batch_size, latent_dim, n_views) 
#         drug_mean, drug_logvar = torch.chunk(
#             torch.stack([leaf(drug_input) for leaf, drug_input in zip(self.drug_leafs, drug_inputs)], dim=-1), 2, dim=1
#             ) if self.drug_multiview else torch.chunk(self.drug_leafs[0](drug_inputs[0]), 2, dim=1)
#         target_mean, target_logvar = torch.chunk(
#             torch.stack([leaf(target_input) for leaf, target_input in zip(self.target_leafs, target_inputs)], dim=-1), 2, dim=1
#             ) if self.target_multiview else torch.chunk(self.target_leafs[0](target_inputs[0]), 2, dim=1)
        
#         # Compute view-wise uncertainty-aware attention weights
#         #   Both: 2 * (batch_size, latent_dim, n_views) -> (batch_size, 1, n_views) 
#         drug_attn = (
#             self.drug_means_to_attn_logits(drug_mean) - self.softplus(self.drug_logvars_to_attn_logits(drug_logvar))
#         ).softmax(dim=-1) if self.drug_multiview else None
#         target_attn = (
#             self.target_means_to_attn_logits(target_mean) - self.softplus(self.target_logvars_to_attn_logits(target_logvar))
#         ).softmax(dim=-1) if self.target_multiview else None

#         # Attention-based fusion w/ helper function
#         #   Both: -> 2 * (batch_size, latent_dim)
#         drug_mean, drug_logvar = self._gaussian_fusion(
#             drug_mean, drug_logvar, drug_attn
#             ) if self.drug_multiview else (drug_mean, drug_logvar)
#         target_mean, target_logvar = self._gaussian_fusion(
#             target_mean, target_logvar, target_attn
#             ) if self.target_multiview else (target_mean, target_logvar)

#         # Reparameterize to obtain latent sample
#         #   Both: -> (batch_size, latent_dim)
#         drug_latents = self._reparameterize(drug_mean, drug_logvar)
#         target_latents = self._reparameterize(target_mean, target_logvar)

#         # Compute interaction output (dot product)
#         #   (batch_size, latent_dim) * (batch_size, latent_dim) -> (batch_size, 1)
#         y_hat = torch.sum(drug_latents * target_latents, dim=1)
        
#         if not compute_kl_loss:
#             return y_hat
#         else:
#             kl_loss = self._kl_divergence(drug_mean, drug_logvar) + self._kl_divergence(target_mean, target_logvar)
#             return y_hat, kl_loss

#     def _gaussian_fusion(self, means, logvars, weights, eps = 1e-6):
#         """
#         Naive fusion of means and logvars w/ attention weights using log-sum-exp trick.

#         Args:
#             - means: torch.Tensor, the means tensor of shape (batch_size, latent_dim, n_views)
#             - logvars: torch.Tensor, the log-variance tensor of shape (batch_size, latent_dim, n_views)
#             - weights: torch.Tensor, the attention weights of shape (batch_size, 1, n_views)
#         Returns:
#             - fused_mean: torch.Tensor, the fused mean tensor of shape (batch_size, latent_dim)
#             - fused_logvar: torch.Tensor, the fused log-variance tensor of shape (batch_size, latent_dim)
#         """
#         # Weighted mean fusion
#         fused_mean = torch.sum(weights * means, dim=-1)

#         # Weighted log-sum-exp fusion for logvars (https://raw.org/math/the-log-sum-exp-trick-in-machine-learning/)
#         max_logvar = torch.max(logvars, dim=-1, keepdim=True)[0] # tuple output (discard index)
#         weighted_var = weights * torch.exp(logvars - max_logvar)
#         sum_weighted_var = torch.sum(weighted_var, dim=-1, keepdim=True)
#         fused_logvar = max_logvar + torch.log(sum_weighted_var + eps)

#         return fused_mean, fused_logvar.squeeze(dim=-1)

#     def _mixture_fusion(self, means, logvars, weights, eps = 1e-6):
#         """
#         Fusion of dependent Gaussian distributions using a mixture model approach.
        
#         Args:
#             - means: torch.Tensor, the means tensor of shape (batch_size, latent_dim, n_views)
#             - logvars: torch.Tensor, the log-variance tensor of shape (batch_size, latent_dim, n_views)
#             - weights: torch.Tensor, the attention weights of shape (batch_size, 1, n_views)
#         Returns:
#             - fused_mean: torch.Tensor, the fused mean tensor of shape (batch_size, latent_dim)
#             - fused_logvar: torch.Tensor, the fused log-variance tensor of shape (batch_size, latent_dim)
#         """
#         # Weighted mean fusion
#         fused_mean = torch.sum(weights * means, dim=-1, keepdim=True) # (batch_size, latent_dim, 1)

#         # Weighted mixture variance with cross-terms
#         max_logvar = torch.max(logvars, dim=-1, keepdim=True)[0] # (batch_size, latent_dim, 1)
#         _vars = torch.exp(logvars - max_logvar) # (batch_size, latent_dim, n_views)
#         fused_var = torch.sum(weights * (_vars + means**2), dim=-1, keepdim=True) - fused_mean**2
#         fused_logvar = max_logvar + torch.log(fused_var + eps)

#         return fused_mean.squeeze(dim=-1), fused_logvar.squeeze(dim=-1)

#     def _reparameterize(self, mean, logvar):
#         """
#         Args:
#             - mean: torch.Tensor, the mean tensor of shape (batch_size, latent_dim)
#             - std: torch.Tensor, the standard-deviations of shape (batch_size, latent_dim)
#         Returns:
#             - z: torch.Tensor, the output tensor of shape (batch_size, latent_dim) after reparameterization
#         """
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std) # sample epsilon from N(0, 1)
#         return mean + eps * std

#     def _kl_divergence(self, mean, logvar):
#         """
#         Args:
#             - mean: torch.Tensor, the mean tensor of shape (batch_size, latent_dim)
#             - logvar: torch.Tensor, the log-variance tensor of shape (batch_size, latent_dim)
#         Returns:
#             - kl_loss: torch.Tensor, the KL divergence loss regularizing the latent space into a standard normal distribution
#         """
#         return - ( 
#             0.5 * torch.sum(1 + logvar - mean**2 - logvar.exp()) 
#             ) / mean.size(1)