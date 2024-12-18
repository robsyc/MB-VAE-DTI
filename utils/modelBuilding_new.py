import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



class Generator(nn.Module):
    """
    Generator model for graph generation.

    Args:
        - latent_dim: int, the dimensionality of the input tensor
        - hidden_dim: int, the dimensionality of the hidden layer(s)
        - depth: int, the number of hidden layers & complexity of the model (residual connections)
        - dropout_prob: float, the dropout probability
        - n_nodes: int, the number of nodes in the graph
        - n_iters: int, the number of iterations for the Gumbel-Softmax sampling
    """
    def __init__(self, latent_dim, hidden_dim, depth, dropout_prob, n_nodes, n_iters):
        super(Generator, self).__init__()
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
        print(x.shape)
        for _ in range(self.n_iters):
            for residual in self.residual_blocks:
                x = x + residual(x)  # Residual connection            
            gb = self._gumbel_softmax(self.hidden2topology(x))  # Sample topology
            x = x + self.gumbel2hidden(gb)  # Integrate topology feedback
        print(x.shape)
        return self._generate_adj(self.hidden2topology(x))

    def _gumbel_softmax(self, x):
        x = torch.reshape(x, (x.size(0), -1, 2)) # 2 halves of symmetric adjacency matrix
        return F.gumbel_softmax(x, tau=1, hard=True)[:,:,0] # mash both halves together & sample binary values

    def _generate_adj(self, x):
        """Convert sampled topology to adjacency matrix."""
        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        print(adj.shape)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:, idx[0], idx[1]] = self._gumbel_softmax(x)  # Fill upper triangular part
        adj = adj + adj.transpose(1, 2)  # Symmetrize
        return adj # (batch_size, n_nodes, n_nodes)





class ResidualBranch(nn.Module):
    """
    MLP with residual connections, LayerNorm, SiLU/ELU activation, and dropout.

    Args:
        - input_dim: int, the dimension of the input tensor
        - hidden_dim: int, the dimension of the hidden layer(s)
        - output_dim: int, the dimension of the output tensor
        - depth: int, the number of hidden residual layers
        - dropout_prob: float, the dropout probability (default: 0.1)
        - activation: str, the activation function (default: 'SiLU')
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            depth: int,
            dropout_prob: float = 0.1,
            activation: str = 'SiLU',
    ):
        super(ResidualBranch, self).__init__()
        self.activation = nn.SiLU() if activation == 'SiLU' else nn.ELU()
        self.input2hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.activation,
            nn.Dropout(dropout_prob),
        )
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                self.activation,
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
            ) for _ in range(depth)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(depth)])
        self.hidden2output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        Args:
            - x: torch.Tensor, the input tensor of shape (batch_size, input_dim)
        Returns:
            - x: torch.Tensor, the output tensor of shape (batch_size, output_dim)
        """
        x = self.input2hidden(x)
        for i, layer in enumerate(self.hidden_layers):
            x = self.layer_norms[i](x + layer(x))
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
        - variational: bool, whether to use the variational variant (default: False)
    """
    def __init__(
            self,
            drug_dims: list,
            target_dims: list,
            hidden_dim: int,
            latent_dim: int,
            depth: int = 0,
            dropout_prob: float = 0.1,
            variational: bool = False,
            classification: bool = False,
    ):
        super(DrugTargetTree, self).__init__()
        self.variational = variational
        self.classification = classification

        # Initialize drug and target leafs
        self.drug_leafs = nn.ModuleList([
            ResidualBranch(drug_dim, hidden_dim, hidden_dim, depth, dropout_prob)
            for drug_dim in drug_dims
        ])
        self.target_leafs = nn.ModuleList([
            ResidualBranch(target_dim, hidden_dim, hidden_dim, depth, dropout_prob)
            for target_dim in target_dims
        ])
        
        # Initialize branches
        latent_dim_out = latent_dim * 2 if variational else latent_dim
        self.drug_branch = ResidualBranch(hidden_dim * len(drug_dims), hidden_dim, latent_dim_out, depth, dropout_prob)
        self.target_branch = ResidualBranch(hidden_dim * len(target_dims), hidden_dim, latent_dim_out, depth, dropout_prob)
        self.stem = ResidualBranch(latent_dim * 2, hidden_dim, latent_dim_out, depth, dropout_prob)

        # Root and output layers
        self.root = ResidualBranch(latent_dim, hidden_dim, latent_dim * 2, depth, dropout_prob)
        self.bud = nn.Linear(latent_dim, 1)

        p_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of parameters\n - Drug Leafs: {get_model_params(self.drug_leafs):,}\n - Target Leafs: {get_model_params(self.target_leafs):,}\n - Drug Branch: {get_model_params(self.drug_branch):,}\n - Target Branch: {get_model_params(self.target_branch):,}\n - Stem: {get_model_params(self.stem):,}\n - Root: {get_model_params(self.root):,}\n - Bud: {get_model_params(self.bud):,}\n - Total Trainable: {p_trainable:,}")

    def forward(self, drug_inputs, target_inputs, compute_recon_loss = True, compute_kl_loss = False):
        """
        Args:
            - drug_inputs: list of torch.Tensor, the input tensor(s) of the drug-branch
            - target_inputs: list of torch.Tensor, the input tensor(s) of the target-branch
            - compute_kl_loss: bool, whether to compute the KL divergence loss (default: False)
        Returns:
            - y_hat: torch.Tensor, the output tensor of shape (batch_size, 1)
            - recon_loss: torch.Tensor, the latent reconstruction loss
            - kl_loss: torch.Tensor, the KL divergence loss
        """
        drug_latents = torch.cat([leaf(drug_input) for leaf, drug_input in zip(self.drug_leafs, drug_inputs)], dim=1)
        target_latents = torch.cat([leaf(target_input) for leaf, target_input in zip(self.target_leafs, target_inputs)], dim=1)
        
        if self.variational:
            # Variational branches
            drug_mean, drug_logvar = torch.chunk(self.drug_branch(drug_latents), 2, dim=1)
            target_mean, target_logvar = torch.chunk(self.target_branch(target_latents), 2, dim=1)
            drug_latents = self._reparameterize(drug_mean, drug_logvar)
            target_latents = self._reparameterize(target_mean, target_logvar)

            stem_mean, stem_logvar = torch.chunk(self.stem(torch.cat([drug_latents, target_latents], dim=1)), 2, dim=1)
            stem_latents = self._reparameterize(stem_mean, stem_logvar)
        else:
            # Deterministic branches
            drug_latents = self.drug_branch(drug_latents)
            target_latents = self.target_branch(target_latents)
            stem_latents = self.stem(torch.cat([drug_latents, target_latents], dim=1))

        # Compute interaction output
        y_hat = self.bud(stem_latents).squeeze()

        if self.classification:
            y_hat = torch.sigmoid(y_hat)
        
        if not compute_recon_loss:
            return y_hat
        
        # Reconstruct drug and target latents using the root and reconstruction loss
        reconstructed_latents = self.root(stem_latents)
        reconstructed_drug_latents, reconstructed_target_latents = torch.chunk(reconstructed_latents, 2, dim=1)
        recon_loss = (
            self._reconstruction_loss(drug_latents, reconstructed_drug_latents) +
            self._reconstruction_loss(target_latents, reconstructed_target_latents)
        ) / 2

        if not compute_kl_loss:
            return y_hat, recon_loss
        
        # Compute KL divergence loss
        kl_loss = (
            self._kl_divergence(drug_mean, drug_logvar) +
            self._kl_divergence(target_mean, target_logvar) +
            self._kl_divergence(stem_mean, stem_logvar)
        ) / 3

        return y_hat, recon_loss, kl_loss

    def _reparameterize(self, mean, logvar):
        """
        Args:
            - mean: torch.Tensor, the mean tensor of shape (batch_size, latent_dim)
            - logvar: torch.Tensor, the log-variance tensor of shape (batch_size, latent_dim)
        Returns:
            - z: torch.Tensor, the output tensor of shape (batch_size, latent_dim) after reparameterization
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def _kl_divergence(self, mean, logvar):
        """
        Args:
            - mean: torch.Tensor, the mean tensor of shape (batch_size, latent_dim)
            - logvar: torch.Tensor, the log-variance tensor of shape (batch_size, latent_dim)
        Returns:
            - kl_loss: torch.Tensor, the KL divergence loss regularizing the latent space into a standard normal distribution
        """
        return ( -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) ) / mean.size(1)
    
    def _reconstruction_loss(self, x, x_reconstructed):
        """
        Args:
            - x: torch.Tensor, the input tensor of shape (batch_size, *)
            - x_reconstructed: torch.Tensor, the reconstructed tensor of shape (batch_size, *)
        Returns:
            - loss: torch.Tensor, the reconstruction loss
        """
        return F.mse_loss(x_reconstructed, x, reduction='mean')