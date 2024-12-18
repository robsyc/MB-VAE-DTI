import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EncoderBlock(nn.Module):
    """
    Encoder with residual connections, LayerNorm, SiLU activation, and dropout.

    Args:
        - input_dim: int, the dimension of the input tensor
        - hidden_dim: int, the dimension of the hidden layer(s)
        - output_dim: int, the dimension of the output tensor
        - depth: int, the number of hidden residual layers
        - dropout_prob: float, the dropout probability (default: 0.1)
        - variational: bool, whether to use a variational encoder (default: False)
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            depth: int,
            dropout_prob: float = 0.1,
            variational: bool = False,
    ):
        super(EncoderBlock, self).__init__()
        self.variational = variational
        self.input2hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_prob),
        )
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.SiLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
            ) for _ in range(depth)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(depth)])
        self.hidden2output = nn.Linear(hidden_dim, output_dim) if not variational else nn.Linear(hidden_dim, 2 * output_dim)
    
    def reparameterize(self, mean, logvar):
        """
        Args:
            - mean: torch.Tensor, the mean tensor of shape (batch_size, output_dim)
            - logvar: torch.Tensor, the log-variance tensor of shape (batch_size, output_dim)
        Returns:
            - z: torch.Tensor, the output tensor of shape (batch_size, output_dim) after reparameterization
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, compute_kl_loss = False):
        """
        Args:
            - x: torch.Tensor, the input tensor of shape (batch_size, input_dim)
        Returns:
            - x: torch.Tensor, the output tensor of shape (batch_size, output_dim)
            or
            - z: torch.Tensor, the output tensor of shape (batch_size, output_dim) after reparameterization
            - kl_loss: torch.Tensor, the KL divergence loss
        """
        x = self.input2hidden(x)
        for i, layer in enumerate(self.hidden_layers):
            x = self.layer_norms[i](x + layer(x))
        x = self.hidden2output(x)

        if not self.variational:
            return x
        
        mean, logvar = torch.chunk(x, 2, dim=1)
        z = self.reparameterize(mean, logvar)

        if not compute_kl_loss:
            return z
        
        kl_loss = ( -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) ) / mean.size(1)
        return z, kl_loss


class MultiViewBlock(nn.Module):
    """
    Multi-view encoding branch. Combines multiple encoded representations through concatenation.

    Args:
        - input_dim_list: list, the dimensions of the input tensors (num_modalities x input_dim)
        - hidden_dim: int, the dimension of the hidden layer(s)
        - latent_dim: int, the dimension of the branch's output tensor
        - depth: int, the number of hidden layers
        - dropout_prob: float, the dropout probability (default: 0.1)
        - variational: bool, whether to use a variational encoder (default: False)
    """
    def __init__(
            self,
            input_dim_list: list,
            hidden_dim: int,
            latent_dim: int,
            depth: int = 0,
            dropout_prob: float = 0.1,
            variational: bool = False,
            ):
        super(MultiViewBlock, self).__init__()
        self.variational = variational
        self.encoders = nn.ModuleList([
                EncoderBlock(
                    input_dim=size,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    depth=depth,
                    dropout_prob=dropout_prob)
                for size in input_dim_list])
        
        self.aggregator = nn.Linear(
            len(input_dim_list) * hidden_dim, latent_dim) if not variational else nn.Linear(
            len(input_dim_list) * hidden_dim, 2 * latent_dim)

    def reparameterize(self, mean, logvar):
        """
        Args:
            - mean: torch.Tensor, the mean tensor of shape (batch_size, output_dim)
            - logvar: torch.Tensor, the log-variance tensor of shape (batch_size, output_dim)
        Returns:
            - z: torch.Tensor, the output tensor of shape (batch_size, output_dim) after reparameterization
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x_list, compute_kl_loss = False):
        """
        Args:
            - x_list: The list of input tensors of shape (batch_size, input_dim) in num_modalities
            - compute_kl_loss: bool, whether to compute the KL divergence loss (default: False)
        Returns:
            - x: torch.Tensor, the output tensor of shape (batch_size, output_dim)
            or
            - z: torch.Tensor, the output tensor of shape (batch_size, output_dim) after reparameterization
            - kl_loss: torch.Tensor, the KL divergence loss
        """
        x = torch.cat(
            [self.encoders[i](x) for i, x in enumerate(x_list)], 
            dim=-1)
        x = self.aggregator(x)

        if not self.variational:
            return x
        
        mean, logvar = torch.chunk(x, 2, dim=1)
        z = self.reparameterize(mean, logvar)

        if not compute_kl_loss:
            return z
        
        kl_loss = ( -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) ) / mean.size(1)
        return z, kl_loss


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

class MultiBranchDTI(nn.Module):
    """
    Multi-branch model for drug-target interaction-prediction.

    Args:
        - input_dim_list_0: lists, the dimension of the input tensor(s) of the drug-branch
        - input_dim_list_1: lists, the dimension of the input tensor(s) of the target-branch
        - hidden_dim: int, the dimensionality of the hidden layer(s)
        - latent_dim: int, the dimensionality of the branches's output tensors before dot-product aggregation
        - depth: int, the number of hidden layers & complexity of the model (residual connections)
        - dropout_prob: float, the dropout probability (default: 0.1)
        - variational: bool, whether to use a variational encoder (default: False)
    """
    def __init__(
            self,
            input_dim_list_0: list,
            input_dim_list_1: list,
            hidden_dim: int,
            latent_dim: int,
            depth: int = 0,
            dropout_prob: float = 0.1,
            variational: bool = False,
            ):
        super(MultiBranchDTI, self).__init__()
        self.variational = variational
        def initialize_branch(input_dim_list, hidden_dim, latent_dim, depth, dropout_prob, variational):
            return MultiViewBlock(
                input_dim_list=input_dim_list,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                depth=depth,
                dropout_prob=dropout_prob,
                variational=variational
            ) if len(input_dim_list) > 1 else EncoderBlock(
                input_dim=input_dim_list[0],
                hidden_dim=hidden_dim,
                output_dim=latent_dim,
                depth=depth,
                dropout_prob=dropout_prob,
                variational=variational
            )
        self.branch0 = initialize_branch(input_dim_list_0, hidden_dim, latent_dim, depth, dropout_prob, variational)
        self.branch1 = initialize_branch(input_dim_list_1, hidden_dim, latent_dim, depth, dropout_prob, variational)
        print(f"Number of parameters\n - Branch 0: {get_model_params(self.branch0):,}\n - Branch 1: {get_model_params(self.branch1):,}")
        
    def forward(self, x0, x1, compute_kl_loss = False):
        """
        Args:
            - x0: Tensor or list of tensors, the input tensor(s) of the drug-branch
            - x1: Tensor of list of tensors, the input tensor(s) of the target-branch
            - compute_kl_loss: bool, whether to compute the KL divergence loss (default: False)
        Returns:
            - y_hat: torch.Tensor, the output tensor of shape (batch_size, 1) after dot-product aggregation
            - kl_loss: torch.Tensor, the KL divergence loss
        """
        if not compute_kl_loss:
            return torch.sum(self.branch0(x0) * self.branch1(x1), dim=1)

        z0, kl0 = self.branch0(x0, compute_kl_loss=True)
        z1, kl1 = self.branch1(x1, compute_kl_loss=True)
        return torch.sum(z0 * z1, dim=1), (kl0 + kl1) / 2