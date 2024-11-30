import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def get_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#######################################################################################################################################

# BLOCKS

class PlainEncoder(nn.Module):
    """
    Plain encoder consisting of fully connected layers with layer normalization, SiLU activation, and dropout.

    Args:
        - input_dim: int, the dimension of the input tensor
        - hidden_dim: int, the dimension of the hidden layer(s)
        - output_dim: int, the dimension of the output tensor
        - depth: int, the number of hidden layers
            - Default: 0 in which case there are 2 linear layers
            - Example: 1 in which case there are 3 linear layers
        - dropout_prob: float, the dropout probability (default: 0.1)
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            depth: int = 0,
            dropout_prob: float = 0.1,
            **kwargs
    ):
        super(PlainEncoder, self).__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_prob),
        ]
        for _ in range(depth):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_prob),
            ])
        layers.extend([
            nn.Linear(hidden_dim, output_dim),
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            - x: torch.Tensor, the input tensor of shape (batch_size, input_dim)
        Returns:
            - x: torch.Tensor, the output tensor of shape (batch_size, output_dim)
        """
        return self.block(x)

class VariationalEncoder(nn.Module):
    """
    Variational encoder consisting of fully connected layers with layer normalization, SiLU activation, and dropout.
    Final layer projects to 2 x output_dim (mean and logvar) and applies reparameterization. 

    Args:
        - input_dim: int, the dimension of the input tensor
        - hidden_dim: int, the dimension of the hidden layer(s)
        - output_dim: int, the dimension of the output tensor
        - depth: int, the number of hidden layers
            - Default: 0 in which case there are 2 linear layers
            - Example: 1 in which case there are 3 linear layers
        - dropout_prob: float, the dropout probability (default: 0.1)
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            depth: int = 0,
            dropout_prob: float = 0.1,
            **kwargs
    ):
        super(VariationalEncoder, self).__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_prob),
        ]
        for _ in range(depth):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_prob),
            ])
        layers.extend([
            nn.Linear(hidden_dim, 2 * output_dim), # mean and logvar
        ])
        self.block = nn.Sequential(*layers)
        self.softplus = nn.Softplus()
    
    def encode(self, x, eps: float = 1e-6):
        """
        Encodes the input data into the latent space.
        
        Args:
            x (torch.Tensor): Input data.
            eps (float): Small value to avoid numerical instability. (sometimes nan's still present though... gives error)
        
        Returns:
            torch.distributions.MultivariateNormal: Normal distribution of the encoded data.
        """
        x = self.block(x)
        mean, logvar = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)
        # print(mean.shape, logvar.shape, scale.shape, scale_tril.shape)
        return torch.distributions.MultivariateNormal(mean, scale_tril=scale_tril) # TODO fix error due to nan?!?

    def reparameterize(self, dist):
        """
        Reparameterizes the encoded data to sample from the latent space.
        
        Args:
            dist (torch.distributions.MultivariateNormal): Normal distribution of the encoded data.
        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        return dist.rsample()

    def forward(self, x):
        """
        Args:
            - x: torch.Tensor, the input tensor of shape (batch_size, input_dim)
        Returns:
            - z: torch.Tensor, the output tensor of shape (batch_size, output_dim) after reparameterization
            - loss_kl: the KL divergence loss
        """
        dist = self.encode(x)
        z = self.reparameterize(dist)
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
        )
        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()
        return z, loss_kl

class AttentiveAggregator(nn.Module):
    """
    Attention-based embedding aggregator. Embeddings are first projected to same space, multiplied by a learned attention ~ 'view' and summed.

    Args:
        - input_dim_list: list[int], the dimensions of the input vectors
        - output_dim: int, size of the aggregated output dimension (default: min(input_dim_list))
    """
    def __init__(
        self,
        input_dim_list: list,
        output_dim: int = None,
        ):
        super().__init__()
        if output_dim is None:
            output_dim = min(input_dim_list)
        self.project_to_samespace = nn.ModuleList([
            nn.Linear(input_dim, output_dim)
            for input_dim in input_dim_list
        ])
        self.to_attn_logits = nn.Conv1d(output_dim, 1, 1)

    def forward(self, x_list):
        x_proj = []
        for x, layer in zip(x_list, self.project_to_samespace):
            x_proj.append(layer(x))
        x_proj = torch.stack(x_proj, -1)            # B, output_dim, len(x_list)
        attn_logits = self.to_attn_logits(x_proj)   # B, 1, len(x_list)
        attn_values = attn_logits.softmax(dim = -1) # B, 1, len(x_list)
        return (
            (x_proj * attn_values).sum(-1),         # B, output_dim
            attn_values.squeeze(1)                  # B, len(x_list)
        )

#######################################################################################################################################

# BRANCHES

class PlainBranch(nn.Module):
    """
    Plain single- or multi-view encoding branch.

    Args:
        - input_dim_list: list, the dimensions of the input tensors
        - hidden_dim: int, the dimension of the hidden layer(s)
        - latent_dim: int, the dimension of the branch's output tensor
        - depth: int, the number of hidden layers
        - dropout_prob: float, the dropout probability (default: 0.1)
    """
    def __init__(
            self,
            input_dim_list: list,
            hidden_dim: int = 512,
            latent_dim: int = 1024,
            depth: int = 0,
            dropout_prob: float = 0.1
            ):
        super(PlainBranch, self).__init__()
        if len(input_dim_list) > 1:
            self.blocks = nn.ModuleList([
                    PlainEncoder(
                        input_dim=size,
                        hidden_dim=hidden_dim,
                        output_dim=hidden_dim,
                        depth=depth,
                        dropout_prob=dropout_prob)
                    for size in input_dim_list])
            self.aggregator = AttentiveAggregator(
                input_dim_list=len(input_dim_list)*[hidden_dim],
                output_dim=latent_dim
            )
        else:
            self.block = PlainEncoder(
                input_dim=input_dim_list[0],
                hidden_dim=hidden_dim,
                output_dim=latent_dim,
                depth=depth,
                dropout_prob=dropout_prob
                )
        
    def forward(self, x_list):
        """
        Args:
            - x_list: The list of input tensors of shape (batch_size, input_dim) in input_dim_list
        Returns:
            - out: torch.Tensor, the output tensor of shape (batch_size, latent_dim)
        """
        if len(x_list) == 1:
            return self.block(x_list[0])
        x_list = [self.blocks[i](x) for i, x in enumerate(x_list)]
        return self.aggregator(x_list) # B x latent_dim & coefficients

class VariationalBranch(nn.Module):
    """
    Variational single- or multi-view encoding branch.

    Args:
        - input_dim_list: list, the dimensions of the input tensors
        - hidden_dim: int, the dimension of the hidden layer(s)
        - latent_dim: int, the dimension of the branch's output tensor
        - depth: int, the number of hidden layers
        - dropout_prob: float, the dropout probability (default: 0.1)
    """
    def __init__(
            self,
            input_dim_list: list,
            hidden_dim: int = 512,
            latent_dim: int = 1024,
            depth: int = 0,
            dropout_prob: float = 0.1
            ):
        super(VariationalBranch, self).__init__()
        if len(input_dim_list) > 1:
            self.blocks = nn.ModuleList([
                    VariationalEncoder(
                        input_dim=size,
                        hidden_dim=hidden_dim,
                        output_dim=hidden_dim,
                        depth=depth,
                        dropout_prob=dropout_prob)
                    for size in input_dim_list])
            self.aggregator = AttentiveAggregator(
                input_dim_list=len(input_dim_list)*[hidden_dim],
                output_dim=latent_dim
            )
        else:
            self.block = VariationalEncoder(
                input_dim=input_dim_list[0],
                hidden_dim=hidden_dim,
                output_dim=latent_dim,
                depth=depth,
                dropout_prob=dropout_prob
                )
        
    def forward(self, x_list):
        """
        Args:
            - x_list: The list of input tensors of shape (batch_size, input_dim) in input_dim_list
        Returns:
            - z: torch.Tensor, the output tensor of shape (batch_size, output_dim) after reparameterization
            - loss_kl: the KL divergence loss
        """
        if len(x_list) == 1:
            return self.block(x_list[0])
        
        loss_kl = 0.0
        z_list = []
        for i, x in enumerate(x_list):
            z, kl = self.blocks[i](x)
            loss_kl += kl
            z_list.append(z)
        out, coeffs = self.aggregator(z_list) # B x latent_dim
        return out, loss_kl, coeffs

#######################################################################################################################################

# MODELS

class PlainMultiBranch(nn.Module):
    """
    Plain multi-branch encoding model for interaction-prediction.

    Args:
        - input_dim_list_0: list, the dimensions of the input tensors of branch 0
        - input_dim_list_1: list, the dimensions of the input tensors of branch 1
        - hidden_dim: int, the dimension of the hidden layer(s)
        - latent_dim: int, the dimension of the branches's output tensors
        - depth: int, the number of hidden layers
        - dropout_prob: float, the dropout probability (default: 0.1)
    """
    def __init__(
            self,
            input_dim_list_0: list,
            input_dim_list_1: list,
            hidden_dim: int = 512,
            latent_dim: int = 1024,
            depth: int = 0,
            dropout_prob: float = 0.1
            ):
        super(PlainMultiBranch, self).__init__()
        self.branch0 = PlainBranch(
            input_dim_list=input_dim_list_0,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            depth=depth,
            dropout_prob=dropout_prob
        )
        self.branch1 = PlainBranch(
            input_dim_list=input_dim_list_1,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            depth=depth,
            dropout_prob=dropout_prob
        )
        
    def forward(self, x0, x1):
        coeffs = []
        if len(x0) == 1:
            out0 = self.branch0(x0)
        else:
            out0, coeffs0 = self.branch0(x0)
            coeffs.append(coeffs0)

        if len(x1) == 1:
            out1 = self.branch1(x1)
        else:
            out1, coeffs1 = self.branch1(x1)
            coeffs.append(coeffs1)

        return torch.sum(out0 * out1, dim=1), coeffs

class VariationalMultiBranch(nn.Module):
    """
    Variational multi-branch encoding model for interaction-prediction.

    Args:
        - input_dim_list_0: list, the dimensions of the input tensors of branch 0
        - input_dim_list_1: list, the dimensions of the input tensors of branch 1
        - hidden_dim: int, the dimension of the hidden layer(s)
        - latent_dim: int, the dimension of the branches's output tensors
        - depth: int, the number of hidden layers
        - dropout_prob: float, the dropout probability (default: 0.1)
    """
    def __init__(
            self,
            input_dim_list_0: list,
            input_dim_list_1: list,
            hidden_dim: int = 512,
            latent_dim: int = 1024,
            depth: int = 0,
            dropout_prob: float = 0.1
            ):
        super(VariationalMultiBranch, self).__init__()
        self.branch0 = VariationalBranch(
            input_dim_list=input_dim_list_0,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            depth=depth,
            dropout_prob=dropout_prob
        )
        self.branch1 = VariationalBranch(
            input_dim_list=input_dim_list_1,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            depth=depth,
            dropout_prob=dropout_prob
        )
        
    def forward(self, x0, x1):
        coeffs = []
        if len(x0) == 1:
            out0, kl0 = self.branch0(x0)
        else:
            out0, kl0, coeffs0 = self.branch0(x0)
            coeffs.append(coeffs0)

        if len(x1) == 1:
            out1, kl1 = self.branch1(x1)
        else:
            out1, kl1, coeffs1 = self.branch1(x1)
            coeffs.append(coeffs1)

        return torch.sum(out0 * out1, dim=1), kl0+kl1, coeffs