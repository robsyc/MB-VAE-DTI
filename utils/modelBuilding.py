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
    Plain encoder with residual connections, LayerNorm, SiLU activation, and dropout.

    Args:
        - input_dim: int, the dimension of the input tensor
        - hidden_dim: int, the dimension of the hidden layer(s)
        - output_dim: int, the dimension of the output tensor
        - depth: int, the number of hidden residual layers
        - dropout_prob: float, the dropout probability (default: 0.1)
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            depth: int = 1,
            dropout_prob: float = 0.1,
            **kwargs
    ):
        super(PlainEncoder, self).__init__()
        self.input2hidden = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_prob),
        )
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            ) for _ in range(max(depth, 0))
        ])
        self.hidden2output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            - x: torch.Tensor, the input tensor of shape (batch_size, input_dim)
        Returns:
            - torch.Tensor, the output tensor of shape (batch_size, output_dim)
        """
        x = self.input2hidden(x)
        for layer in self.hidden_layers:
            x = x + layer(x)
        return self.hidden2output(x)

class VariationalEncoder(nn.Module):
    """
    Variational encoder with residual connections, LayerNorm, SiLU activation, and dropout.
    Final layer projects to 2 x output_dim (mean and logvar) and applies reparameterization. 

    Args:
        - input_dim: int, the dimension of the input tensor
        - hidden_dim: int, the dimension of the hidden layer(s)
        - output_dim: int, the dimension of the output tensor
        - depth: int, the number of hidden residual layers
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
        self.input2hidden = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_prob),
        )
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            ) for _ in range(max(depth, 0))
        ])
        self.hidden2output = nn.Linear(hidden_dim, 2 * output_dim) # mean and logvar

    def encode(self, x):
        """
        Encodes the input data into the latent space.
        
        Args:
            x (torch.Tensor): Input data.
        
        Returns:
            mean, logvar (torch.Tensor): Mean and logvar of the encoded data of shape (batch_size, output_dim).
        """
        x = self.input2hidden(x)
        for layer in self.hidden_layers:
            x = x + layer(x)
        return self.hidden2output(x).chunk(2, dim=-1)

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

    def forward(self, x, compute_loss=True):
        """
        Args:
            - x: torch.Tensor, the input tensor of shape (batch_size, input_dim)
        Returns:
            - z: torch.Tensor, the output tensor of shape (batch_size, output_dim) after reparameterization
            - loss_kl: the KL divergence loss
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        if not compute_loss:
            return z, None
        loss_kl = ( -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) ) / mean.size(1)
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
    Plain multi-view encoding branch.

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
        
    def forward(self, x_list):
        """
        Args:
            - x_list: The list of input tensors of shape (batch_size, input_dim) in input_dim_list
        Returns:
            - out: torch.Tensor, the output tensor of shape (batch_size, latent_dim)
        """
        x_list = [self.blocks[i](x) for i, x in enumerate(x_list)]
        return self.aggregator(x_list) # B x latent_dim & coefficients

class VariationalBranch(nn.Module):
    """
    Variational multi-view encoding branch.

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
        
    def forward(self, x_list, compute_loss = True):
        """
        Args:
            - x_list: The list of input tensors of shape (batch_size, input_dim) in input_dim_list
        Returns:
            - z: torch.Tensor, the output tensor of shape (batch_size, output_dim) after reparameterization
            - loss_kl: the KL divergence loss
        """
        loss_kl = 0.0
        z_list = []
        for i, x in enumerate(x_list):
            z, kl = self.blocks[i](x, compute_loss=compute_loss)
            loss_kl += kl if compute_loss else 0.0
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
        ) if len(input_dim_list_0) > 1 else PlainEncoder(
            input_dim=input_dim_list_0[0],
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            depth=depth,
            dropout_prob=dropout_prob
        )
        self.branch1 = PlainBranch(
            input_dim_list=input_dim_list_1,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            depth=depth,
            dropout_prob=dropout_prob
        ) if len(input_dim_list_1) > 1 else PlainEncoder(
            input_dim=input_dim_list_1[0],
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            depth=depth,
            dropout_prob=dropout_prob
        )
        print(f"Number of parameters\n - Branch 0: {get_model_params(self.branch0):,}\n - Branch 1: {get_model_params(self.branch1):,}")
        
    def forward(self, x0, x1):
        coeffs = []
        if len(x0) == 1:
            out0 = self.branch0(x0[0])
        else:
            out0, coeffs0 = self.branch0(x0)
            coeffs.append(coeffs0)

        if len(x1) == 1:
            out1 = self.branch1(x1[0])
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
        ) if len(input_dim_list_0) > 1 else VariationalEncoder(
            input_dim=input_dim_list_0[0],
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            depth=depth,
            dropout_prob=dropout_prob
        )
        self.branch1 = VariationalBranch(
            input_dim_list=input_dim_list_1,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            depth=depth,
            dropout_prob=dropout_prob
        ) if len(input_dim_list_1) > 1 else VariationalEncoder(
            input_dim=input_dim_list_1[0],
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            depth=depth,
            dropout_prob=dropout_prob
        )
        print(f"Number of parameters\n - Branch 0: {get_model_params(self.branch0):,}\n - Branch 1: {get_model_params(self.branch1):,}")
        
    def forward(self, x0, x1, compute_loss = True):
        coeffs = []
        if len(x0) == 1:
            out0, kl0 = self.branch0(x0[0], compute_loss=compute_loss)
        else:
            out0, kl0, coeffs0 = self.branch0(x0, compute_loss=compute_loss)
            coeffs.append(coeffs0)

        if len(x1) == 1:
            out1, kl1 = self.branch1(x1[0], compute_loss=compute_loss)
        else:
            out1, kl1, coeffs1 = self.branch1(x1, compute_loss=compute_loss)
            coeffs.append(coeffs1)
        
        kl_loss = kl0 + kl1 if compute_loss else None
        return torch.sum(out0 * out1, dim=1), kl_loss, coeffs