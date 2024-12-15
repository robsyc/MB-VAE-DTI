import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def get_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#######################################################################################################################################

class Encoder(nn.Module):
    """
    Encoder with residual connections, LayerNorm, SiLU activation, and dropout.

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
            depth: int,
            dropout_prob: float = 0.1,
            **kwargs
    ):
        super(Encoder, self).__init__()
        self.input2hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_prob),
        )
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.SiLU(),
                nn.Dropout(dropout_prob),
            ) for _ in range(depth)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(depth)])
        self.hidden2output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            - x: torch.Tensor, the input tensor of shape (batch_size, input_dim)
        Returns:
            - torch.Tensor, the output tensor of shape (batch_size, output_dim)
        """
        x = self.input2hidden(x)
        for i, layer in enumerate(self.hidden_layers):
            x = self.layer_norms[i](x + layer(x))
        return self.hidden2output(x)

class Branch(nn.Module):
    """
    Multi-view encoding branch.

    Args:
        - input_dim_list: list, the dimensions of the input tensors (num_modalities)
        - hidden_dim: int, the dimension of the hidden layer(s)
        - latent_dim: int, the dimension of the branch's output tensor
        - depth: int, the number of hidden layers
        - dropout_prob: float, the dropout probability (default: 0.1)
    """
    def __init__(
            self,
            input_dim_list: list,
            hidden_dim: int,
            latent_dim: int,
            depth: int = 0,
            dropout_prob: float = 0.1
            ):
        super(Branch, self).__init__()
        self.encoders = nn.ModuleList([
                Encoder(
                    input_dim=size,
                    hidden_dim=hidden_dim,
                    output_dim=latent_dim,
                    depth=depth,
                    dropout_prob=dropout_prob)
                for size in input_dim_list])
        
        self.to_attn_logits = nn.Conv1d(latent_dim, 1, 1)
        
    def forward(self, x_list):
        """
        Args:
            - x_list: The list of input tensors of shape (batch_size, input_dim) in num_modalities
        Returns:
            - z: torch.Tensor, the output tensor of shape (batch_size, latent_dim)
            - attn_values: torch.Tensor, the attention coefficients of shape (batch_size, len(x_list))
        """
        x = torch.stack(
            [self.encoders[i](x) for i, x in enumerate(x_list)], 
            dim=-1)                                 # B x latent_dim x num_modalities
        attn_logits = self.to_attn_logits(x)        # B x 1 x num_modalities
        attn_values = attn_logits.softmax(dim=-1)   # B x 1 x num_modalities (sums to 1)
        return (
            (x * attn_values).sum(-1),              # B x latent_dim
            attn_values.squeeze(1)                  # B x num_modalities
         ) # out, attn_values over batch


# class VariationalBranch(nn.Module):
#     """
#     Variational multi-view encoding branch.

#     Args:
#         - input_dim_list: list, the dimensions of the input tensors (num_modalities)
#         - hidden_dim: int, the dimension of the hidden layer(s)
#         - latent_dim: int, the dimension of the branch's output tensor
#         - depth: int, the number of hidden layers
#         - dropout_prob: float, the dropout probability (default: 0.1)
#     """
#     def __init__(
#             self,
#             input_dim_list: list,
#             hidden_dim: int,
#             latent_dim: int,
#             depth: int,
#             dropout_prob: float = 0.1
#             ):
#         super(VariationalBranch, self).__init__()
#         self.encoders = nn.ModuleList([
#                 Encoder(
#                     input_dim=size,
#                     hidden_dim=hidden_dim,
#                     output_dim=2 * latent_dim, # mu and logvar
#                     depth=depth,
#                     dropout_prob=dropout_prob)
#                 for size in input_dim_list])
        
#         self.to_attn_logits = nn.Conv1d(2 * latent_dim, 1, 1)

#     def reparameterize(self, mean, logvar):
#         """
#         Args:
#             - mean: torch.Tensor, the mean tensor of shape (batch_size, output_dim)
#             - logvar: torch.Tensor, the log-variance tensor of shape (batch_size, output_dim)
#         Returns:
#             - z: torch.Tensor, the output tensor of shape (batch_size, output_dim) after reparameterization
#         """
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mean + eps * std

#     def forward(self, x_list, compute_kl_loss = True):
#         """
#         Args:
#             - x_list: The list of input tensors of shape (batch_size, input_dim) in input_dim_list
#         Returns:
#             - z: torch.Tensor, the output tensor of shape (batch_size, latent_dim) after reparameterization
#             - kl_loss: torch.Tensor, the KL divergence loss
#             - attn_values: torch.Tensor, the attention coefficients of shape (batch_size, len(x_list))
#         """
#         x = torch.stack(
#             [self.encoders[i](x) for i, x in enumerate(x_list)], 
#             dim=-1)                                 # B x 2*latent_dim x num_modalities
#         attn_logits = self.to_attn_logits(x)
#         attn_values = attn_logits.softmax(dim=-1)   # B x 1 x num_modalities (sums to 1)
#         return (x * attn_values).sum(-1)               # B x 2*latent_dim
        # return torch.chunk(x, 2, dim=1)             # each B x latent_dim
        # z = self.reparameterize(mu, logvar)         # B x latent_dim

        # if not compute_kl_loss:
        #     return z, attn_values.squeeze(1), None
        
        # loss_kl = ( -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) ) / mu.size(1)
        # return z, attn_values.squeeze(1), loss_kl
        

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
            hidden_dim: int,
            latent_dim: int,
            depth: int = 0,
            dropout_prob: float = 0.1
            ):
        super(PlainMultiBranch, self).__init__()
        self.branch0 = Branch(
            input_dim_list=input_dim_list_0,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            depth=depth,
            dropout_prob=dropout_prob
        ) if len(input_dim_list_0) > 1 else Encoder(
            input_dim=input_dim_list_0[0],
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            depth=depth,
            dropout_prob=dropout_prob
        )
        self.branch1 = Branch(
            input_dim_list=input_dim_list_1,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            depth=depth,
            dropout_prob=dropout_prob
        ) if len(input_dim_list_1) > 1 else Encoder(
            input_dim=input_dim_list_1[0],
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            depth=depth,
            dropout_prob=dropout_prob
        )
        print(f"Number of parameters\n - Branch 0: {get_model_params(self.branch0):,}\n - Branch 1: {get_model_params(self.branch1):,}")
        
    def forward(self, x0, x1):
        attn_values = []
        if len(x0) == 1:
            out0 = self.branch0(x0[0])
        else:
            out0, attn_values0 = self.branch0(x0)
            attn_values.append(attn_values0)

        if len(x1) == 1:
            out1 = self.branch1(x1[0])
        else:
            out1, attn_values1 = self.branch1(x1)
            attn_values.append(attn_values1)

        return torch.sum(out0 * out1, dim=1), attn_values

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
            hidden_dim: int,
            latent_dim: int,
            depth: int,
            dropout_prob: float = 0.1
            ):
        super(VariationalMultiBranch, self).__init__()
        self.branch0 = Branch(
            input_dim_list=input_dim_list_0,
            hidden_dim=hidden_dim,
            latent_dim=2 * latent_dim,  # mu and logvar
            depth=depth,
            dropout_prob=dropout_prob
        ) if len(input_dim_list_0) > 1 else Encoder(
            input_dim=input_dim_list_0[0],
            hidden_dim=hidden_dim,
            output_dim=2 * latent_dim,
            depth=depth,
            dropout_prob=dropout_prob
        )
        self.branch1 = Branch(
            input_dim_list=input_dim_list_1,
            hidden_dim=hidden_dim,
            latent_dim=2 * latent_dim,  # mu and logvar
            depth=depth,
            dropout_prob=dropout_prob
        ) if len(input_dim_list_1) > 1 else Encoder(
            input_dim=input_dim_list_1[0],
            hidden_dim=hidden_dim,
            output_dim=2 * latent_dim,
            depth=depth,
            dropout_prob=dropout_prob
        )
        print(f"Number of parameters\n - Branch 0: {get_model_params(self.branch0):,}\n - Branch 1: {get_model_params(self.branch1):,}")
        
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
    
    def forward(self, x0, x1, compute_kl_loss = True):
        attn_values = []
        if len(x0) == 1:
            out0 = self.branch0(x0[0])
        else:
            out0, attn_values0 = self.branch0(x0)
            attn_values.append(attn_values0)

        mu, logvar = torch.chunk(out0, 2, dim=1)
        z0 = self.reparameterize(mu, logvar)
        kl0 = ( -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) ) / mu.size(1) if compute_kl_loss else None

        if len(x1) == 1:
            out1 = self.branch1(x1[0])
        else:
            out1, attn_values1 = self.branch1(x1)
            attn_values.append(attn_values1)
        
        mu, logvar = torch.chunk(out1, 2, dim=1)
        z1 = self.reparameterize(mu, logvar)
        kl1 = ( -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) ) / mu.size(1) if compute_kl_loss else None
        
        kl_loss = (kl0 + kl1) / 2 if compute_kl_loss else None
        return torch.sum(z0 * z1, dim=1), attn_values, kl_loss