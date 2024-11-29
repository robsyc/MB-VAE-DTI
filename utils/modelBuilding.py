import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5Tokenizer, T5EncoderModel
import re

from torch.nn.utils.rnn import pad_sequence
from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel
from bmfm_sm.core.data_modules.namespace import LateFusionStrategy
from bmfm_sm.predictive.data_modules.graph_finetune_dataset import Graph2dFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.image_finetune_dataset import ImageFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.text_finetune_dataset import TextFinetuneDataPipeline


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def get_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# CUSTOM MODELS
class HiddenBlock(nn.Module):
    """
    A block of fully connected layers with layer normalization, SiLU activation, and dropout.

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
        super(HiddenBlock, self).__init__()
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
        
class VariationalBlock(nn.Module):
    """
    A block of fully connected layers with layer normalization, SiLU activation, and dropout, followed by a variational layer.

    Args:
        - input_dim: int, the dimension of the input tensor
        - hidden_dim: int, the dimension of the hidden layer(s)
        - output_dim: int, the dimension of the output tensor (2x for mean and log-variance)
        - depth: int, the number of hidden layers
            - Default: 1 in which case there are 3 linear layers
            - Example: 2 in which case there are 4 linear layers
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
        super(VariationalBlock, self).__init__()
        self.encoder = HiddenBlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=2*output_dim,
            depth=depth,
            dropout_prob=dropout_prob,
        )
        self.softplus = nn.Softplus()
        self.silu = nn.SiLU()

    def encode(self, x):
        x = self.encoder(x)
        mu, log_var = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(log_var)
        scale_tril = torch.diag_embed(scale)
        return torch.distributions.MultivariateNormal(mu, scale_tril)
    
    def reparameterize(self, dist):
        return dist.rsample()
    
    def forward(self, x, compute_loss: bool = True):
        """
        Args:
            - x: torch.Tensor, the input tensor of shape (batch_size, input_dim)
            - compute_loss: bool, whether to compute the KL loss (default: True)
        Returns:
            - z: torch.Tensor, the output tensor after reparameterization of shape (batch_size, output_dim)
            - loss_kl: torch.Tensor, the KL loss (default: 0)
        """
        dist = self.encode(x)
        z = self.reparameterize(dist)
        if not compute_loss:
            return z
        loss_kl = torch.distributions.kl_divergence(dist, torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
        )).mean()
        return z, loss_kl

class AggregatorBlock(nn.Module):
    """
    A block that aggregates multiple input tensors into a single tensor using a 2-layer attention mechanism.

    Args:
        - input_dim_list: list, the list of dimensions of the input tensors
        - hidden_dim: int, the dimension of the hidden layer(s) (default: min(input_dim_list))
        - output_dim: int, the dimension of the output tensor (default: min(input_dim_list))
        - depth: int, the number of hidden layers
            - Default: 0 in which case there are 2*len(input_dim_list), 2 linear layers = 4 layers (+2 linear att layers)
            - Example: 1 in which case there are 2*len(input_dim_list), 3 linear layers = 5 layers (+2 linear att layers)
        - dropout_prob: float, the dropout probability
    """
    def __init__(
            self,
            input_dim_list: list,
            hidden_dim: int = None,
            output_dim: int = None,
            depth: int = 0,
            dropout_prob: float = 0.1,
            **kwargs
    ):
        super(AggregatorBlock, self).__init__()
        if hidden_dim is None:
            hidden_dim = min(input_dim_list)
        if output_dim is None:
            output_dim = min(input_dim_list)
        self.encoder = nn.ModuleList([
            HiddenBlock(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                depth=0,
                dropout_prob=dropout_prob,
            ) for input_dim in input_dim_list
        ])
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.decoder = HiddenBlock(
            input_dim=hidden_dim * len(input_dim_list),
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dapth=depth,
            dropout_prob=dropout_prob,
        )

    def forward(self, x_list):
        """
        Args:
            - x_list: list, the list of input tensors with shape (batch_size, input_dim) in input_dim_list
        Returns:
            - x: torch.Tensor, the output tensor of shape (batch_size, output_dim)
            - coeffs: torch.Tensor, the attention coefficients of shape len(input_dim_list)
        """
        x = [encoder(x) for encoder, x in zip(self.encoder, x_list)]
        x = torch.stack(x, dim=1)
        att = self.attention(F.normalize(x, dim=-1)).mean(dim=0)
        att = F.softmax(att, dim=0)
        att = att.expand((x.shape[0],) + att.shape)
        x = att * x
        x = torch.flatten(x, start_dim=1)
        return self.decoder(x), att.squeeze(2)[0]

class ExpanderBlock(nn.Module):
    """
    A block that expands a single input tensor into multiple output tensors using a coeff-provided attention mechanism.

    Args:
        - input_dim: int, the dimension of the input tensor
        - output_dim_list: list, the list of dimensions of the output tensors
        - hidden_dim: int, the dimension of the hidden layer(s) (default: min(output_dim_list))
        - depth: int, the number of hidden layers
            - Default: 0 in which case there are 2, 2*len(input_dim) linear layers = 4 layers
            - Example: 1 in which case there are 3, 2*len(input_dim) linear layers = 5 layers
        - dropout_prob: float, the dropout probability
    """

    def __init__(
            self,
            input_dim: int,
            output_dim_list: list,
            hidden_dim: int = None,
            depth: int = 0,
            dropout_prob: float = 0.1,
            **kwargs
    ):
        super(ExpanderBlock, self).__init__()
        if hidden_dim is None:
            hidden_dim = min(output_dim_list)
        self.encoder = HiddenBlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim * len(output_dim_list),
            depth=depth,
            dropout_prob=dropout_prob,
        )
        self.decoder = nn.ModuleList([
            HiddenBlock(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dapth=0,
                dropout_prob=dropout_prob,
            ) for output_dim in output_dim_list
        ])

    def forward(self, x, coeffs=None):
        """
        Args:
            - x: torch.Tensor, the input tensor of shape (batch_size, input_dim)
            - coeffs: torch.Tensor, the attention coefficients of shape len(output_dim_list)
        Returns:
            - x_list: list, the list of output tensors with shape (batch_size, output_dim) in output_dim_list
        """
        x = self.encoder(x)
        x = x.view(x.shape[0], len(self.decoder), -1)
        if coeffs is not None:
            att = coeffs.unsqueeze(1)
            att = att.expand((x.shape[0],) + att.shape)
            x = x * att

        x = [project(x[:, i]) for i, project in enumerate(self.decoder)]
        return x

class DrugBranch(nn.Module):
    def __init__(
            self,
            embeddings_sizes: list = [512, 512, 768],
            hidden_dim: int = 512,
            latent_dim: int = 1024,
            ):
        super(DrugBranch, self).__init__()
        # self.encoder = BiomedMultiViewMoleculeEncoder() # we use pre-compued embeddings!

        self.encodings_to_z = nn.ModuleList([
            VariationalBlock(
                input_dim=i, 
                hidden_dim=hidden_dim,
                output_dim=i, 
                depth=1) 
            for i in embeddings_sizes])
        self.aggregator = AggregatorBlock(
            input_dim_list=embeddings_sizes, 
            hidden_dim=hidden_dim, 
            output_dim=hidden_dim, 
            depth=0)
        self.aggrigate_to_z = VariationalBlock(
            input_dim=hidden_dim, 
            hidden_dim=hidden_dim, 
            output_dim=latent_dim, 
            depth=1)

        self.z_to_aggrigate = HiddenBlock(
            input_dim=latent_dim, 
            hidden_dim=hidden_dim, 
            output_dim=hidden_dim, 
            depth=1)
        self.expander = ExpanderBlock(
            input_dim=hidden_dim, 
            hidden_dim=hidden_dim, 
            output_dim_list=embeddings_sizes,
            depth=0)
        self.z_to_encodings = nn.ModuleList([
            HiddenBlock(
                input_dim=i, 
                hidden_dim=hidden_dim,
                output_dim=i, 
                depth=1) 
            for i in embeddings_sizes])

        print(f"""
Number of parameters per component
SMVV encoder       : {get_model_params(self.encoder):,}
Branch up
- Encodings to Z   : {get_model_params(self.encodings_to_z):,}
- Aggregator to Z* : {get_model_params(self.aggregator):,}
- Aggrigate to Z*  : {get_model_params(self.aggrigate_to_z):,}
Branch down
- Z* to Aggrigate' : {get_model_params(self.z_to_aggrigate):,}
- Expander to Z'   : {get_model_params(self.expander):,}
- Z' to Encodings' : {get_model_params(self.z_to_encodings):,}\n""")

    def forward(self, x, compute_loss: bool = True):
        """
        Args:
            - x: list, the list of input tensors of shape (batch_size, input_dim) in embeddings_sizes
            - compute_loss: bool, whether to compute the loss (default: True)
        Returns:
            - z: torch.Tensor, the output tensor of shape (batch_size, latent_dim)
            - loss_recon: torch.Tensor, the reconstruction loss (default: 0)
            - loss_kl: torch.Tensor, the KL loss (default: 0)
        """
        loss_kl = 0
        loss_recon = 0
        # Encode to list of embeddings with biomed-smmv molecular foundation model
        # SMILES -> 512, 512, 768 (graph, image, text)
        # embeddings = self.encoder(x) # we have pre-computed these embeddings!
        embeddings = x

        # 1) Encode each embedding to z w/ VAE & sum KL loss
        z_list = []
        for i, emb in enumerate(embeddings):
            z, kl = self.encodings_to_z[i](emb, compute_loss=True)
            z_list.append(z)
            loss_kl += kl
        # 2) Aggregate embedding z's into a single aggregated embedding
        x, coeffs = self.aggregator(z_list)
        # 3) Encode aggregated embedding to z w/ VAE
        z, tmp = self.aggrigate_to_z(x, compute_loss=True)
        loss_kl += tmp

        if not compute_loss:
            return z

        # 3) Decode z to aggregated embedding
        x = self.z_to_aggrigate(z)
        # 2) Expand aggregated embedding to list of embeddings z's
        z_list = self.expander(x, coeffs)
        # 1) Decode each z to embedding
        for i, item in enumerate(z_list):
            loss_recon += F.mse_loss(self.z_to_encodings[i](item), embeddings[i])
        
        return z, loss_recon, loss_kl

class TargetBranch(nn.Module):
    def __init__(
            self,
            input_dim: int = 1024,
            hidden_dim: int = 512,
            latent_dim: int = 1024,
            ):
        super(TargetBranch, self).__init__()
        # self.encoder = T5ProstTargetEncoder() # we use pre-compued embeddings!
        self.encodings_to_z = VariationalBlock(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=latent_dim, 
            depth=2)
        self.z_to_encodings = HiddenBlock(
            input_dim=latent_dim, 
            hidden_dim=hidden_dim, 
            output_dim=input_dim, 
            dpeth=2)

# Protein encoder: {get_model_params(self.encoder):,}
        print(f"""
Number of parameters per component
Branch up
- Encodings to Z: {get_model_params(self.encodings_to_z):,}
Branch down
- Z to Encodings: {get_model_params(self.z_to_encodings):,}\n""")
        
    def forward(self, x, compute_loss: bool = True):
        # Encode protein sequence to embedding
        # AA sequence -> 1024
        # x = self.encoder(x) # we have pre-computed these embeddings!

        # Encode embedding to z w/ VAE
        z, loss_kl = self.encodings_to_z(x, compute_loss=True)

        if not compute_loss:
            return z

        # Decode z to embedding
        x_reconstructed = self.z_to_encodings(z)
        
        # Reconstruct embedding
        loss_recon = F.mse_loss(x, x_reconstructed)
        return z, loss_recon, loss_kl

class MultiBranchDTI(nn.Module):
    def __init__(
            self,
            hidden_dim_drug: int = 1024,
            hidden_dim_target: int = 1024,
            latent_dim: int = 1024,
            ):
        super(MultiBranchDTI, self).__init__()
        self.drug_branch = DrugBranch(hidden_dim=hidden_dim_drug, latent_dim=latent_dim)
        self.target_branch = TargetBranch(hidden_dim=hidden_dim_target, latent_dim=latent_dim)

        print(f"""\nNumber of parameters per component:
Drug branch: {get_model_params(self.drug_branch):,}
Target branch: {get_model_params(self.target_branch):,}\n""")
        
    def forward(self, drug, protein, y):
        # Forward pass through both branches
        z_drug, loss_recon_drug, loss_kl_drug = self.drug_branch(drug, compute_loss=True)
        z_target, loss_recon_target, loss_kl_target = self.target_branch(protein, compute_loss=True)

        # Compute interaction prediction & loss
        y_hat = torch.sum(z_drug * z_target, dim=1)
        loss_mse = F.mse_loss(y_hat, y)
        loss_kl = loss_kl_drug + loss_kl_target
        loss_recon = loss_recon_drug + loss_recon_target
        return y_hat, loss_mse, loss_kl, loss_recon

