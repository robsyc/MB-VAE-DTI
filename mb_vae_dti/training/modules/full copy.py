"""
Multi-hybrid DTI PyTorch Lightning module with
- Multi-modal input (multiple drug/target features with aggregation)
- Multi-output prediction (DTI head for multiple simultaneous DTI score prediction)
- Contrastive pretraining support (InfoNCE loss for individual branch pretraining)
- Support for all training phases: pretrain_drug, pretrain_target, train, finetune
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
import rdkit.Chem as Chem
from typing import Dict, List, Optional, Any, Tuple, Literal
import logging
import wandb
from tqdm import tqdm

from .utils import AbstractDTIModel

from mb_vae_dti.training.models import (
    ResidualEncoder, TransformerEncoder,
    ConcatAggregator, AttentiveAggregator,
    CrossAttentionFusion
)
from mb_vae_dti.training.models.heads import DTIHead, InfoNCEHead, KLVariationalHead, ReconstructionHead
from mb_vae_dti.training.metrics import DTIMetricsCollection, RealDTIMetrics

from mb_vae_dti.training.diffusion.utils import *
from mb_vae_dti.training.diffusion.augmentation import Augmentation
from mb_vae_dti.training.diffusion.discrete_noise import PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
from mb_vae_dti.training.models.graph_transformer import GraphTransformer
from mb_vae_dti.training.metrics.validation_metrics import *

logger = logging.getLogger(__name__)


class FullDTIModel(AbstractDTIModel):
    """
    Full DTI model with variational drug branch and diffusion decoder.
    
    Architecture:
    - Drug branch: Multiple features → Encoders → Aggregation → VAE → z → InfoNCE/DTI/Decoder
    - Target branch: Multiple features → Encoders → Aggregation → InfoNCE/DTI  
    - Diffusion decoder: z → Graph transformer → Reconstructed molecular graph
    - DTI prediction: Drug/Target embeddings → Fusion → DTI head → Multiple DTI scores
    
    Training phases:
    - pretrain_drug: Drug branch pretraining with InfoNCE + KL + reconstruction losses
    - pretrain_target: Target branch pretraining with InfoNCE loss
    - train: General DTI training (multi-score prediction on combined dataset)
    - finetune: Fine-tuning on benchmark datasets (single-score prediction)
    """
    def __init__(
        self,

        phase: Literal["pretrain_drug", "pretrain_target", "train", "finetune"],
        finetune_score: Optional[Literal["Y_pKd", "Y_KIBA"]],

        learning_rate: float,
        weight_decay: float,
        scheduler: Optional[Literal["const", "step", "one_cycle", "cosine"]],

        weights: List[float],      # [accuracy, complexity, contrastive, reconstruction]
        dti_weights: List[float],  # [Y, Y_pKd, Y_pKi, Y_KIBA]
        diff_weights: List[int],   # [X, E]
        contrastive_temp: float,

        drug_features: Dict[
            Literal["FP-Morgan", "EMB-BioMedGraph", "EMB-BioMedImg", "EMB-BioMedText"], 
            int],  # dict mapping feature name to input dimension
        target_features: Dict[
            Literal["FP-ESM", "FP-NT", "FP-ESP"], 
            int
        ],

        embedding_dim: int,
        hidden_dim: int,
        factor: int,
        n_layers: int,
        activation: nn.Module,
        dropout: float,
        bias: bool,

        encoder_type: Literal["resnet", "transformer"],
        aggregator_type: Literal["concat", "attentive"],
        
        # Diffusion decoder parameters
        diffusion_steps: int,
        num_samples_to_generate: int,
        graph_transformer_kwargs: Optional[Dict],
        dataset_infos: Dict[str, Any]
        # Shape of dataset_infos:
        # {
        #     "general": {
        #         "atom_types": list,
        #         "bond_types": list,
        #         "num_atom_types": int,
        #         "max_n_nodes": int,
        #         "atom_valencies": list,
        #         "atom_weights": list,
        #         "max_weight": int
        #     },
        #     "dataset": {
        #         "dataset_name": str,
        #         "max_nodes": int,
        #         "node_count_distribution": dict,
        #         "node_marginals": list,
        #         "edge_marginals": list,
        #         "valency_distribution": list,
        #     }
        # }
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_dtype = torch.float32
        self.phase = phase
        self.finetune_score = finetune_score
        self.attentive = aggregator_type == "attentive"

        self.weights = weights
        self.dti_weights = dti_weights
        self.diff_weights = diff_weights
        self.contrastive_temp = contrastive_temp
        
        # Create encoders for processing features
        encoder_type = ResidualEncoder if encoder_type == "resnet" else TransformerEncoder
        self.drug_encoders = nn.ModuleDict()
        for feat_name in drug_features:
            self.drug_encoders[feat_name] = encoder_type(
                input_dim=drug_features[feat_name],
                hidden_dim=hidden_dim,
                output_dim=hidden_dim if len(drug_features) > 1 else embedding_dim,
                n_layers=n_layers,
                factor=factor,
                dropout=dropout,
                bias=bias,
                activation=activation
            )
        self.target_encoders = nn.ModuleDict()
        for feat_name in target_features:
            self.target_encoders[feat_name] = encoder_type(
                input_dim=target_features[feat_name],
                hidden_dim=hidden_dim,
                output_dim=hidden_dim if len(target_features) > 1 else embedding_dim,
                n_layers=n_layers,
                factor=factor,
                dropout=dropout,
                bias=bias,
                activation=activation
            )
        
        # Intra-branch aggregators for merging features
        aggregator_type = AttentiveAggregator if self.attentive else ConcatAggregator
        self.drug_aggregator = aggregator_type(
                input_dim=hidden_dim,
                output_dim=embedding_dim,
                n_features=len(drug_features),
                dropout=dropout,
                activation=activation
            ) if len(drug_features) > 1 else nn.Identity()
        self.target_aggregator = aggregator_type(
                input_dim=hidden_dim,
                output_dim=embedding_dim,
                n_features=len(target_features),
            ) if len(target_features) > 1 else nn.Identity()
        
        # Heads & fusion for DTI prediction
        self.drug_kl_head = KLVariationalHead(
            input_dim=embedding_dim,
            proj_dim=embedding_dim,
        )
        self.drug_infonce_head = InfoNCEHead(
            input_dim=embedding_dim,
            proj_dim=hidden_dim / 2,
            dropout=dropout,
            bias=bias,
            activation=activation,
            contrastive_temp=contrastive_temp
        )
        self.target_infonce_head = InfoNCEHead(
            input_dim=embedding_dim,
            proj_dim=hidden_dim / 2,
            dropout=dropout,
            bias=bias,
            activation=activation,
            contrastive_temp=contrastive_temp
        )
        self.fusion = CrossAttentionFusion(
            input_dim=embedding_dim,
            output_dim=hidden_dim,
            n_layers=n_layers,
            factor=factor,
            dropout=dropout,
            bias=bias,
            activation=activation
        )
        self.dti_head = DTIHead(
            input_dim=hidden_dim,
            proj_dim=hidden_dim / 2,
            dropout=dropout,
            bias=bias,
            activation=activation,
            dti_weights=dti_weights
        )
        self.drug_reconstruction_head = ReconstructionHead(diff_weights=diff_weights)

        # Diffusion decoder
        self.T = diffusion_steps
        self.visualization_tools = MolecularVisualization(dataset_infos["general"]["atom_types"])
        self.graph_converter = SmilesToPyG() # default atom and bond encodings used here bcs lazy
        self.nodes_dist = DistributionNodes(dataset_infos["dataset"]["node_count_distribution"])
        self.limit_dist = PlaceHolder(
            X=torch.tensor(dataset_infos["dataset"]["node_marginals"]).to(self.device), 
            E=torch.tensor(dataset_infos["dataset"]["edge_marginals"]).to(self.device), 
            y=torch.ones(graph_transformer_kwargs["output_dims"]["y"]) / graph_transformer_kwargs["output_dims"]["y"]
        )
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(timesteps=diffusion_steps)
        self.transition_model = MarginalUniformTransition( # transition model for applying noise to clean graphs
            x_marginals=torch.tensor(dataset_infos["dataset"]["node_marginals"]).to(self.device),
            e_marginals=torch.tensor(dataset_infos["dataset"]["edge_marginals"]).to(self.device),
            y_classes=graph_transformer_kwargs["output_dims"]["y"]
        )
        self.augmentation = Augmentation( # adds features to discretized noisy graphs
            valencies=dataset_infos["general"]["atom_valencies"],
            max_weight=dataset_infos["general"]["max_weight"],
            atom_weights=dataset_infos["general"]["atom_weights"],
            max_n_nodes=dataset_infos["general"]["max_n_nodes"]
        )
        self.Xdim_output = graph_transformer_kwargs['output_dims']['X']
        self.Edim_output = graph_transformer_kwargs['output_dims']['E']
        self.ydim_output = graph_transformer_kwargs['output_dims']['y']

        assert self.ydim_output == embedding_dim

        self.drug_decoder = GraphTransformer(
            n_layers=graph_transformer_kwargs['n_layers'],
            input_dims=graph_transformer_kwargs['input_dims'],
            output_dims=graph_transformer_kwargs['output_dims'],
            hidden_mlp_dims=graph_transformer_kwargs['hidden_mlp_dims'],
            hidden_dims=graph_transformer_kwargs['hidden_dims'],
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU()
        )
        
        # Metrics based on phase
        if phase in ["pretrain_drug", "pretrain_target"]:
            # No specific metrics for pretraining - just log contrastive loss
            self.train_metrics_dti = None
            self.val_metrics_dti = None
            self.test_metrics_dti = None
            # TODO: need to add pretraining metrics?!
            # TODO: need to add reconstruction metrics!!
        elif phase == "train":
            # Multi-score metrics for general training
            self.train_metrics_dti = DTIMetricsCollection(
                include_binary=True,
                include_real=True,
                real_score_names=["pKd", "pKi", "KIBA"],
                prefix="train/"
            )
            self.val_metrics_dti = DTIMetricsCollection(
                include_binary=True,
                include_real=True,
                real_score_names=["pKd", "pKi", "KIBA"],
                prefix="val/"
            )
            self.test_metrics_dti = DTIMetricsCollection(
                include_binary=True,
                include_real=True,
                real_score_names=["pKd", "pKi", "KIBA"],
                prefix="test/"
            )
        else:  # finetune
            # Single-score metrics for fine-tuning
            if self.finetune_score is None:
                raise ValueError("finetune_score must be specified for finetune phase")

            self.train_metrics_dti = RealDTIMetrics(prefix="train/")
            self.val_metrics_dti = RealDTIMetrics(prefix="val/")
            self.test_metrics_dti = RealDTIMetrics(prefix="test/")
    
    def apply_noise(self, G, node_mask):
        """
        Apply noise to the drug graph
        Args:
            G: Drug graph (X, E)
            node_mask: Mask for nodes
        Returns:
            G_t: Noisy drug graph (X_t, E_t)
            noise_params: Noise parameters sampled from the noise schedule
        """
        # Sample timestep t (uniformly from [0, T])
        t_int = torch.randint(0, self.T + 1, size=(G["X"].size(0), 1), device=G["X"].device).float()
        s_int = t_int - 1

        # Normalize timesteps to [0, 1] for noise scheduler
        t_float = t_int / self.T
        s_float = s_int / self.T

        # Get noise schedule parameters
        beta_t = self.noise_schedule(t_normalized=t_float)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)

        # Get transition matrices Q_t_bar for forward diffusion
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Forward diffusion: Compute transition probabilities
        probX = G["X"] @ Qtb.X               # (bs, n, dx_out)    - node transition probabilities p(X_t | X_0)
        probE = G["E"] @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out) - edge transition probabilities p(E_t | E_0)

        # Sample discrete features from the transition probabilities
        sampled_t = sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)
        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (G["X"].shape == X_t.shape) and (G["E"].shape == E_t.shape)
        
        G_t = PlaceHolder(X=X_t, E=E_t).type_as(X_t).mask(node_mask)

        noise_params = {
            "t_int": t_int,
            "t": t_float,
            "beta_t": beta_t,
            "alpha_s_bar": alpha_s_bar,
            "alpha_t_bar": alpha_t_bar,
        }
        return G_t, noise_params

    def augment_graph(self, G_t, noise_params, node_mask):
        """
        At every training step (after adding noise & sampling discrete features)
        extra features are computed (graph-structural features & molecular features)
        and appended to the graph transformer input.
        + timestep info t is added to graph-level features y to inform the model about the noise level
        # X gets 8 extra features, E none and y gets 13
        """
        extra_feats = self.augmentation(G_t, node_mask)
        t = noise_params['t'] # normalized timestep

        G_t.X = torch.cat((G_t.X, extra_feats.X), dim=2)
        G_t.E = torch.cat((G_t.E, extra_feats.E), dim=3)
        G_t.y = torch.cat((extra_feats.y, t), dim=-1)
        
        return G_t
    
    def forward(
        self, 
        drug_features: List[torch.Tensor], 
        target_features: List[torch.Tensor],
        G_t: PlaceHolder,
        node_mask: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass through the multi-hybrid model.
        
        Args:
            drug_features: List of drug feature tensors
            target_features: List of target feature tensors
            G_t: Discretized & augmented noisy graph (X, E, y) (y still lacks global drug embedding)
            node_mask: Mask for nodes
        Returns two dicts:
            - predictions: dti and G_hat predictions
            - outputs: intermediate embeddings & attention weights
        """
        # Encode drug features
        drug_result = self._encode_drug_features(drug_features)
        if self.attentive:
            drug_embedding, drug_attention = drug_result
        else:
            drug_embedding, drug_attention = drug_result, None
        
        # Encode target features
        target_result = self._encode_target_features(target_features)
        if self.attentive:
            target_embedding, target_attention = target_result
        else:
            target_embedding, target_attention = target_result, None

        # Variational encoding
        drug_embedding, mu, logvar = self.drug_kl_head(drug_embedding)

        # Fusion of both embeddings
        fused_emb = self.fusion(drug_embedding, target_embedding)

        # DTI predictions
        dti_preds = self.dti_head(fused_emb)

        # Graph predictions
        G_t.y = torch.cat((drug_embedding, G_t.y), dim=1)
        G_hat = self.drug_decoder(G_t.X, G_t.E, G_t.y, node_mask)

        predictions = {
            "dti_preds": dti_preds,
            "G_hat": G_hat
        }

        outputs = {
            "drug_embedding": drug_embedding,
            "target_embedding": target_embedding,
            "drug_mu": mu,
            "drug_logvar": logvar,
            "drug_att": drug_attention,
            "target_att": target_attention,
            "fused_emb": fused_emb,
        }

        return predictions, outputs
    
    def _common_step(
        self, 
        batch: Dict[str, Any]
    ):
        """
        Common step logic shared across train/val/test.
        
        Args:
            batch Dict[str, Any]: Batch data
            
        Returns three dicts:
            - predictions: with keys dti_preds, G_hat
            - targets: with keys dti_targets, G
            - losses: with keys contrastive, complexity, reconstruction, accuracy
            # TODO: possibly need to add more diffusion stuff like G_t, ...
        """
        # Get features and targets
        drug_features, target_features = self._get_features_from_batch(batch)
        smiles = self._get_smiles_from_batch(batch)
        drug_fp, target_fp = self._get_fingerprints_from_batch(batch)
        dti_targets = self._get_targets_from_batch(batch)
        dti_masks = self._get_targets_masks_from_batch(batch)

        # Get graph data
        G = self.graph_converter.smiles_to_pyg_batch(smiles)
        G, node_mask = to_dense(G.x, G.edge_index, G.edge_attr, G.batch)
        G = G.mask(node_mask)

        # Forward diffusion
        G_t, noise_params = self.apply_noise(G, node_mask)
        G_t = self.augment_graph(G_t, noise_params, node_mask)

        # Forward pass
        predictions, outputs = self(drug_features, target_features, G_t, node_mask)

        # Compute losses
        accuracy_loss = self.dti_head.loss(
            predictions=predictions["dti_preds"],
            targets=dti_targets,
            masks=dti_masks
        )
        complexity_loss = self.drug_kl_head.kl_divergence(
            outputs["drug_mu"], outputs["drug_logvar"]
        )
        contrastive_loss = self.drug_infonce_head.forward(
            outputs["drug_embedding"],
            drug_fp
        ) + self.target_infonce_head.forward(
            outputs["target_embedding"],
            target_fp
        )
        reconstruction_loss = self.drug_reconstruction_head.forward(
            pred=predictions["G_hat"],
            true=G_t
        )

        total_loss = (
            self.weights[0] * accuracy_loss +
            self.weights[1] * complexity_loss +
            self.weights[2] * contrastive_loss +
            self.weights[3] * reconstruction_loss
        )

        targets = {
            "dti_targets": dti_targets,
            "G": G
        }
        losses = {
            "total": total_loss,
            "accuracy": accuracy_loss,
            "complexity": complexity_loss,
            "contrastive": contrastive_loss,
            "reconstruction": reconstruction_loss
        }
        return predictions, targets, losses