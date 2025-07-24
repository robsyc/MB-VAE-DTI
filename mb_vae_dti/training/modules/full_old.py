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
import pytorch_lightning as pl
from torch_geometric.data import Data, Batch
import rdkit.Chem as Chem
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Literal
import logging
import wandb
from tqdm import tqdm
import time

from mb_vae_dti.training.models import (
    ResidualEncoder, TransformerEncoder,
    ConcatAggregator, AttentiveAggregator,
    CrossAttentionFusion
)
from mb_vae_dti.training.models.heads import DTIHead, InfoNCEHead, KLVariationalHead
from mb_vae_dti.training.metrics import DTIMetricsCollection, RealDTIMetrics
from .optimizer_utils import configure_optimizer_and_scheduler

from mb_vae_dti.training.diffusion.utils import *
from mb_vae_dti.training.diffusion.augmentation import Augmentation
from mb_vae_dti.training.diffusion.discrete_noise import PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
from mb_vae_dti.training.models.graph_transformer import GraphTransformer
from mb_vae_dti.training.diffusion.loss import TrainLossDiscrete
from mb_vae_dti.training.metrics.validation_metrics import *
from mb_vae_dti.training.metrics.molecular_metrics import TrainMolecularMetricsDiscrete

logger = logging.getLogger(__name__)


class FullDTIModel(pl.LightningModule):
    """
    Full DTI model with variational drug branch and diffusion decoder.
    
    Architecture:
    - Drug branch: Multiple features → Encoders → Aggregation → VAE → z → InfoNCE/DTI/Decoder
    - Target branch: Multiple features → Encoders → Aggregation → InfoNCE/DTI  
    - Diffusion decoder: z → Graph transformer → Reconstructed molecular graph
    - DTI prediction: Drug/Target embeddings → Inter-branch aggregation → DTI head → Multiple DTI scores
    
    Training phases:
    - pretrain_drug: Drug branch pretraining with InfoNCE + KL + reconstruction losses
    - pretrain_target: Target branch pretraining with InfoNCE loss
    - train: General DTI training (multi-score prediction on combined dataset)
    - finetune: Fine-tuning on benchmark datasets (single-score prediction)
    """
    def __init__(
        self,
        # Architecture parameters
        embedding_dim: int,
        drug_features: Dict[str, int],
        target_features: Dict[str, int],

        encoder_type: Literal["resnet", "transformer"],
        encoder_kwargs: Optional[Dict],

        aggregator_type: Literal["concat", "attentive"],
        aggregator_kwargs: Optional[Dict],

        fusion_kwargs: Optional[Dict],
        
        dti_head_kwargs: Optional[Dict],
        infonce_head_kwargs: Optional[Dict],
        
        # Diffusion decoder parameters # TODO: NEW
        diffusion_steps: int,
        num_samples_to_generate: int,
        graph_transformer_kwargs: Optional[Dict],
        dataset_infos: Dict,
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
        #     }
        #     "dataset": {
        #         "dataset_name": str,
        #         "max_nodes": int,
        #         "node_count_distribution": dict,
        #         "node_marginals": list,
        #         "edge_marginals": list,
        #         "valency_distribution": list,
        #     }
        # nodes_dist: DistributionNodes,

        learning_rate: float,
        weight_decay: float,
        scheduler: Optional[Literal["const", "step", "one_cycle", "cosine"]],
        
        # Phase-specific parameters
        phase: Literal["pretrain_drug", "pretrain_target", "train", "finetune"],
        finetune_score: Optional[Literal["Y_pKd", "Y_KIBA"]],
        
        # Loss weights
        contrastive_weight: float,
        complexity_weight: float,
        accuracy_weight: float,
        reconstruction_weight: float,
        lambda_train: Optional[List[float]], # TODO: NEW This is default [1, 5, 0] (used in DiGress)
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_dtype = torch.float32
        
        # Create encoders based on type
        encoder_type = ResidualEncoder if encoder_type == "resnet" else TransformerEncoder
        
        # Drug encoders
        self.drug_encoders = nn.ModuleDict()
        for feat_name in drug_features:
            self.drug_encoders[feat_name] = encoder_type(
                input_dim=drug_features[feat_name],
                output_dim=encoder_kwargs["hidden_dim"] if len(drug_features) > 1 else embedding_dim,
                **encoder_kwargs
            )
        
        # Target encoders
        self.target_encoders = nn.ModuleDict()
        for feat_name in target_features:
            self.target_encoders[feat_name] = encoder_type(
                input_dim=target_features[feat_name],
                output_dim=encoder_kwargs["hidden_dim"] if len(target_features) > 1 else embedding_dim,
                **encoder_kwargs
            )
        
        # Intra-branch aggregators
        aggregator_type = ConcatAggregator if aggregator_type == "concat" else AttentiveAggregator
        
        # Drug aggregator
        if len(drug_features) > 1:
            self.drug_aggregator = aggregator_type(
                input_dim=encoder_kwargs["hidden_dim"],
                output_dim=embedding_dim,
                n_features=len(drug_features),
                **aggregator_kwargs
            )
        else:
            self.drug_aggregator = None  # No aggregation needed for single feature
        
        # TODO: NEW
        self.drug_variational_head = KLVariationalHead(
            input_dim=embedding_dim,
            output_dim=embedding_dim
        )

        # Target aggregator
        if len(target_features) > 1:
            self.target_aggregator = aggregator_type(
                input_dim=encoder_kwargs["hidden_dim"],
                output_dim=embedding_dim,
                n_features=len(target_features),
                **aggregator_kwargs
            )
        else:
            self.target_aggregator = None  # No aggregation needed for single feature
        
        # Inter-branch aggregator (for DTI prediction)
        if phase in ["train", "finetune"]:
            self.fusion = CrossAttentionFusion(
                input_dim=embedding_dim,
                **fusion_kwargs
            )
            
            # DTI prediction head
            self.dti_head = DTIHead(
                input_dim=fusion_kwargs["output_dim"],
                **dti_head_kwargs
            )
        
        # Contrastive (for pretraining)
        if phase != "pretrain_target":
            self.drug_infonce_head = InfoNCEHead(
                input_dim=embedding_dim,
                **infonce_head_kwargs
            )
        if phase != "pretrain_drug":
            self.target_infonce_head = InfoNCEHead(
                input_dim=embedding_dim,
                **infonce_head_kwargs
            )
        
        #########################################################
        # Diffusion decoder
        #########################################################
        if phase != "pretrain_target":
            self.T = diffusion_steps
            self.best_val_nll = 1e8
            self.val_counter = 0
            self.sample_every_val = 1000
            self.visualization_tools = MolecularVisualization(dataset_infos["general"]["atom_types"])
            self.graph_converter = SmilesToPyG() # default atom and bond encodings used here bcs lazy

            # Limit distribution for forward noise
            self.nodes_dist = DistributionNodes(dataset_infos["dataset"]["node_count_distribution"])
            self.limit_dist = PlaceHolder(
                X=torch.tensor(dataset_infos["dataset"]["node_marginals"]).to(self.device), 
                E=torch.tensor(dataset_infos["dataset"]["edge_marginals"]).to(self.device), 
                y=torch.ones(graph_transformer_kwargs["output_dims"]["y"]) / graph_transformer_kwargs["output_dims"]["y"]
            )
            # Noise schedule and transition model for applying noise to clean graphs
            self.noise_schedule = PredefinedNoiseScheduleDiscrete(timesteps=diffusion_steps)
            self.transition_model = MarginalUniformTransition(
                x_marginals=torch.tensor(dataset_infos["dataset"]["node_marginals"]).to(self.device),
                e_marginals=torch.tensor(dataset_infos["dataset"]["edge_marginals"]).to(self.device),
                y_classes=graph_transformer_kwargs["output_dims"]["y"]
            )
            # Augmentation for adding features to the discretized noisy graphs
            self.augmentation = Augmentation(
                valencies=dataset_infos["general"]["atom_valencies"],
                max_weight=dataset_infos["general"]["max_weight"],
                atom_weights=dataset_infos["general"]["atom_weights"],
                max_n_nodes=dataset_infos["general"]["max_n_nodes"]
            )
            # Denoising graph transformer
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

            # TODO: Reconstruction loss
            self.train_reconstruction_loss = TrainLossDiscrete(lambda_train = lambda_train)
            self.train_molecular_metrics = TrainMolecularMetricsDiscrete(dataset_infos["general"]["atom_types"])

            self.val_nll = NLL()                      # used at the end of compute_val_loss
            self.val_X_kl = SumExceptBatchKL()        # used in nll component (compute_Lt) -> loss_all_t (diffusion loss)
            self.val_E_kl = SumExceptBatchKL()
            self.val_X_logp = SumExceptBatchMetric()  # used in nll component (reconstruction_logp) -> loss_term_0 (reconstruction loss)
            self.val_E_logp = SumExceptBatchMetric()
            self.test_nll = NLL()
            self.test_X_kl = SumExceptBatchKL()
            self.test_E_kl = SumExceptBatchKL()
            self.test_X_logp = SumExceptBatchMetric()
            self.test_E_logp = SumExceptBatchMetric()
            # chemical validity metrics? TODO (tanimoto similarity to ground truth & valid/invalid)
            # called `sampling_metrics` in DiGress
        
        # DTI prediction loss
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        # Loss weights for DTI prediction (based on inverse frequency in combined DTI dataset)
        self.register_buffer('loss_weights', torch.tensor([1.0, 0.903964, 0.282992, 0.755172]))
        # TODO: handle this dti loss a smarter way... how does this work when there are only one vs. multiple targets?
        
        # Setup metrics based on phase
        self.phase = phase
        self.finetune_score = finetune_score
        
        if phase in ["pretrain_drug", "pretrain_target"]:
            # No specific metrics for pretraining - just log contrastive loss
            self.train_metrics = None
            self.val_metrics = None
            self.test_metrics = None
        elif phase == "train":
            # Multi-score metrics for general training
            self.train_metrics = DTIMetricsCollection(
                include_binary=True,
                include_real=True,
                real_score_names=["pKd", "pKi", "KIBA"],
                prefix="train/"
            )
            self.val_metrics = DTIMetricsCollection(
                include_binary=True,
                include_real=True,
                real_score_names=["pKd", "pKi", "KIBA"],
                prefix="val/"
            )
            self.test_metrics = DTIMetricsCollection(
                include_binary=True,
                include_real=True,
                real_score_names=["pKd", "pKi", "KIBA"],
                prefix="test/"
            )
        else:  # finetune
            # Single-score metrics for fine-tuning
            if finetune_score is None:
                raise ValueError("finetune_score must be specified for finetune phase")

            self.train_metrics = RealDTIMetrics(prefix="train/")
            self.val_metrics = RealDTIMetrics(prefix="val/")
            self.test_metrics = RealDTIMetrics(prefix="test/")
        
        # Store aggregator type for handling different return formats
        self.aggregator_type = aggregator_type
        
        # Note: Checkpoint loading is now handled in run.py setup function
        # to support both single and dual checkpoint loading
    
    def load_pretrained_weights(self, checkpoint_path: str = None, drug_checkpoint_path: str = None, target_checkpoint_path: str = None) -> None:
        """
        Load pretrained weights from checkpoint(s) with smart matching.
        
        Args:
            checkpoint_path: Path to a single checkpoint file (original functionality)
            drug_checkpoint_path: Path to drug branch pretrained weights
            target_checkpoint_path: Path to target branch pretrained weights
        """
        if checkpoint_path is not None:
            # Original single-checkpoint loading
            self._load_single_checkpoint(checkpoint_path)
        elif drug_checkpoint_path is not None or target_checkpoint_path is not None:
            # Load drug and target branch weights separately
            self._load_branch_checkpoints(drug_checkpoint_path, target_checkpoint_path)
        else:
            logger.warning("No checkpoint path provided")
            return
    
    def _load_single_checkpoint(self, checkpoint_path: str) -> None:
        """Load weights from a single checkpoint file."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading pretrained weights from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Extract state dict
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Load compatible weights
        model_dict = self.state_dict()
        pretrained_dict = {}
        
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                pretrained_dict[k] = v
                logger.debug(f"Loaded weight: {k}")
            else:
                logger.warning(f"Skipping weight: {k} (shape mismatch or not found)")
        
        # Update model dict and load
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        
        logger.info(f"Loaded {len(pretrained_dict)} pretrained weights")
        
        # Optionally freeze encoder weights for fine-tuning
        if self.phase == "finetune":
            self.freeze_encoders()
    
    def _load_branch_checkpoints(self, drug_checkpoint_path: str = None, target_checkpoint_path: str = None) -> None:
        """Load drug and target branch weights from separate checkpoints."""
        model_dict = self.state_dict()
        total_loaded = 0
        
        # Load drug branch weights
        if drug_checkpoint_path is not None:
            drug_path = Path(drug_checkpoint_path)
            if drug_path.exists():
                logger.info(f"Loading drug branch weights from {drug_path}")
                drug_checkpoint = torch.load(drug_path, map_location="cpu")
                
                # Extract state dict
                if "state_dict" in drug_checkpoint:
                    drug_state_dict = drug_checkpoint["state_dict"]
                else:
                    drug_state_dict = drug_checkpoint
                
                # Load drug-specific weights
                drug_loaded = 0
                for k, v in drug_state_dict.items():
                    if k.startswith("drug_") and k in model_dict and model_dict[k].shape == v.shape:
                        model_dict[k] = v
                        drug_loaded += 1
                        logger.debug(f"Loaded drug weight: {k}")
                
                logger.info(f"Loaded {drug_loaded} drug branch weights")
                total_loaded += drug_loaded
            else:
                logger.warning(f"Drug checkpoint not found: {drug_path}")
        
        # Load target branch weights
        if target_checkpoint_path is not None:
            target_path = Path(target_checkpoint_path)
            if target_path.exists():
                logger.info(f"Loading target branch weights from {target_path}")
                target_checkpoint = torch.load(target_path, map_location="cpu")
                
                # Extract state dict
                if "state_dict" in target_checkpoint:
                    target_state_dict = target_checkpoint["state_dict"]
                else:
                    target_state_dict = target_checkpoint
                
                # Load target-specific weights
                target_loaded = 0
                for k, v in target_state_dict.items():
                    if k.startswith("target_") and k in model_dict and model_dict[k].shape == v.shape:
                        model_dict[k] = v
                        target_loaded += 1
                        logger.debug(f"Loaded target weight: {k}")
                
                logger.info(f"Loaded {target_loaded} target branch weights")
                total_loaded += target_loaded
            else:
                logger.warning(f"Target checkpoint not found: {target_path}")
        
        # Update model
        self.load_state_dict(model_dict)
        logger.info(f"Total loaded weights: {total_loaded}")
        
        # Optionally freeze encoder weights for fine-tuning
        if self.phase == "finetune":
            self.freeze_encoders()
    
    def freeze_encoders(self) -> None:
        """Freeze encoder weights during fine-tuning."""
        logger.info("Freezing encoder weights for fine-tuning")
        
        for param in self.drug_encoders.parameters():
            param.requires_grad = False
        
        for param in self.target_encoders.parameters():
            param.requires_grad = False
    
    def unfreeze_encoders(self) -> None:
        """Unfreeze encoder weights."""
        logger.info("Unfreezing encoder weights")
        
        for param in self.drug_encoders.parameters():
            param.requires_grad = True
        
        for param in self.target_encoders.parameters():
            param.requires_grad = True
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler using utility function."""
        return configure_optimizer_and_scheduler(
            model_parameters=self.parameters(),
            learning_rate=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            scheduler=self.hparams.scheduler,
            trainer=self.trainer
        )
    
    def _encode_drug_features(self, drug_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Encode and aggregate drug features.
        
        Args:
            drug_features: List of drug feature tensors
            
        Returns:
            - Aggregated drug embedding
            - Optional: drug_attention weights (if using attentive aggregator)
        """
        # Encode individual drug features
        drug_embeddings = []
        for feat_name, feat in zip(self.hparams.drug_features, drug_features):
            encoded = self.drug_encoders[feat_name](feat)
            drug_embeddings.append(encoded)
        
        # Aggregate drug embeddings
        if self.drug_aggregator is not None:
            if self.aggregator_type == "attentive":
                drug_embedding, drug_attention = self.drug_aggregator(drug_embeddings)
                return drug_embedding, drug_attention
            else:
                drug_embedding = self.drug_aggregator(drug_embeddings)
                return drug_embedding
        else:
            return drug_embeddings[0]  # Single feature, no aggregation needed
    
    def _encode_target_features(self, target_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Encode and aggregate target features.
        
        Args:
            target_features: List of target feature tensors
            
        Returns:
            Aggregated target embedding
        """
        # Encode individual target features
        target_embeddings = []
        for feat_name, feat in zip(self.hparams.target_features, target_features):
            encoded = self.target_encoders[feat_name](feat)
            target_embeddings.append(encoded)
        
        # Aggregate target embeddings
        if self.target_aggregator is not None:
            if self.aggregator_type == "attentive":
                target_embedding, target_attention = self.target_aggregator(target_embeddings)
                return target_embedding, target_attention
            else:
                target_embedding = self.target_aggregator(target_embeddings)
                return target_embedding
        else:
            return target_embeddings[0]  # Single feature, no aggregation needed
    
    def _apply_noise(self, X, E, node_mask):
        """
        Apply noise to the drug graph
        # TODO: NEW
        """
        # Sample timestep t (uniformly from [0, T])
        t_int = torch.randint(0, self.T + 1, size=(X.size(0), 1), device=X.device).float()
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
        probX = X @ Qtb.X               # (bs, n, dx_out)    - node transition probabilities p(X_t | X_0)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out) - edge transition probabilities p(E_t | E_0)

        # Sample discrete features from the transition probabilities
        sampled_t = sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)
        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)
        
        z_t = PlaceHolder(X=X_t, E=E_t).type_as(X_t).mask(node_mask) # NOTE: noise is NOT applied to y!
        
        noisy_data = {
            't_int': t_int, 't': t_float, 'beta_t': beta_t, 
            'alpha_s_bar': alpha_s_bar, 'alpha_t_bar': alpha_t_bar, 
            'X_t': z_t.X, 'E_t': z_t.E, 'node_mask': node_mask} #'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def _augment_data(self, noisy_data):
        """
        At every training step (after adding noise & sampling discrete features)
        extra features are computed (graph-structural features & molecular features)
        and appended to the graph transformer input.
        + timestep info t is added to graph-level features y to inform the model about the noise level
        # TODO: NEW
        """
        extra_features = self.augmentation(noisy_data)
        extra_X = extra_features.X
        extra_E = extra_features.E
        extra_y = extra_features.y

        t = noisy_data['t'] # normalized timestep
        extra_y = torch.cat((extra_y, t), dim=1)
        # X gets 8 extra features, E none and y gets 13
        return PlaceHolder(X=extra_X, E=extra_E, y=extra_y)
    
    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = mask_distributions(
            true_X=limit_X.clone(),
            true_E=limit_E.clone(),
            pred_X=probX,
            pred_E=probE,
            node_mask=node_mask
        )

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')

        return sum_except_batch(kl_distance_X) + \
               sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = posterior_distributions(
            X=X, E=E, y=y, 
            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'], y_t=noisy_data['y_t'], 
            Qt=Qt, Qsb=Qsb, Qtb=Qtb
        )
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))

        prob_pred = posterior_distributions(
            X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'], y_t=noisy_data['y_t'], 
            Qt=Qt, Qsb=Qsb, Qtb=Qtb
        )
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = mask_distributions(
            true_X=prob_true.X,
            true_E=prob_true.E,
            pred_X=prob_pred.X,
            pred_E=prob_pred.E,
            node_mask=node_mask
        )
        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, y, node_mask):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1).type_as(y0)}
        extra_data = self._augment_data(noisy_data)
        # here only the denoising is called
        pred0 = self._decode_drug(y0, noisy_data, extra_data, node_mask)

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return PlaceHolder(X=probX0, E=probE0, y=proby0)

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False):
        """
        Computes an estimator for the variational lower bound.
        """
        t = noisy_data['t']

        # 1. Prior on graph size: log p(N) where N is number of nodes
        # This encourages the model to generate graphs of realistic sizes
        N = node_mask.sum(1).long()
        log_pN = self.nodes_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        # At T, the noisy data should be close to the prior (marginal distribution)
        kl_prior = self.kl_prior(X, E, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, y, node_mask)

        loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log())

        # Combine terms to get the ELBO
        # Note: We negate log_pN and loss_term_0 because we want to maximize them
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)        # Average over the batch

        if wandb.run:
            wandb.log({"kl prior": kl_prior.mean(),
                       "Estimator loss terms": loss_all_t.mean(),
                       "log_pn": log_pN.mean(),
                       "loss_term_0": loss_term_0,
                       'batch_test_nll' if test else 'val_nll': nll}, commit=False)
        return nll

    @torch.no_grad()
    def sample_batch(self, batch: Batch) -> list[Chem.Mol]:
        """
        Used to iteratively sample a clean graph from a noisy graph.
        # NOTE: this VERY IMPORTANT function is defined very differently in the DiGress & DiffMS implementations
        """
        dense_data, node_mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # Sample initial noisy graph G^T from the limit distribution
        z_T = sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, batch.y # noisy nodes & edges along with conditional features y
        assert (E == torch.transpose(E, 1, 2)).all()

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in tqdm(reversed(range(0, self.T)), desc='Sampling', leave=False):
            s_array = s_int * torch.ones((len(batch), 1), dtype=torch.float32, device=self.device)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s (I think we can use discrete_sampled_s to save intermediate samples through diffusion steps)
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask)
            X, E, y = sampled_s.X, sampled_s.E, batch.y # y is kept fixed

        # Final sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, batch.y
        # In DiffMS, X & y are kept fixed and only E is updated
        # In DiGress all three X, E, y are updated

        mols = []
        for nodes, adj_mat in zip(X, E):
            mols.append(self.visualization_tools.mol_from_graphs(nodes, adj_mat))
        return mols

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self._augment_data(noisy_data)
        pred = self._decode_drug(y_t, noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        p_s_and_t_given_0_X = compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)


    # TODO: we need to update the forward pass to include the decoding process (when NOT pretrain-target)
    # IMPORTANT STEPS UNIQUE TO FULL MODEL:
    # - dataloader returns batch of PyG data objects in batch["G"]
    # - to dense_data, node_mask with utils.to_dense(x, edge_index, edge_attr, batch)
    # - noisy_data = apply_noise(G, node_mask) to transform clean dense graph to noisy dense graph (discretized!)
    # - extra_data = augment_data(G_t, noise_params, node_mask) to compute extra features
    # - extract X, E, y from both and concat into X, E, y input for graph transformer
    # - forward pass through the graph transformer drug_decoder(X, E, y, node_mask)
    # - predictions of size output dims (X, E, y) as **PlaceHolder**
    # DURING TRAINING/VAL/TEST STEP THIS IS ONE SHOT GENERATION OF CLEAN GRAPH FROM NOISY GRAPH
    # DURING VAL/TEST EPOCH END, WE DO ITERATIVE SAMPLING W/ sample_batch() & sampling_metrics or chemical_validity_metrics (not yet implemented)
    # - train loss w/ train_reconstruction_loss class using predicted & true X, E, y (y has no contribution to the loss due to lambda_train)
    # - val/test use complex compute_val_loss method (NLL)
    # - update ALL metrics train/val/test
    # - notably there seems to be a weird discrepancy between the loss/metrics computed in train step vs. val/test step (there might be lots of redundancy or unnecessary components here in the original implementation)

    def _decode_drug(self, drug_embedding, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((drug_embedding, extra_data.y)).float()
        return self.drug_decoder(X, E, y, node_mask)

    def forward(
        self, 
        drug_features: List[torch.Tensor] = None, 
        target_features: List[torch.Tensor] = None,
        noisy_data = None, extra_data = None, node_mask = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the multi-hybrid model.
        
        Args:
            drug_features: List of drug feature tensors
            target_features: List of target feature tensors
            noisy_data, extra_data, node_mask: inputs for graph denoising
        Returns:
            Dictionary containing predictions and embeddings
        """
        if (drug_features is None or len(drug_features) == 0 or noisy_data is None or extra_data is None or node_mask is None) and (target_features is None or len(target_features) == 0):
            raise ValueError("Either drug_features or target_features must be provided")
        
        # Encode drug features & reparametrize
        if drug_features is not None and len(drug_features) > 0:
            drug_result = self._encode_drug_features(drug_features)
            if self.aggregator_type == AttentiveAggregator and self.drug_aggregator is not None:
                drug_embedding, drug_attention = drug_result
            else:
                drug_embedding, drug_attention = drug_result, None
            drug_mu, drug_logvar, drug_embedding = self.drug_variational_head(drug_embedding)
        else:
            drug_embedding, drug_attention, drug_mu, drug_logvar = None, None, None, None
        
        # Encode target features
        if target_features is not None and len(target_features) > 0:
            target_result = self._encode_target_features(target_features)
            if self.aggregator_type == AttentiveAggregator and self.target_aggregator is not None:
                target_embedding, target_attention = target_result
            else:
                target_embedding, target_attention = target_result, None
        else:
            target_embedding, target_attention = None, None
        
        outputs = {
            "drug_embedding": drug_embedding,
            "target_embedding": target_embedding,
            "drug_mu": drug_mu,
            "drug_logvar": drug_logvar,
            "drug_attention": drug_attention,
            "target_attention": target_attention,
        }
        
        # DTI prediction (for train/finetune phases)
        if self.phase in ["train", "finetune"]:
            # Ensure both embeddings are available for DTI prediction
            if drug_embedding is None or target_embedding is None:
                raise ValueError("Both drug and target embeddings are required for DTI prediction")
            
            # Inter-branch aggregation with cross-attention module
            combined_features = self.fusion(drug_embedding, target_embedding)
            
            # DTI predictions
            predictions = self.dti_head(combined_features)
            outputs.update({
                "combined_features": combined_features,
                "binary_logits": predictions["binary_logits"],
                "binary_pred": torch.sigmoid(predictions["binary_logits"]),
                "pKd_pred": predictions["pKd_pred"],
                "pKi_pred": predictions["pKi_pred"],
                "KIBA_pred": predictions["KIBA_pred"],
            })
        
        # Denoising step
        if self.phase != "pretrain_target":
            outputs.update({
                "drug_pred": self._decode_drug(drug_embedding, noisy_data, extra_data, node_mask)
                }) # PlaceHolder obj w/ X, E, y
        
        return outputs
    
    def _get_features_from_batch(
        self, 
        batch: Dict[str, Any]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Extract and prepare features from batch."""
        drug_feats = []
        target_feats = []
        smiles = []
        
        if self.phase == "pretrain_drug":
            # For drug pretraining, extract drug features from batch["features"]
            for feat_name in self.hparams.drug_features:
                if feat_name in batch["features"]:
                    drug_feats.append(batch["features"][feat_name])
                else:
                    raise ValueError(f"Missing drug feature '{feat_name}' in batch")
            smiles = batch["representations"]["smiles"]
    
        elif self.phase == "pretrain_target":
            # For target pretraining, extract target features from batch["features"]
            for feat_name in self.hparams.target_features:
                if feat_name in batch["features"]:
                    target_feats.append(batch["features"][feat_name])
                else:
                    raise ValueError(f"Missing target feature '{feat_name}' in batch")
            
        else: # DTI batch
            for feat_name in self.hparams.drug_features:
                if feat_name in batch["drug"]["features"]:
                    drug_feats.append(batch["drug"]["features"][feat_name])
                else:
                    raise ValueError(f"Missing drug feature '{feat_name}' in batch")
            smiles = batch["drug"]["representations"]["smiles"]
            for feat_name in self.hparams.target_features:
                if feat_name in batch["target"]["features"]:
                    target_feats.append(batch["target"]["features"][feat_name])
                else:
                    raise ValueError(f"Missing target feature '{feat_name}' in batch")
        
        return drug_feats, target_feats, smiles
    
    def _compute_dti_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, Any],
        step_name: str = "train"
    ) -> torch.Tensor:
        """
        Compute DTI loss with proper masking and weighting.
        
        Args:
            outputs: Model outputs containing predictions
            batch: Batch data containing targets and masks
            step_name: Name of the training step (train/val/test)
            
        Returns:
            Combined loss tensor
        """
        total_loss = 0.0
        loss_count = 0
        
        # Binary loss (always present)
        binary_loss = self.bce_loss(
            outputs["binary_logits"].squeeze(), 
            batch["y"]["Y"].float()
        )
        total_loss += self.loss_weights[0] * binary_loss
        loss_count += 1
        
        # Real-valued losses (with masking)
        real_losses = []
        
        # pKd loss
        if "Y_pKd" in batch["y"] and batch["y"]["Y_pKd_mask"].any():
            valid_mask = batch["y"]["Y_pKd_mask"]
            pred_kd = outputs["pKd_pred"][valid_mask].squeeze()
            target_kd = batch["y"]["Y_pKd"][valid_mask].squeeze()
            kd_loss = self.mse_loss(pred_kd, target_kd)
            total_loss += self.loss_weights[1] * kd_loss
            loss_count += 1
            real_losses.append(("pKd", kd_loss.item()))
        
        # pKi loss
        if "Y_pKi" in batch["y"] and batch["y"]["Y_pKi_mask"].any():
            valid_mask = batch["y"]["Y_pKi_mask"]
            pred_ki = outputs["pKi_pred"][valid_mask].squeeze()
            target_ki = batch["y"]["Y_pKi"][valid_mask].squeeze()
            ki_loss = self.mse_loss(pred_ki, target_ki)
            total_loss += self.loss_weights[2] * ki_loss
            loss_count += 1
            real_losses.append(("pKi", ki_loss.item()))
        
        # KIBA loss
        if "Y_KIBA" in batch["y"] and batch["y"]["Y_KIBA_mask"].any():
            valid_mask = batch["y"]["Y_KIBA_mask"]
            pred_kiba = outputs["KIBA_pred"][valid_mask].squeeze()
            target_kiba = batch["y"]["Y_KIBA"][valid_mask].squeeze()
            kiba_loss = self.mse_loss(pred_kiba, target_kiba)
            total_loss += self.loss_weights[3] * kiba_loss
            loss_count += 1
            real_losses.append(("KIBA", kiba_loss.item()))
        
        # Normalize by number of active losses
        if loss_count > 0:
            total_loss /= loss_count
        
        # Log individual loss components
        self.log(f"{step_name}/binary_loss", binary_loss.item(), prog_bar=False)
        for loss_name, loss_value in real_losses:
            self.log(f"{step_name}/{loss_name}_loss", loss_value, prog_bar=False)
        
        return total_loss
    
    def _common_step(
        self, 
        batch: Dict[str, Any],
        step_name: str = "train"
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Common step logic shared across train/val/test.
        
        Args:
            batch: Batch data
            step_name: Name of the training step (train/val/test)
            
        Returns:
            Tuple of (outputs, loss)
        """
        # Get features
        drug_features, target_features, smiles = self._get_features_from_batch(batch)

        # Get molecular graph data
        if self.phase != "pretrain_target":
            # SMILES -> PyG -> X, E
            graphs = self.graph_converter.smiles_to_pyg_batch(smiles)
            dense_data, node_mask = to_dense(graphs.x, graphs.edge_index, graphs.edge_attr, graphs.batch)
            dense_data = dense_data.mask(node_mask)

            # Apply forward diffusion process (add noise)
            noisy_data = self._apply_noise(dense_data.X, dense_data.E, node_mask)
            extra_data = self._augment_data(noisy_data)
        else:
            dense_data, noisy_data, extra_data, node_mask = None, None, None, None

        # Forward pass
        outputs = self(drug_features, target_features, noisy_data, extra_data, node_mask)

        if noisy_data is not None:
            noisy_data['y_t'] = outputs['drug_embedding']

        losses = {
            "contrastive": 0.,
            "complexity": 0.,
            "reconstruction": 0.,  # handled differently in train/val/test
            "accuracy": 0.,
        }

        if self.phase == "pretrain_target":
            # Only contrastive loss, determined by target_infonce_head
            losses["contrastive"] = self.hparams.contrastive_weight * self.target_infonce_head(
                outputs["target_embedding"],
                batch["features"]["FP-ESP"]
            )
            # No complexity, accuracy, or reconstruction
        elif self.phase == "pretrain_drug":
            # Contrastive loss only from drug_infonce_head
            losses["contrastive"] = self.hparams.contrastive_weight * self.drug_infonce_head(
                outputs["drug_embedding"],
                batch["features"]["FP-Morgan"]
            )
            # Complexity and reconstruction losses included
            losses["complexity"] = self.hparams.complexity_weight * self.drug_variational_head.kl_divergence(
                outputs["drug_mu"], outputs["drug_logvar"]
            )
            # No accuracy loss
        else:
            # All losses included
            # Contrastive is average of drug and target
            drug_contrastive = self.drug_infonce_head(
                outputs["drug_embedding"],
                batch["drug"]["features"]["FP-Morgan"]
            )
            target_contrastive = self.target_infonce_head(
                outputs["target_embedding"],
                batch["target"]["features"]["FP-ESP"]
            )
            losses["contrastive"] = self.hparams.contrastive_weight * (drug_contrastive + target_contrastive) / 2.0
            losses["complexity"] = self.hparams.complexity_weight * self.drug_variational_head.kl_divergence(
                outputs["drug_mu"], outputs["drug_logvar"]
            )
            losses["accuracy"] = self._compute_dti_loss(outputs, batch, step_name)
            # reconstruction handled outside

        return outputs, losses, dense_data, noisy_data, node_mask

    def on_train_epoch_start(self):
        self.print("Starting train epoch...")
        if self.train_metrics is not None:
            self.train_metrics.reset()
        if self.phase != "pretrain_target":
            self.train_molecular_metrics.reset()
            self.train_reconstruction_loss.reset()

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """Training step."""
        outputs, losses, dense_data, _, _ = self._common_step(batch, "train")

        if self.phase != "pretrain_target":
            losses["reconstruction"] = self.hparams.reconstruction_weight * self.train_reconstruction_loss(
                masked_pred_X=outputs["drug_pred"].X, 
                masked_pred_E=outputs["drug_pred"].E, 
                pred_y=outputs["drug_pred"].y,
                true_X=dense_data.X,
                true_E=dense_data.E,
                true_y=outputs["drug_embedding"],
                log=False
            )
            self.train_molecular_metrics(
                masked_pred_X=outputs["drug_pred"].X, 
                masked_pred_E=outputs["drug_pred"].E, 
                true_X=dense_data.X,
                true_E=dense_data.E,
                log=False
            )
        else:
            losses["reconstruction"] = 0.0
        
        loss = sum(losses.values())
        
        # Update metrics based on phase
        if self.phase == "train" and self.train_metrics is not None:
            self._update_multi_score_metrics(outputs, batch, self.train_metrics)
        elif self.phase == "finetune" and self.train_metrics is not None:
            self._update_single_score_metrics(outputs, batch, self.train_metrics)
        
        # Logging
        self.log("train/loss", loss)

        return loss
    
    def on_validation_epoch_start(self):
        self.print("Starting validation epoch...")
        if self.val_metrics is not None:
            self.val_metrics.reset()
        if self.phase != "pretrain_target":
            self.val_nll.reset()
            self.val_X_kl.reset()
            self.val_E_kl.reset()
            self.val_X_logp.reset()
            self.val_E_logp.reset()
            if self.global_rank == 0:
                self.val_counter += 1

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """Validation step."""
        outputs, losses, dense_data, noisy_data, node_mask = self._common_step(batch, "val")

        if self.phase != "pretrain_target":
            pred = outputs["drug_pred"]
            pred.Y = outputs["drug_embedding"] # overwrite pred.Y with true y
            losses["reconstruction"] = self.compute_val_loss(
                pred, noisy_data, dense_data.X, dense_data.E, outputs["drug_embedding"], node_mask, test=False
            )
            # Generate molecules for evaluation (periodically)
            if self.val_counter % self.sample_every_val == 0:
                if batch_idx == 0 and self.global_rank == 0:
                    self.print(f"Generating {self.val_num_samples} molecules per validation sample for evaluation...")

                smiles = batch["drug"]["representations"]["smiles"] if self.phase != "pretrain_drug" else batch["representations"]["smiles"]
                mols_true = [self.graph_converter.smiles_to_mol(smiles) for smiles in smiles]
                mols_predicted = [list() for _ in range(len(smiles))]
                batch = self.graph_converter.smiles_to_pyg_batch(smiles)
                batch.y = outputs["drug_embedding"]

                # iterative denoising w/ sample_batch()
                for _ in range(self.hparams.num_samples_to_generate):
                    for idx, mol in enumerate(self.sample_batch(batch)):
                        mols_predicted[idx].append(mol)
            
                # for idx in range(len(data)):
                    # NOTE: here we see DiffMS specific molecular validity metrics being computed
                    # based on predicted iteratively denoised molecules & true molecules
                    # self.val_k_acc.update(predicted_mols[idx], true_mols[idx])
                    # self.val_sim_metrics.update(predicted_mols[idx], true_mols[idx])
                    # self.val_validity.update(predicted_mols[idx])
        else:
            losses["reconstruction"] = 0.0

        loss = sum(losses.values())
        
        # Update metrics based on phase
        if self.phase == "train" and self.val_metrics is not None:
            self._update_multi_score_metrics(outputs, batch, self.val_metrics)
        elif self.phase == "finetune" and self.val_metrics is not None:
            self._update_single_score_metrics(outputs, batch, self.val_metrics)
        
        # Logging
        self.log("val/loss", loss)

        return loss
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        """Test step."""
        outputs, losses, dense_data, noisy_data, node_mask = self._common_step(batch, "val")

        if self.phase != "pretrain_target":
            pred = outputs["drug_pred"]
            pred.Y = outputs["drug_embedding"] # overwrite pred.Y with true y
            losses["reconstruction"] = self.compute_val_loss(
                pred, noisy_data, dense_data.X, dense_data.E, outputs["drug_embedding"], node_mask, test=True
            )
            # Generate molecules for evaluation
            smiles = batch["drug"]["representations"]["smiles"] if self.phase != "pretrain_drug" else batch["representations"]["smiles"]
            mols_true = [self.graph_converter.smiles_to_mol(smiles) for smiles in smiles]
            mols_predicted = [list() for _ in range(len(smiles))]
            graph_batch = self.graph_converter.smiles_to_pyg_batch(smiles)
            graph_batch.y = outputs["drug_embedding"]

            # iterative denoising w/ sample_batch()
            for _ in range(self.hparams.num_samples_to_generate):
                for idx, mol in enumerate(self.sample_batch(graph_batch)):
                    mols_predicted[idx].append(mol)
        
            # for idx in range(len(data)):
                # NOTE: here we see DiffMS specific molecular validity metrics being computed
                # based on predicted iteratively denoised molecules & true molecules
                # self.val_k_acc.update(predicted_mols[idx], true_mols[idx])
                # self.val_sim_metrics.update(predicted_mols[idx], true_mols[idx])
                # self.val_validity.update(predicted_mols[idx])

            # TODO: possibly save the predicted molecules to disk 
        else:
            losses["reconstruction"] = 0.0

        loss = sum(losses.values())
        
        # Update metrics based on phase
        if self.phase == "train" and self.test_metrics is not None:
            self._update_multi_score_metrics(outputs, batch, self.test_metrics)
        elif self.phase == "finetune" and self.test_metrics is not None:
            self._update_single_score_metrics(outputs, batch, self.test_metrics)
        
        # Logging
        self.log("test/loss", loss)

        return loss
    
    def _update_multi_score_metrics(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, Any], 
        metrics: DTIMetricsCollection
    ):
        """Update multi-score metrics for general training."""
        # Binary metrics
        binary_preds = outputs["binary_pred"].squeeze()
        binary_targets = batch["y"]["Y"]
        
        # Real-valued metrics
        real_preds = {}
        real_targets = {}
        real_masks = {}
        
        # Check each score
        for score_name, pred_key in [("pKd", "pKd_pred"), ("pKi", "pKi_pred"), ("KIBA", "KIBA_pred")]:
            batch_key = f"Y_{score_name}"
            mask_key = f"{batch_key}_mask"
            
            if batch_key in batch["y"] and mask_key in batch["y"]:
                real_preds[score_name] = outputs[pred_key].squeeze()
                real_targets[score_name] = batch["y"][batch_key]
                real_masks[score_name] = batch["y"][mask_key]
        
        # Update metrics
        metrics.update(
            binary_preds=binary_preds,
            binary_targets=binary_targets,
            real_preds=real_preds,
            real_targets=real_targets,
            real_masks=real_masks
        )
    
    def _update_single_score_metrics(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, Any], 
        metrics
    ):
        """Update single-score metrics for fine-tuning."""
        if self.finetune_score is None:
            return
        
        # Get the specific score being fine-tuned
        pred_key = f"{self.finetune_score.split('_')[1]}_pred"
        
        score_pred = outputs[pred_key].squeeze()
        score_target = batch["y"][self.finetune_score]
        score_mask = batch["y"][f"{self.finetune_score}_mask"]
        
        # Update metrics only for valid samples
        if score_mask.any():
            metrics.update(score_pred, score_target, score_mask)
    
    def on_train_epoch_end(self):
        """Compute and log training metrics."""
        if self.train_metrics is not None:
            train_metrics = self.train_metrics.compute()
            for name, value in train_metrics.items():
                self.log(name, value)
            self.train_metrics.reset()
    
    def on_validation_epoch_end(self):
        """Compute and log validation metrics."""
        if self.val_metrics is not None:
            val_metrics = self.val_metrics.compute()
            for name, value in val_metrics.items():
                self.log(name, value)
            self.val_metrics.reset()
        if self.phase != "pretrain_target":
            metrics = [
                self.val_nll.compute(), 
                self.val_X_kl.compute(), 
                self.val_E_kl.compute(),
                self.val_X_logp.compute(), 
                self.val_E_logp.compute()
            ]

            log_dict = {
                "val/NLL": metrics[0],
                "val/X_KL": metrics[1],
                "val/E_KL": metrics[2],
                "val/X_logp": metrics[3],
                "val/E_logp": metrics[4]
            }

            # TODO: these are DiffMS metrics related to molecular validity!

            # if self.val_counter % self.sample_every_val == 0:
            #     for key, value in self.val_k_acc.compute().items():
            #         log_dict[f"val/{key}"] = value
            #     for key, value in self.val_sim_metrics.compute().items():
            #         log_dict[f"val/{key}"] = value
            #     log_dict["val/validity"] = self.val_validity.compute()

            self.log_dict(log_dict, sync_dist=True)
    
    def on_test_epoch_end(self):
        """Compute and log test metrics."""
        if self.test_metrics is not None:
            test_metrics = self.test_metrics.compute()
            for name, value in test_metrics.items():
                self.log(name, value)
            self.test_metrics.reset() 