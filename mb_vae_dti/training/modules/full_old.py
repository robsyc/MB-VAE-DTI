"""
Full DTI PyTorch Lightning module with
- Multi-modal input (multiple drug/target features with aggregation)  
- Variational drug branch with KL divergence
- Discrete diffusion decoder for drug reconstruction
- Multi-output prediction (DTI head for multiple simultaneous DTI score prediction)
- Contrastive pretraining support (InfoNCE loss for individual branch pretraining)
- Support for all training phases: pretrain_drug, pretrain_target, train, finetune
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Literal
import logging
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from collections import Counter

from mb_vae_dti.training.models import (
    ResidualEncoder, TransformerEncoder,
    ConcatAggregator, AttentiveAggregator, CrossAttentionFusion,
)
from .optimizer_utils import configure_optimizer_and_scheduler
from mb_vae_dti.training.models.heads import DTIHead, InfoNCEHead, KLVariationalHead
from mb_vae_dti.training.models.decoders import DiscreteDiffusionDecoder
from mb_vae_dti.training.metrics import DTIMetricsCollection, RealDTIMetrics
from mb_vae_dti.training.diffusion.utils import PlaceHolder, to_dense, sample_discrete_feature_noise
from .optimizer_utils import configure_optimizer_and_scheduler


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
        drug_features: Dict[str, int],
        target_features: Dict[str, int],
        encoder_type: Literal["resnet", "transformer"],
        encoder_kwargs: Optional[Dict],
        aggregator_type: Literal["concat", "attentive", "cross_attention"],
        aggregator_kwargs: Optional[Dict],
        inter_branch_aggregator_kwargs: Optional[Dict],
        
        # Head parameters
        dti_head_kwargs: Optional[Dict],
        infonce_head_kwargs: Optional[Dict],
        variational_head_kwargs: Optional[Dict],
        
        # Diffusion decoder parameters
        diffusion_decoder_kwargs: Optional[Dict],
        
        # Training parameters
        learning_rate: float,
        weight_decay: float,
        scheduler: Optional[Literal["const", "step", "one_cycle", "cosine"]],
        
        # Phase-specific parameters
        phase: Literal["pretrain_drug", "pretrain_target", "train", "finetune"],
        finetune_score: Optional[Literal["Y_pKd", "Y_KIBA"]],
        checkpoint_path: Optional[str],
        
        # Loss weights
        contrastive_weight: float,
        kl_weight: float,
        reconstruction_weight: float,
        temperature: float,
        
        # Dataset statistics (computed from training data)
        dataset_statistics: Optional[Dict],
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Validate phase-specific parameters
        if phase == "finetune" and finetune_score is None:
            raise ValueError("finetune_score must be specified for finetune phase")
        
        # Create encoders based on type
        encoder_class = ResidualEncoder if encoder_type == "resnet" else TransformerEncoder
        
        # Drug encoders
        self.drug_encoders = nn.ModuleDict()
        for feat_name in drug_features:
            self.drug_encoders[feat_name] = encoder_class(
                input_dim=drug_features[feat_name],
                **encoder_kwargs
            )
        
        # Target encoders
        self.target_encoders = nn.ModuleDict()
        for feat_name in target_features:
            self.target_encoders[feat_name] = encoder_class(
                input_dim=target_features[feat_name],
                **encoder_kwargs
            )
        
        # Intra-branch aggregators
        aggregator_type = ConcatAggregator if aggregator_type == "concat" else AttentiveAggregator
        
        # Drug aggregator
        if len(drug_features) > 1:
            self.drug_aggregator = aggregator_type(
                input_dim=encoder_kwargs["output_dim"],
                n_features=len(drug_features),
                **aggregator_kwargs
            )
        else:
            self.drug_aggregator = None
            
        # Target aggregator
        if len(target_features) > 1:
            self.target_aggregator = aggregator_type(
                input_dim=encoder_kwargs["output_dim"],
                n_features=len(target_features),
                **aggregator_kwargs
            )
        else:
            self.target_aggregator = None
        
        # Variational head for drug branch (always present)
        variational_kwargs = variational_head_kwargs or {}
        self.drug_variational_head = KLVariationalHead(
            input_dim=aggregator_kwargs["output_dim"],
            output_dim=variational_kwargs.get("latent_dim", 256)
        )
        self.latent_dim = variational_kwargs.get("latent_dim", 256)
        
        # Inter-branch aggregator (for DTI prediction)
        if phase in ["train", "finetune"]:
            self.inter_branch_aggregator = CrossAttentionFusion(
                input_dim=aggregator_kwargs["output_dim"],
                **inter_branch_aggregator_kwargs
            )
            
            # DTI prediction head
            self.dti_head = DTIHead(
                input_dim=inter_branch_aggregator_kwargs["output_dim"],
                **dti_head_kwargs
            )
        
        # Contrastive learning heads (for pretraining)
        if phase in ["pretrain_drug", "pretrain_target"]:
            contrastive_kwargs = infonce_head_kwargs or {}
            contrastive_kwargs.setdefault("temperature", temperature)
            
            if phase == "pretrain_drug":
                # Drug InfoNCE uses latent z as input
                self.drug_infonce_head = InfoNCEHead(
                    input_dim=self.latent_dim,
                    **contrastive_kwargs
                )
            elif phase == "pretrain_target":
                self.target_infonce_head = InfoNCEHead(
                    input_dim=aggregator_kwargs["output_dim"],
                    **contrastive_kwargs
                )
        
        # Diffusion decoder (for drug reconstruction during pretraining)
        if phase == "pretrain_drug":
            if dataset_statistics is None:
                logger.warning("Dataset statistics not provided - using defaults for diffusion decoder")
                # Default statistics (will be overridden by actual data)
                dataset_statistics = self._get_default_dataset_statistics()
            
            decoder_kwargs = diffusion_decoder_kwargs or {}
            
            # Update graph transformer input dims to include latent z
            graph_transformer_params = decoder_kwargs.get("graph_transformer_params", {})
            graph_transformer_params["input_dims"]["y"] = self.latent_dim
            decoder_kwargs["graph_transformer_params"] = graph_transformer_params
            
            self.diffusion_decoder = DiscreteDiffusionDecoder(
                dataset_infos=dataset_statistics,
                **decoder_kwargs
            )
            
            # Store atom/bond decoders for molecule conversion
            self.atom_decoder = dataset_statistics.get("atom_decoder", ['C', 'O', 'P', 'N', 'S', 'Cl', 'F', 'H'])
            self.bond_decoder = dataset_statistics.get("bond_decoder", {0: None, 1: BT.SINGLE, 2: BT.DOUBLE, 3: BT.TRIPLE, 4: BT.AROMATIC})
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        
        # Loss weights for DTI prediction
        self.register_buffer('dti_loss_weights', torch.tensor([1.0, 0.903964, 0.282992, 0.755172]))
        
        # Setup metrics based on phase
        self.phase = phase
        self.finetune_score = finetune_score
        
        if phase in ["pretrain_drug", "pretrain_target"]:
            # No specific metrics for pretraining
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
            self.train_metrics = RealDTIMetrics(prefix="train/")
            self.val_metrics = RealDTIMetrics(prefix="val/")
            self.test_metrics = RealDTIMetrics(prefix="test/")
        
        # Storage for manual analysis
        self.test_predictions = []
        self.test_targets = []
        
        # Store aggregator type for handling different return formats
        self.aggregator_type = aggregator_type
        
    def _get_default_dataset_statistics(self) -> Dict:
        """Get default dataset statistics for diffusion decoder initialization."""
        # These are reasonable defaults based on typical drug molecules
        return {
            "heavy_atom_counts": {i: 100 for i in range(10, 51)},  # Uniform distribution
            "max_heavy_atoms": 64,
            "max_mol_weight": 500.0,
            "atom_weights": {0: 12.01, 1: 16.00, 2: 30.97, 3: 14.01, 4: 32.07, 5: 35.45, 6: 19.00, 7: 1.01},
            "valencies": [4, 2, 5, 3, 2, 1, 1, 1],  # C, O, P, N, S, Cl, F, H
            "x_marginals": torch.tensor([0.736, 0.131, 0.001, 0.107, 0.012, 0.003, 0.010, 0.000]),
            "e_marginals": torch.tensor([0.913, 0.044, 0.006, 0.0002, 0.037]),
            "atom_decoder": ['C', 'O', 'P', 'N', 'S', 'Cl', 'F', 'H'],
            "bond_decoder": {0: None, 1: BT.SINGLE, 2: BT.DOUBLE, 3: BT.TRIPLE, 4: BT.AROMATIC}
        }
    
    def load_pretrained_weights(self, checkpoint_path: str = None, drug_checkpoint_path: str = None, target_checkpoint_path: str = None) -> None:
        """
        Load pretrained weights from checkpoint(s) with smart matching.
        
        Args:
            checkpoint_path: Path to a single checkpoint file
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
                
                # Load drug-specific weights (encoders, aggregator, variational head, decoder)
                drug_loaded = 0
                for k, v in drug_state_dict.items():
                    if (k.startswith("drug_") or k.startswith("diffusion_decoder")) and k in model_dict and model_dict[k].shape == v.shape:
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
    
    def _encode_drug_features(self, drug_features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode and aggregate drug features through VAE.
        
        Args:
            drug_features: List of drug feature tensors
            
        Returns:
            Tuple of (aggregated_embedding, z_sampled, kl_loss)
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
            else:
                drug_embedding = self.drug_aggregator(drug_embeddings)
                drug_attention = None
        else:
            drug_embedding = drug_embeddings[0]
            drug_attention = None
        
        # Pass through variational head
        z_sampled, kl_loss = self.drug_variational_head(drug_embedding)
        
        return drug_embedding, z_sampled, kl_loss, drug_attention
    
    def _encode_target_features(self, target_features: List[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode and aggregate target features.
        
        Args:
            target_features: List of target feature tensors
            
        Returns:
            Tuple of (aggregated_embedding, attention_weights)
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
            else:
                target_embedding = self.target_aggregator(target_embeddings)
                target_attention = None
        else:
            target_embedding = target_embeddings[0]
            target_attention = None
        
        return target_embedding, target_attention
    
    def _smiles_to_graph(self, smiles_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert SMILES strings to molecular graphs.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Tuple of (X, E, node_mask) where:
                X: Node features (batch_size, max_nodes, num_atom_types)
                E: Edge features (batch_size, max_nodes, max_nodes, num_edge_types)
                node_mask: Valid nodes mask (batch_size, max_nodes)
        """
        batch_size = len(smiles_list)
        max_nodes = self.diffusion_decoder.augmentation.extra_graph_features.max_n_nodes
        num_atom_types = len(self.atom_decoder)
        num_edge_types = len(self.bond_decoder)
        
        # Initialize tensors
        X = torch.zeros(batch_size, max_nodes, num_atom_types)
        E = torch.zeros(batch_size, max_nodes, max_nodes, num_edge_types)
        node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
        
        atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
        
        for idx, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                continue
            
            # Remove hydrogens
            mol = Chem.RemoveHs(mol)
            n_atoms = mol.GetNumAtoms()
            
            if n_atoms > max_nodes:
                logger.warning(f"Molecule has {n_atoms} atoms, exceeding max {max_nodes}")
                n_atoms = max_nodes
            
            # Set node features
            for i, atom in enumerate(mol.GetAtoms()):
                if i >= max_nodes:
                    break
                atom_type = atom.GetSymbol()
                if atom_type in atom_encoder:
                    X[idx, i, atom_encoder[atom_type]] = 1
                else:
                    # Default to carbon if unknown atom type
                    X[idx, i, 0] = 1
                node_mask[idx, i] = True
            
            # Set edge features
            # First set all edges to "no bond" (index 0)
            E[idx, :n_atoms, :n_atoms, 0] = 1
            
            # Then update with actual bonds
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                if i >= max_nodes or j >= max_nodes:
                    continue
                
                bond_type = bond.GetBondType()
                # Map bond type to edge feature index
                if bond_type == BT.SINGLE:
                    edge_idx = 1
                elif bond_type == BT.DOUBLE:
                    edge_idx = 2
                elif bond_type == BT.TRIPLE:
                    edge_idx = 3
                elif bond_type == BT.AROMATIC:
                    edge_idx = 4
                else:
                    edge_idx = 0  # Unknown bond type
                
                # Set bond in both directions
                E[idx, i, j, 0] = 0
                E[idx, i, j, edge_idx] = 1
                E[idx, j, i, 0] = 0
                E[idx, j, i, edge_idx] = 1
        
        return X, E, node_mask
    
    def forward(
        self, 
        drug_features: List[torch.Tensor] = None, 
        target_features: List[torch.Tensor] = None,
        drug_smiles: List[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the full model.
        
        Args:
            drug_features: List of drug feature tensors
            target_features: List of target feature tensors
            drug_smiles: List of SMILES strings (for reconstruction loss)
            
        Returns:
            Dictionary containing predictions, embeddings, and losses
        """
        if (drug_features is None or len(drug_features) == 0) and (target_features is None or len(target_features) == 0):
            raise ValueError("Either drug_features or target_features must be provided")
        
        outputs = {}
        
        # Encode drug features through VAE
        if drug_features is not None and len(drug_features) > 0:
            drug_embedding, z_sampled, kl_loss, drug_attention = self._encode_drug_features(drug_features)
            outputs["drug_embedding"] = drug_embedding
            outputs["drug_z"] = z_sampled
            outputs["kl_loss"] = kl_loss
            if drug_attention is not None:
                outputs["drug_attention"] = drug_attention
        else:
            drug_embedding = None
            z_sampled = None
            kl_loss = None
        
        # Encode target features
        if target_features is not None and len(target_features) > 0:
            target_embedding, target_attention = self._encode_target_features(target_features)
            outputs["target_embedding"] = target_embedding
            if target_attention is not None:
                outputs["target_attention"] = target_attention
        else:
            target_embedding = None
        
        # DTI prediction (for train/finetune phases)
        if self.phase in ["train", "finetune"]:
            if drug_embedding is None or target_embedding is None:
                raise ValueError("Both drug and target embeddings are required for DTI prediction")
            
            # Inter-branch aggregation
            combined_features = self.inter_branch_aggregator([drug_embedding, target_embedding])
            
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
        
        # Drug reconstruction (for pretrain_drug phase)
        if self.phase == "pretrain_drug" and drug_smiles is not None:
            # Convert SMILES to molecular graphs
            X_true, E_true, node_mask = self._smiles_to_graph(drug_smiles)
            X_true = X_true.to(z_sampled.device)
            E_true = E_true.to(z_sampled.device)
            node_mask = node_mask.to(z_sampled.device)
            
            # Apply noise for training
            noisy_data = self.diffusion_decoder.apply_noise(X_true, E_true, z_sampled, node_mask)
            
            # Compute extra features
            extra_data = self.diffusion_decoder.compute_extra_data(noisy_data)
            
            # Forward through diffusion decoder
            pred_X, pred_E = self.diffusion_decoder(noisy_data, extra_data, node_mask)
            
            # Compute reconstruction loss
            loss_dict = self.diffusion_decoder.train_loss(
                pred_X, pred_E, 
                X_true, E_true,
                node_mask
            )
            
            outputs.update({
                "reconstruction_loss": loss_dict["loss"],
                "X_loss": loss_dict["X_loss"],
                "E_loss": loss_dict["E_loss"],
                "pred_X": pred_X,
                "pred_E": pred_E,
                "true_X": X_true,
                "true_E": E_true,
                "node_mask": node_mask
            })
        
        return outputs
    
    def _get_features_from_batch(
        self, 
        batch: Dict[str, Any]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[List[str]]]:
        """Extract and prepare features from batch."""
        drug_feats = []
        target_feats = []
        drug_smiles = None
        
        if self.phase == "pretrain_drug":
            # For drug pretraining, extract drug features and SMILES
            for feat_name in self.hparams.drug_features:
                if feat_name in batch["features"]:
                    drug_feats.append(batch["features"][feat_name])
                else:
                    raise ValueError(f"Missing drug feature '{feat_name}' in batch")
            
            # Extract SMILES for reconstruction
            if "representations" in batch and "smiles" in batch["representations"]:
                drug_smiles = batch["representations"]["smiles"]
            elif "representations" in batch and "SMILES" in batch["representations"]:
                drug_smiles = batch["representations"]["SMILES"]
            else:
                raise ValueError("SMILES representations not found in batch")
            
            target_feats = []
    
        elif self.phase == "pretrain_target":
            # For target pretraining, extract target features
            for feat_name in self.hparams.target_features:
                if feat_name in batch["features"]:
                    target_feats.append(batch["features"][feat_name])
                else:
                    raise ValueError(f"Missing target feature '{feat_name}' in batch")
            drug_feats = []
            
        else:  # DTI batch (train/finetune)
            # Extract drug features
            for feat_name in self.hparams.drug_features:
                if feat_name in batch["drug"]["features"]:
                    drug_feats.append(batch["drug"]["features"][feat_name])
                else:
                    raise ValueError(f"Missing drug feature '{feat_name}' in batch")
            
            # Extract target features
            for feat_name in self.hparams.target_features:
                if feat_name in batch["target"]["features"]:
                    target_feats.append(batch["target"]["features"][feat_name])
                else:
                    raise ValueError(f"Missing target feature '{feat_name}' in batch")
            
            # Extract SMILES if available (for potential future use)
            if "representations" in batch.get("drug", {}) and "SMILES" in batch["drug"]["representations"]:
                drug_smiles = batch["drug"]["representations"]["SMILES"]
        
        return drug_feats, target_feats, drug_smiles
    
    def _compute_contrastive_loss(
        self, 
        embeddings: torch.Tensor, 
        fingerprints: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss using InfoNCE with Tanimoto similarity.
        
        Args:
            embeddings: Learned embeddings (batch_size, embedding_dim)
            fingerprints: Molecular fingerprints for similarity computation
            
        Returns:
            Contrastive loss (scalar)
        """
        # Use the appropriate InfoNCE head
        if self.phase == "pretrain_drug":
            loss = self.drug_infonce_head(embeddings, fingerprints)
        else:  # pretrain_target
            loss = self.target_infonce_head(embeddings, fingerprints)
        
        return loss
    
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
        total_loss += self.dti_loss_weights[0] * binary_loss
        loss_count += 1
        
        # Real-valued losses (with masking)
        real_losses = []
        
        # Phase-specific loss computation
        if self.phase == "finetune":
            # Single score prediction
            score_name = self.finetune_score  # e.g., "Y_pKd" or "Y_KIBA"
            pred_name = score_name.replace("Y_", "") + "_pred"  # e.g., "pKd_pred"
            
            if f"{score_name}_mask" in batch["y"]:
                mask = batch["y"][f"{score_name}_mask"]
                if mask.any():
                    pred = outputs[pred_name][mask]
                    target = batch["y"][score_name][mask]
                    score_loss = self.mse_loss(pred.squeeze(), target)
                    real_losses.append((score_name, score_loss))
                    
                    # Weight index: pKd=1, pKi=2, KIBA=3
                    weight_idx = {"Y_pKd": 1, "Y_pKi": 2, "Y_KIBA": 3}[score_name]
                    total_loss += self.dti_loss_weights[weight_idx] * score_loss
                    loss_count += 1
        else:
            # Multi-score prediction
            for i, (score_name, pred_name) in enumerate([
                ("Y_pKd", "pKd_pred"),
                ("Y_pKi", "pKi_pred"),
                ("Y_KIBA", "KIBA_pred")
            ], 1):
                if f"{score_name}_mask" in batch["y"]:
                    mask = batch["y"][f"{score_name}_mask"]
                    if mask.any():
                        pred = outputs[pred_name][mask]
                        target = batch["y"][score_name][mask]
                        score_loss = self.mse_loss(pred.squeeze(), target)
                        real_losses.append((score_name, score_loss))
                        total_loss += self.dti_loss_weights[i] * score_loss
                        loss_count += 1
        
        # Log individual losses
        self.log(f"{step_name}/binary_loss", binary_loss.item())
        for score_name, loss_val in real_losses:
            self.log(f"{step_name}/{score_name}_loss", loss_val.item())
        
        # Normalize by number of active losses
        if loss_count > 0:
            total_loss = total_loss / loss_count
        
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
        drug_features, target_features, drug_smiles = self._get_features_from_batch(batch)
        
        # Forward pass
        outputs = self(drug_features, target_features, drug_smiles)
        
        # Compute loss based on phase
        if self.phase == "pretrain_drug":
            # Drug branch losses: contrastive + KL + reconstruction
            total_loss = 0.0
            
            # Contrastive loss (using latent z)
            contrastive_loss = self._compute_contrastive_loss(
                outputs["drug_z"], 
                batch["features"]["FP-Morgan"]
            )
            total_loss += self.hparams.contrastive_weight * contrastive_loss
            self.log(f"{step_name}/contrastive_loss", contrastive_loss.item())
            
            # KL loss
            kl_loss = outputs["kl_loss"]
            total_loss += self.hparams.kl_weight * kl_loss
            self.log(f"{step_name}/kl_loss", kl_loss.item())
            
            # Reconstruction loss
            if "reconstruction_loss" in outputs:
                recon_loss = outputs["reconstruction_loss"]
                total_loss += self.hparams.reconstruction_weight * recon_loss
                self.log(f"{step_name}/reconstruction_loss", recon_loss.item())
                self.log(f"{step_name}/X_loss", outputs["X_loss"].item())
                self.log(f"{step_name}/E_loss", outputs["E_loss"].item())
            
            loss = total_loss
            
        elif self.phase == "pretrain_target":
            # Target branch contrastive loss
            contrastive_loss = self._compute_contrastive_loss(
                outputs["target_embedding"], 
                batch["features"]["FP-ESP"]
            )
            loss = self.hparams.contrastive_weight * contrastive_loss
            self.log(f"{step_name}/contrastive_loss", contrastive_loss.item())
            
        else:  # train or finetune
            # DTI prediction loss
            loss = self._compute_dti_loss(outputs, batch, step_name)
        
        return outputs, loss
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """Training step."""
        outputs, loss = self._common_step(batch, "train")
        
        # Update metrics based on phase
        if self.phase == "train" and self.train_metrics is not None:
            self._update_multi_score_metrics(outputs, batch, self.train_metrics)
        elif self.phase == "finetune" and self.train_metrics is not None:
            self._update_single_score_metrics(outputs, batch, self.train_metrics)
        
        # Logging
        self.log("train/loss", loss)
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """Validation step."""
        outputs, loss = self._common_step(batch, "val")
        
        # Update metrics based on phase
        if self.phase == "train" and self.val_metrics is not None:
            self._update_multi_score_metrics(outputs, batch, self.val_metrics)
        elif self.phase == "finetune" and self.val_metrics is not None:
            self._update_single_score_metrics(outputs, batch, self.val_metrics)
        
        # Logging
        self.log("val/loss", loss)
        
        return loss
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        """Test step - similar to validation."""
        outputs, loss = self._common_step(batch, "test")
        
        # Update metrics based on phase
        if self.phase == "train" and self.test_metrics is not None:
            self._update_multi_score_metrics(outputs, batch, self.test_metrics)
        elif self.phase == "finetune" and self.test_metrics is not None:
            self._update_single_score_metrics(outputs, batch, self.test_metrics)
            
            # Store predictions for correlation calculation
            valid_mask = batch["y"][f"{self.finetune_score}_mask"]
            if valid_mask.any():
                pred_name = self.finetune_score.replace("Y_", "") + "_pred"
                valid_preds = outputs[pred_name][valid_mask]
                valid_targets = batch["y"][self.finetune_score][valid_mask]
                self.test_predictions.append(valid_preds.cpu())
                self.test_targets.append(valid_targets.cpu())
        
        # Logging
        self.log("test/loss", loss)
        
        return loss
    
    def _update_multi_score_metrics(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, Any], 
        metrics: DTIMetricsCollection
    ):
        """Update metrics for multi-score prediction."""
        # Binary predictions
        binary_preds = outputs["binary_pred"]
        binary_targets = batch["y"]["Y"]
        
        # Real-valued predictions with masking
        real_preds = {}
        real_targets = {}
        
        for score_name, pred_name in [
            ("Y_pKd", "pKd_pred"),
            ("Y_pKi", "pKi_pred"),
            ("Y_KIBA", "KIBA_pred")
        ]:
            if f"{score_name}_mask" in batch["y"]:
                mask = batch["y"][f"{score_name}_mask"]
                if mask.any():
                    real_preds[pred_name] = outputs[pred_name][mask]
                    real_targets[score_name.replace("Y_", "")] = batch["y"][score_name][mask]
        
        # Update metrics
        metrics.update(
            binary_preds=binary_preds,
            binary_targets=binary_targets,
            real_preds=real_preds,
            real_targets=real_targets
        )
    
    def _update_single_score_metrics(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, Any], 
        metrics
    ):
        """Update metrics for single-score prediction."""
        # Get predictions and targets for the specific score
        score_name = self.finetune_score
        pred_name = score_name.replace("Y_", "") + "_pred"
        
        valid_mask = batch["y"][f"{score_name}_mask"]
        if valid_mask.any():
            valid_preds = outputs[pred_name][valid_mask]
            valid_targets = batch["y"][score_name][valid_mask]
            
            # Update metrics
            metrics.update(valid_preds, valid_targets)
    
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
    
    def on_test_epoch_end(self):
        """Compute and log test metrics."""
        if self.test_metrics is not None:
            test_metrics = self.test_metrics.compute()
            for name, value in test_metrics.items():
                self.log(name, value)
            
            # Additional manual correlation calculation for fine-tuning
            if self.phase == "finetune" and len(self.test_predictions) > 0:
                all_preds = torch.cat(self.test_predictions)
                all_targets = torch.cat(self.test_targets)
                
                # Final correlation
                pred_mean = all_preds.mean()
                target_mean = all_targets.mean()
                
                pred_centered = all_preds - pred_mean
                target_centered = all_targets - target_mean
                
                final_correlation = (pred_centered * target_centered).sum() / (
                    torch.sqrt((pred_centered ** 2).sum()) * 
                    torch.sqrt((target_centered ** 2).sum())
                )
                
                self.log("test/manual_correlation", final_correlation)
            
            # Clear stored predictions and reset metrics
            self.test_predictions.clear()
            self.test_targets.clear()
            self.test_metrics.reset()
