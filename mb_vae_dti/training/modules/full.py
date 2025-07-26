import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Literal
import logging

from .utils import *
from mb_vae_dti.training.models import (
    ResidualEncoder, TransformerEncoder,
    ConcatAggregator, AttentiveAggregator,
    CrossAttentionFusion, DTIHead, InfoNCEHead, KLVariationalHead, ReconstructionHead
)
from mb_vae_dti.training.data_containers import (
    BatchData, EmbeddingData, PredictionData, LossData
)

# TODO: we want to make a nice metrics collection for our graphs, just like we have DTI metrics collection
# Items: NLL (validation loss), MolValidity (simple RDKit validation), TanimotoSimilarity (for comparing conditioned G_hat to true G)
from mb_vae_dti.training.metrics.graph_metrics import *
from mb_vae_dti.training.diffusion.augmentation import Augmentation
from mb_vae_dti.training.diffusion.discrete_noise import PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
from mb_vae_dti.training.models.graph_transformer import GraphTransformer
from mb_vae_dti.training.diffusion.utils import * # TODO: perhaps improve structure of this one-and-all utils file

logger = logging.getLogger(__name__)


class FullDTIModel(AbstractDTIModel):
    """
    Full DTI model with variational drug branch and diffusion decoder.
    
    Architecture:
    - Drug branch: Multiple features → Encoders → Aggregation → VAE → z → InfoNCE/DTI/Decoder
    - Target branch: Multiple features → Encoders → Aggregation → InfoNCE/DTI  
    - Diffusion decoder: z → Graph transformer → Reconstructed molecular graph G_hat
    - DTI prediction: Drug/Target embeddings → Fusion → DTI head → Multiple DTI scores
    
    Training phases:
    - pretrain_drug: Drug branch pretraining with InfoNCE + KL + reconstruction losses
    - pretrain_target: Target branch pretraining with InfoNCE loss
    - train: General DTI training (multi-score prediction on combined dataset)
    - finetune: Fine-tuning on benchmark datasets (single-score prediction & graph reconstruction)
    """
    def __init__(
        self,

        phase: Literal["pretrain_drug", "pretrain_target", "train", "finetune"],
        finetune_score: Optional[Literal["Y_pKd", "Y_KIBA"]],

        learning_rate: float,
        weight_decay: float,
        scheduler: Optional[Literal["const", "step", "one_cycle", "cosine"]],

        weights: List[float], # accuracy, complexity, contrastive, reconstruction
        dti_weights: Optional[List[float]], # Y, pKd, pKi, KIBA weights
        diff_weights: Optional[List[int]],   # [X, E]
        contrastive_temp: Optional[float],

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
        activation = self.parse_activation(activation)
        self.attentive = aggregator_type == "attentive"

        self.weights = weights
        self.dti_weights = dti_weights
        self.diff_weights = diff_weights
        self.contrastive_temp = contrastive_temp
        
        # Check that the model is configured correctly
        if phase == "finetune":
            assert finetune_score is not None, "finetune_score must be specified for finetune phase"
            assert dti_weights is None, "dti_weights must be None for finetune phase"
            assert contrastive_temp is None, "contrastive_temp must be None for finetune phase"
        elif phase == "train":
            assert dti_weights is not None, "dti_weights must be specified for general DTI training"
            assert contrastive_temp is not None, "contrastive_temp must be specified for general DTI training"
        elif phase in ["pretrain_drug", "pretrain_target"]:
            assert contrastive_temp is not None, "contrastive_temp must be specified for pretrain phases"
        else:
            raise ValueError(f"Invalid phase: {phase}")
        
        if phase != "pretrain_target":
            assert diff_weights is not None, "diff_weights must be specified for instantiating the drug branch"
            assert diffusion_steps is not None, "diffusion_steps must be specified for instantiating the drug branch"
            assert num_samples_to_generate is not None, "num_samples_to_generate must be specified for instantiating the drug branch"
            assert graph_transformer_kwargs is not None, "graph_transformer_kwargs must be specified for instantiating the drug branch"
            assert dataset_infos is not None, "dataset_infos must be specified for instantiating the drug branch"
            assert graph_transformer_kwargs["output_dims"]["y"] == embedding_dim, "embedding_dim must match y_dim of graph_transformer_kwargs"
        
        logger.info(f"""Full DTI model with:
        - Drug branch: {drug_features.keys()} (dims: {drug_features.values()})
        - Target branch: {target_features.keys()} (dims: {target_features.values()})
        - Phase: {phase} (finetune_score: {finetune_score})
        - Aggregator type: {aggregator_type}""")

        encoder_type = ResidualEncoder if encoder_type == "resnet" else TransformerEncoder
        aggregator_type = ConcatAggregator if aggregator_type == "concat" else AttentiveAggregator


        # Drug branch
        if self.phase != "pretrain_target":
            self.drug_encoders = nn.ModuleDict()
            for feat_name in drug_features:
                self.drug_encoders[feat_name] = encoder_type(
                    input_dim=drug_features[feat_name],
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim if len(drug_features) > 1 else embedding_dim, # individual view encoders project to hidden_dim
                    n_layers=n_layers,
                    factor=factor,
                    activation=activation,
                    dropout=dropout,
                    bias=bias
                )
            if len(drug_features) > 1:
                self.drug_aggregator = aggregator_type(
                    input_dim=hidden_dim,
                    n_features=len(drug_features),
                    output_dim=embedding_dim, # but together they are aggregated to embedding_dim
                )
            else:
                self.drug_aggregator = torch.nn.Identity()  # No aggregation needed for single feature
        
            self.drug_contrastive_head = InfoNCEHead(
                input_dim=embedding_dim,
                proj_dim=hidden_dim // 2,
                dropout=dropout,
                bias=bias,
                activation=activation
            )

            self.drug_kl_head = KLVariationalHead(
                input_dim=embedding_dim,
                proj_dim=embedding_dim,
            )

            # DIFFUSION DECODER
            self.T = diffusion_steps
            self.visualization_tools = MolecularVisualization(dataset_infos["general"]["atom_types"])
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

            self.drug_decoder = GraphTransformer(
                n_layers=graph_transformer_kwargs['n_layers'],
                input_dims=graph_transformer_kwargs['input_dims'],
                output_dims=graph_transformer_kwargs['output_dims'],
                hidden_mlp_dims=graph_transformer_kwargs['hidden_mlp_dims'],
                hidden_dims=graph_transformer_kwargs['hidden_dims'],
                act_fn_in=nn.ReLU(),
                act_fn_out=nn.ReLU()
            )

        # Target branch
        if self.phase != "pretrain_drug":
            self.target_encoders = nn.ModuleDict()
            for feat_name in target_features:
                self.target_encoders[feat_name] = encoder_type(
                    input_dim=target_features[feat_name],
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim if len(target_features) > 1 else embedding_dim,
                    n_layers=n_layers,
                    factor=factor,
                    activation=activation,
                    dropout=dropout,
                    bias=bias
                )
            if len(target_features) > 1:
                self.target_aggregator = aggregator_type(
                    input_dim=hidden_dim,
                    n_features=len(target_features),
                    output_dim=embedding_dim,
                )
            else:
                self.target_aggregator = torch.nn.Identity()  # No aggregation needed for single feature
            
            self.target_contrastive_head = InfoNCEHead(
                input_dim=embedding_dim,
                proj_dim=hidden_dim // 2,
                dropout=dropout,
                bias=bias,
                activation=activation
            )

        # DTI prediction
        if phase in ["train", "finetune"]:
            # Fusion module for aggregating drug and target embeddings
            self.fusion = CrossAttentionFusion(
                input_dim=embedding_dim,
                output_dim=hidden_dim,
                n_layers=n_layers,
                factor=factor,
                dropout=dropout,
                bias=bias,
                activation=activation
            )
            
            # DTI prediction head
            self.dti_head = DTIHead(
                input_dim=hidden_dim,
                proj_dim=hidden_dim // 2,
                dropout=dropout,
                bias=bias,
                activation=activation
            )
        
        # Setup metrics using AbstractDTIModel's method
        self.setup_metrics(phase=phase, finetune_score=finetune_score)


    def _create_batch_data( 
            self, 
            batch: Dict[str, Any]
        ) -> BatchData:
        """Create structured batch data using basic utilities & custom logic."""
        batch_data = BatchData(raw_batch=batch)

        # One will be None in pretrain_target/pretrain_drug case
        batch_data.drug_features, batch_data.target_features = self._get_features_from_batch(batch)

        # We want fingerprints for contrastive loss (one will be None in pretrain_target/pretrain_drug case)
        batch_data.drug_fp, batch_data.target_fp = self._get_fingerprints_from_batch(batch)

        # Only interested in targets during DTI train/finetune phase
        if self.phase not in ["pretrain_drug", "pretrain_target"]:
            batch_data.dti_targets, batch_data.dti_masks = self._get_targets_masks_from_batch(batch)
        
        # Only interested in graph data when decoder is in use
        if self.phase != "pretrain_target":
            batch_data.graph_data = self._extract_graph_data(batch)
        
        return batch_data

# we can now fetch all data we need:
# drug_features, target_features, drug_fp, target_fp
# dti_targets, dti_masks
# graph_data

# Next steps:
# - process graph with foward diffusion process, populating the GraphData object's G_t & noise_params
# - encode drug and target features (incl info_nce & kl for drug) -> complexity & contrastive
# - DTI prediction -> accuracy loss
# - use the drug_embedding to populate the graph_data.y (conditional signal)
# - augment features (populating GraphData G_augmented)
# - pass to GraphTransformer decoder (populating GraphData.G_hat) -> reconstruction loss

# Later steps:
# - graph validation metrics (NLL)
# - sampling process for conditional generation (MolValidity, TanimotoSimilarity)

############################################################################
############################################################################
############################################################################

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
        drug_features: Optional[List[torch.Tensor]], 
        target_features: Optional[List[torch.Tensor]]
    ) -> Tuple[EmbeddingData, PredictionData]:
        """
        Forward pass through the full model (encoder portion only).
        
        Args:
            drug_features: Drug feature tensors [batch_size, drug_input_dim] (optional in case of pretrain_target)
            target_features: Target feature tensors [batch_size, target_input_dim] (optional in case of pretrain_drug)
            
        Returns:
            embedding_data: Structured drug & target embeddings (and attention weights if using attentive aggregator)
            prediction_data: Structured predictions with multi-score DTI prediction or single score for finetune
        """
        embedding_data = EmbeddingData()

        # Encode & aggregate drug and/or target features
        if self.phase != "pretrain_target":
            drug_embeddings = [
                self.drug_encoders[feat_name](feat)
                for feat_name, feat in zip(self.hparams.drug_features, drug_features)
            ]
            if self.attentive:
                drug_embedding_raw, embedding_data.drug_attention = self.drug_aggregator(drug_embeddings)
            else:
                drug_embedding_raw = self.drug_aggregator(drug_embeddings)
            
            # Apply variational encoding (KL head)
            embedding_data.drug_embedding, embedding_data.drug_mu, embedding_data.drug_logvar = self.drug_kl_head(drug_embedding_raw)

        if self.phase != "pretrain_drug":
            target_embeddings = [
                self.target_encoders[feat_name](feat)
                for feat_name, feat in zip(self.hparams.target_features, target_features)
            ]
            if self.attentive:
                embedding_data.target_embedding, embedding_data.target_attention = self.target_aggregator(target_embeddings)
            else:
                embedding_data.target_embedding = self.target_aggregator(target_embeddings)

        # Fuse drug and target embeddings & get DTI predictions
        if self.phase not in ["pretrain_drug", "pretrain_target"]:
            embedding_data.fused_embedding, _ = self.fusion(
                embedding_data.drug_embedding, 
                embedding_data.target_embedding
            )
            # Get predictions (extract only single score for finetune phase)
            dti_scores = self.dti_head(embedding_data.fused_embedding)
            dti_scores = dti_scores[self.finetune_score] if self.phase == "finetune" else dti_scores
            prediction_data = PredictionData(
                dti_scores=dti_scores
            )
        else:
            prediction_data = PredictionData()

        return embedding_data, prediction_data
    
    def _decode_drug_graph(
        self, 
        embedding_data: EmbeddingData, 
        batch_data: BatchData
    ) -> PlaceHolder:
        """
        Apply diffusion decoder to reconstruct drug graphs.
        
        Args:
            embedding_data: Contains drug_embedding to use as conditional signal
            batch_data: Contains graph_data with noisy/augmented graph
            
        Returns:
            G_hat: Reconstructed graph predictions
        """
        if self.phase == "pretrain_target":
            return None
            
        # Use drug embedding as conditional signal y
        G_t_conditioned = PlaceHolder(
            X=batch_data.graph_data.X_augmented,
            E=batch_data.graph_data.E_augmented, 
            y=torch.cat([embedding_data.drug_embedding, batch_data.graph_data.y_augmented], dim=1)
        )
        
        # Apply graph transformer decoder
        G_hat = self.drug_decoder(
            G_t_conditioned.X, 
            G_t_conditioned.E, 
            G_t_conditioned.y, 
            batch_data.graph_data.node_mask
        )
        
        return G_hat

    def _common_step(
        self, 
        batch: Dict[str, Any]
    ) -> Tuple[BatchData, EmbeddingData, PredictionData, LossData]:
        """
        Common step logic shared across train/val/test.
        
        Returns:
            batch_data: Structured batch data incl. targets, masks, and graph data
            embedding_data: Structured embeddings (incl. variational components)
            prediction_data: Structured predictions (DTI + graph reconstruction)
            loss_data: All loss components (accuracy, complexity, contrastive, reconstruction)
        """
        # Create structured batch data (includes graph extraction if needed)
        batch_data = self._create_batch_data(batch)
        
        # Apply forward diffusion process (add noise to clean graphs)
        if self.phase != "pretrain_target" and batch_data.graph_data is not None:
            # Extract clean graph and apply noise
            G_clean = PlaceHolder(X=batch_data.graph_data.X, E=batch_data.graph_data.E)
            G_t, noise_params = self.apply_noise(G_clean, batch_data.graph_data.node_mask)
            
            # Store noisy graph data
            batch_data.graph_data.X_t = G_t.X
            batch_data.graph_data.E_t = G_t.E  
            batch_data.graph_data.noise_params = noise_params
            
            # Augment noisy graph with extra features
            G_t_augmented = self.augment_graph(G_t, noise_params, batch_data.graph_data.node_mask)
            batch_data.graph_data.X_augmented = G_t_augmented.X
            batch_data.graph_data.E_augmented = G_t_augmented.E
            batch_data.graph_data.y_augmented = G_t_augmented.y
        
        # Forward pass (encoders + DTI prediction)
        embedding_data, prediction_data = self.forward(
            batch_data.drug_features, 
            batch_data.target_features
        )

        # Graph reconstruction using diffusion decoder
        if self.phase != "pretrain_target" and batch_data.graph_data is not None:
            G_hat = self._decode_drug_graph(embedding_data, batch_data)
            prediction_data.graph_reconstruction = G_hat

        # Compute all loss components
        loss_data = LossData()

        # 1. Complexity loss (KL divergence for VAE)
        if self.phase != "pretrain_target":
            loss_data.complexity = self.drug_kl_head.kl_divergence(
                embedding_data.drug_mu, 
                embedding_data.drug_logvar
            )
        else:
            loss_data.complexity = torch.tensor(0.0, device=self.device)

        # 2. Contrastive losses
        drug_contrastive_loss = self.drug_contrastive_head(
            x=embedding_data.drug_embedding,
            fingerprints=batch_data.drug_fp,
            temperature=self.contrastive_temp
        ) if self.phase not in ["pretrain_target", "finetune"] else torch.tensor(0.0, device=self.device)

        target_contrastive_loss = self.target_contrastive_head(
            x=embedding_data.target_embedding,
            fingerprints=batch_data.target_fp,
            temperature=self.contrastive_temp
        ) if self.phase not in ["pretrain_drug", "finetune"] else torch.tensor(0.0, device=self.device)
        
        loss_data.contrastive = drug_contrastive_loss + target_contrastive_loss
        loss_data.components = {
            "drug_contrastive": drug_contrastive_loss,
            "target_contrastive": target_contrastive_loss
        }

        # 3. Reconstruction loss (diffusion decoder)
        if self.phase != "pretrain_target" and batch_data.graph_data is not None:
            # Create reconstruction head if not exists
            if not hasattr(self, 'reconstruction_head'):
                from mb_vae_dti.training.models.heads import ReconstructionHead
                self.reconstruction_head = ReconstructionHead(self.diff_weights)
            
            G_true = PlaceHolder(X=batch_data.graph_data.X_t, E=batch_data.graph_data.E_t)
            loss_data.reconstruction = self.reconstruction_head(
                pred=prediction_data.graph_reconstruction,
                true=G_true
            )
        else:
            loss_data.reconstruction = torch.tensor(0.0, device=self.device)

        # 4. Accuracy loss (DTI prediction)
        if self.phase not in ["pretrain_drug", "pretrain_target"]:
            if self.phase == "finetune":
                # Single-score accuracy loss with masking
                if batch_data.dti_masks.any():
                    masked_preds = prediction_data.dti_scores[batch_data.dti_masks]
                    masked_targets = batch_data.dti_targets[batch_data.dti_masks]
                    loss_data.accuracy = F.mse_loss(masked_preds, masked_targets)
                else:
                    loss_data.accuracy = torch.tensor(0.0, device=self.device)
            else:  # phase == "train"
                # Multi-score accuracy loss
                components = {
                    "Y": F.binary_cross_entropy_with_logits(
                        prediction_data.dti_scores["Y"],
                        batch_data.dti_targets["Y"]
                    )
                }
                
                # Continuous target losses (when data available)
                for score_name in ["Y_pKd", "Y_pKi", "Y_KIBA"]:
                    valid_mask = batch_data.dti_masks[score_name]
                    if valid_mask.any():
                        masked_preds = prediction_data.dti_scores[score_name][valid_mask]
                        masked_targets = batch_data.dti_targets[score_name][valid_mask]
                        components[score_name] = F.mse_loss(masked_preds, masked_targets)
                    else:
                        components[score_name] = torch.tensor(0.0, device=self.device)
                
                # Compute weighted total accuracy loss
                loss_data.accuracy = sum(
                    self.dti_weights[i] * loss 
                    for i, (score_name, loss) in enumerate(components.items())
                )
                loss_data.components.update(components)
        else:
            loss_data.accuracy = torch.tensor(0.0, device=self.device)
        
        return batch_data, embedding_data, prediction_data, loss_data

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        batch_data, embedding_data, prediction_data, loss_data = self._common_step(batch)
        
        # Update DTI metrics based on phase
        if self.phase not in ["pretrain_drug", "pretrain_target"]:
            if prediction_data.dti_scores is not None:
                if self.phase == "finetune":
                    # For finetune, only update with valid (masked) samples
                    if batch_data.dti_masks.any():
                        self.train_metrics.update(
                            prediction_data.dti_scores[batch_data.dti_masks],
                            batch_data.dti_targets[batch_data.dti_masks]
                        )
                else:
                    # For train phase, metrics handle masking internally
                    self.train_metrics.update(
                        prediction_data.dti_scores,
                        batch_data.dti_targets
                    )
            
            # Log individual loss components
            for name, value in loss_data.components.items():
                self.log(f"train/loss_{name}", value)
        
        # TODO: Add diffusion metrics here for molecular validity, etc.
        # Update diffusion metrics if available
        # if self.train_diffusion_metrics is not None:
        #     self.train_diffusion_metrics.update(
        #         prediction_data.graph_reconstruction, 
        #         batch_data.graph_data
        #     )

        # Compute total loss and log individual components
        loss = loss_data.compute_loss(self.weights)
        
        self.log("train/loss_accuracy", loss_data.accuracy)
        self.log("train/loss_complexity", loss_data.complexity)
        self.log("train/loss_contrastive", loss_data.contrastive)
        self.log("train/loss_reconstruction", loss_data.reconstruction)
        self.log("train/loss", loss)
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        batch_data, embedding_data, prediction_data, loss_data = self._common_step(batch)
        
        # Update DTI metrics based on phase
        if self.phase not in ["pretrain_drug", "pretrain_target"]:
            if prediction_data.dti_scores is not None:
                if self.phase == "finetune":
                    # For finetune, only update with valid (masked) samples
                    if batch_data.dti_masks.any():
                        self.val_metrics.update(
                            prediction_data.dti_scores[batch_data.dti_masks],
                            batch_data.dti_targets[batch_data.dti_masks]
                        )
                else:
                    # For train phase, metrics handle masking internally
                    self.val_metrics.update(
                        prediction_data.dti_scores,
                        batch_data.dti_targets
                    )
            
            # Log individual loss components
            for name, value in loss_data.components.items():
                self.log(f"val/loss_{name}", value)
        
        # TODO: Add diffusion metrics here for NLL and molecular validity
        # Update diffusion metrics if available
        # if self.val_diffusion_metrics is not None:
        #     self.val_diffusion_metrics.update(
        #         prediction_data.graph_reconstruction, 
        #         batch_data.graph_data
        #     )

        # Compute total loss and log individual components
        loss = loss_data.compute_loss(self.weights)
        
        self.log("val/loss_accuracy", loss_data.accuracy)
        self.log("val/loss_complexity", loss_data.complexity)
        self.log("val/loss_contrastive", loss_data.contrastive)
        self.log("val/loss_reconstruction", loss_data.reconstruction)
        self.log("val/loss", loss)
        
        return loss
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        batch_data, embedding_data, prediction_data, loss_data = self._common_step(batch)
        
        # Update DTI metrics based on phase
        if self.phase not in ["pretrain_drug", "pretrain_target"]:
            if prediction_data.dti_scores is not None:
                if self.phase == "finetune":
                    # For finetune, only update with valid (masked) samples
                    if batch_data.dti_masks.any():
                        self.test_metrics.update(
                            prediction_data.dti_scores[batch_data.dti_masks],
                            batch_data.dti_targets[batch_data.dti_masks]
                        )
                else:
                    # For train phase, metrics handle masking internally
                    self.test_metrics.update(
                        prediction_data.dti_scores,
                        batch_data.dti_targets
                    )
            
            # Log individual loss components
            for name, value in loss_data.components.items():
                self.log(f"test/loss_{name}", value)
        
        # TODO: Add diffusion metrics here for NLL and molecular validity
        # Update diffusion metrics if available
        # if self.test_diffusion_metrics is not None:
        #     self.test_diffusion_metrics.update(
        #         prediction_data.graph_reconstruction, 
        #         batch_data.graph_data
        #     )

        # Compute total loss and log individual components
        loss = loss_data.compute_loss(self.weights)
        
        self.log("test/loss_accuracy", loss_data.accuracy)
        self.log("test/loss_complexity", loss_data.complexity)
        self.log("test/loss_contrastive", loss_data.contrastive)
        self.log("test/loss_reconstruction", loss_data.reconstruction)
        self.log("test/loss", loss)
        
        return loss