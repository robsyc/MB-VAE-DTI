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

from mb_vae_dti.training.metrics.graph_metrics import *
from mb_vae_dti.training.diffusion.augmentation import ExtraFeatures, ExtraMolecularFeatures
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
            self.sample_every_val = 5
            self.val_samples_per_embedding = 3 # TODO: these are new!
            self.test_samples_per_embedding = 1
            self.reconstruction_head = ReconstructionHead(diff_weights)
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
            self.graph_structure_features = ExtraFeatures(
                max_n_nodes=dataset_infos["general"]["max_n_nodes"]
            )
            self.molecular_features = ExtraMolecularFeatures(
                valencies=dataset_infos["general"]["atom_valencies"],
                max_weight=dataset_infos["general"]["max_weight"],
                atom_weights=dataset_infos["general"]["atom_weights"]
            )
            self.Xdim_output = graph_transformer_kwargs['output_dims']['X']
            self.Edim_output = graph_transformer_kwargs['output_dims']['E']
            # self.ydim_output = graph_transformer_kwargs['output_dims']['y']

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

        # Diffusion specific metrics
        if phase != "pretrain_target":
            # val_nll = ... # collection of NLL components & total NLL (log_pN, kl_prior, loss_all_t, loss_term_0)
            # test_nll = ...
            # train_mol_metrics = ... # cross-entropy terms over atom/bond types (gives insight into the actual train loss (which is just sum of these))
            # val_mol_metrics = ... # collection of molecular metrics (validity, tanimoto, accuracy)
            # test_mol_metrics = ...
            pass
        else:
            self.val_nll = None
            self.test_nll = None
            self.train_mol_metrics = None
            self.val_mol_metrics = None
            self.test_mol_metrics = None


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
            batch_data.smiles = self._get_smiles_from_batch(batch)
        
        return batch_data

    # Same forward pass as in multi_hybrid.py
    def forward(
        self, 
        drug_features: Optional[List[torch.Tensor]], 
        target_features: Optional[List[torch.Tensor]]
    ) -> Tuple[EmbeddingData, PredictionData]:
        """
        Forward pass through the multi-modal two-tower model.
        
        Args:
            drug_features: Drug feature tensors [batch_size, drug_input_dim] (optional in case of pretrain_target)
            target_features: Target feature tensors [batch_size, target_input_dim] (optional in case of pretrain_drug)
            
        Returns:
            embedding_data: Structured drug & target embeddings (and attention weights if using attentive aggregator)
            prediction_data: Structured predictions with single-score dot-product prediction
        """
        embedding_data = EmbeddingData()

        # Encode & aggregate drug and/or target features
        if self.phase != "pretrain_target":
            drug_embeddings = [
                self.drug_encoders[feat_name](feat)
                for feat_name, feat in zip(self.hparams.drug_features, drug_features)
            ]
            if self.attentive:
                embedding_data.drug_embedding, embedding_data.drug_attention = self.drug_aggregator(drug_embeddings)
            else:
                embedding_data.drug_embedding = self.drug_aggregator(drug_embeddings)
            
            # Variational head for drug embedding
            embedding_data.drug_embedding, \
            embedding_data.drug_mu, \
            embedding_data.drug_logvar = self.drug_kl_head(embedding_data.drug_embedding)

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
            prediction_data = PredictionData(
                dti_scores=dti_scores[self.finetune_score] if self.phase == "finetune" else dti_scores
            )
        else:
            prediction_data = PredictionData()

        return embedding_data, prediction_data


    def apply_noise(self, X, E, node_mask):
        """
        Apply noise to the drug graph
        Args:
            X: Node features
            E: Edge features
            node_mask: Node mask
        Returns:
            X_t: Noisy node features
            E_t: Noisy edge features
            noise_params: Noise parameters sampled from the noise schedule
        """
        # Sample timestep t (uniformly from [0, T])
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(
            lowest_t, self.T + 1, 
            size=(X.size(0), 1), 
            device=X.device
        ).float()
        s_int = t_int - 1

        # Normalize timesteps to [0, 1] for noise scheduler
        t_float = t_int / self.T
        s_float = s_int / self.T

        # Get noise schedule parameters
        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                     # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)  # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)  # (bs, 1)

        # Get transition matrices Q_t_bar for forward diffusion
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Forward diffusion: Compute transition probabilities
        probX = X @ Qtb.X               # (bs, n, dx_out)    - node transition probabilities p(X_t | X_0)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out) - edge transition probabilities p(E_t | E_0)

        # Sample discrete features from the transition probabilities
        sampled_t = sample_discrete_features(
            probX=probX, probE=probE, node_mask=node_mask)
        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)
        
        G_t = PlaceHolder(X=X_t, E=E_t).type_as(X_t).mask(node_mask)

        noise_params = {
            "t_int": t_int,
            "t": t_float,
            "beta_t": beta_t,
            "alpha_s_bar": alpha_s_bar,
            "alpha_t_bar": alpha_t_bar,
        }
        return G_t.X, G_t.E, noise_params


    def get_extra_features(self, X_t, E_t, noise_params, node_mask):
        """
        At every training step (after adding noise & sampling discrete features)
        extra features are computed (graph-structural features & molecular features)
        and appended to the graph transformer input.
        + timestep info t is added to graph-level features y to inform the model about the noise level
        # X gets 8 extra features, E none and y gets 13

        Args:
            X_t: Noisy node features
            E_t: Noisy edge features
            noise_params: Noise parameters sampled from the noise schedule
            node_mask: Node mask

        Returns:
            X_extra: Extra node features for transformer input (graph structural & molecular)
            y_extra: Extra global features (timestep info, etc.)
        """
        X_graph_feats, y_graph_feats = self.graph_structure_features(E_t, node_mask)
        X_mol_feats, y_mol_feats = self.molecular_features(X_t, E_t)

        t = noise_params['t'] # normalized timestep

        return (
            torch.cat((X_graph_feats, X_mol_feats), dim=2),
            torch.cat((y_graph_feats, y_mol_feats, t), dim=1)
        )
    

    def denoise_drug_graph(
        self, 
        embedding_data: EmbeddingData, 
        batch_data: BatchData
    ) -> PlaceHolder:
        """
        Apply diffusion decoder to reconstruct drug graphs.
        
        Args:
            embedding_data: Contains drug_embedding to use as conditional signal
            batch_data: Contains graph_data with noisy graph and extra features
            
        Returns:
            G_hat: Reconstructed graph predictions (pre-masked PlaceHolder object)
        """
        if self.phase == "pretrain_target":
            return None
        
        # Apply graph transformer decoder
        G_hat = self.drug_decoder(
            X=torch.cat([batch_data.graph_data.X_t, batch_data.graph_data.X_extra], dim=2).float(),
            E=batch_data.graph_data.E_t.float(), # no extra edge features
            y=torch.cat([embedding_data.drug_embedding, batch_data.graph_data.y_extra], dim=1).float(), 
            node_mask=batch_data.graph_data.node_mask
        )
        
        return G_hat # pre-masked PlaceHolder object


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
            # Apply noise to clean graph G -> G_t (X_t, E_t)
            batch_data.graph_data.X_t, \
            batch_data.graph_data.E_t, \
            batch_data.graph_data.noise_params = self.apply_noise(
                batch_data.graph_data.X, 
                batch_data.graph_data.E, 
                batch_data.graph_data.node_mask
            )
            
            # Augment noisy graph with extra features
            batch_data.graph_data.X_extra, \
            batch_data.graph_data.y_extra = self.get_extra_features(
                batch_data.graph_data.X_t, 
                batch_data.graph_data.E_t, 
                batch_data.graph_data.noise_params, 
                batch_data.graph_data.node_mask
            )


        # Forward pass (encoders + DTI prediction)
        embedding_data, prediction_data = self.forward(
            batch_data.drug_features, 
            batch_data.target_features
        )

        # Graph reconstruction using diffusion decoder
        if self.phase != "pretrain_target" and batch_data.graph_data is not None:
            prediction_data.graph_reconstruction = self.denoise_drug_graph(
                embedding_data, 
                batch_data
            )

        # Compute all loss components
        loss_data = LossData()

        # 1. Complexity loss (KL divergence for VAE) - only for the drug branch
        if self.phase != "pretrain_target":
            loss_data.complexity = self.drug_kl_head.kl_divergence(
                embedding_data.drug_mu, 
                embedding_data.drug_logvar
            )
        else:
            loss_data.complexity = torch.tensor(0.0, device=self.device)

        # 2. Contrastive losses - skipped when finetuning
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
            loss_data.reconstruction = self.reconstruction_head(
                masked_pred_X=prediction_data.graph_reconstruction.X,
                masked_pred_E=prediction_data.graph_reconstruction.E,
                true_X=batch_data.graph_data.X,
                true_E=batch_data.graph_data.E
            )
        else:
            loss_data.reconstruction = torch.tensor(0.0, device=self.device)

        # 4. Accuracy loss (DTI prediction)
        if self.phase not in ["pretrain_drug", "pretrain_target"]:
            if self.phase == "finetune":
                # mask
                prediction_data.dti_scores = prediction_data.dti_scores[batch_data.dti_masks]
                batch_data.dti_targets = batch_data.dti_targets[batch_data.dti_masks]
                if batch_data.dti_masks.any():
                    # single score accuracy loss
                    loss_data.accuracy = F.mse_loss(
                        prediction_data.dti_scores,
                        batch_data.dti_targets
                    )
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
                    prediction_data.dti_scores[score_name] = prediction_data.dti_scores[score_name][valid_mask]
                    batch_data.dti_targets[score_name] = batch_data.dti_targets[score_name][valid_mask]
                    components[score_name] = F.mse_loss(
                        prediction_data.dti_scores[score_name],
                        batch_data.dti_targets[score_name]
                    ) if valid_mask.any() else torch.tensor(0.0, device=self.device)
                
                # Compute weighted total accuracy loss
                loss_data.accuracy = sum(
                    self.dti_weights[i] * loss 
                    for i, (score_name, loss) in enumerate(components.items())
                )
                loss_data.components.update(components)
        
        return batch_data, embedding_data, prediction_data, loss_data


    def kl_prior(self, batch_data: BatchData) -> torch.Tensor:
        """
        Computes the KL between q(z_T | x) and the prior p(z_T) = limit distribution.
        Adapted from DiGress/DiffMS for our data structure.
        """
        if self.phase == "pretrain_target" or batch_data.graph_data is None:
            return torch.tensor(0.0, device=self.device)
            
        X = batch_data.graph_data.X
        E = batch_data.graph_data.E
        node_mask = batch_data.graph_data.node_mask
        
        # Compute the last alpha value, alpha_T
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)

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
        
        return sum_except_batch(kl_distance_X) + sum_except_batch(kl_distance_E)


    def compute_Lt(
        self, 
        batch_data: BatchData, 
        prediction_data: PredictionData,
        test: bool = False
    ) -> torch.Tensor:
        """
        Compute diffusion loss term Lt (KL between true and predicted posterior).
        Adapted from DiGress/DiffMS for our data structure.
        """
        if self.phase == "pretrain_target" or batch_data.graph_data is None:
            return torch.tensor(0.0, device=self.device)
            
        X = batch_data.graph_data.X
        E = batch_data.graph_data.E
        node_mask = batch_data.graph_data.node_mask
        noise_params = batch_data.graph_data.noise_params
        pred = prediction_data.graph_reconstruction
        
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1) if pred.y is not None else None

        Qtb = self.transition_model.get_Qt_bar(noise_params['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noise_params['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noise_params['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        y = torch.zeros(bs, 0).to(X.device)  # Empty y for compatibility
        
        prob_true = posterior_distributions(
            X=X, E=E, y=y, 
            X_t=batch_data.graph_data.X_t, 
            E_t=batch_data.graph_data.E_t,
            y_t=y, 
            Qt=Qt, Qsb=Qsb, Qtb=Qtb
        )
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        
        prob_pred = posterior_distributions(
            X=pred_probs_X, E=pred_probs_E, y=y,
            X_t=batch_data.graph_data.X_t, 
            E_t=batch_data.graph_data.E_t,
            y_t=y, 
            Qt=Qt, Qsb=Qsb, Qtb=Qtb
        )
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred_X, prob_pred_E = mask_distributions(
            true_X=prob_true.X,
            true_E=prob_true.E,
            pred_X=prob_pred.X,
            pred_E=prob_pred.E,
            node_mask=node_mask
        )
        
        # Use metrics from the model for consistency
        # TODO: integrate this better into the loss components?
        # may again want to log these individually & return total logp?
        if test:
            # We'll need to add these metrics - for now use a simple KL calculation
            kl_x = F.kl_div(torch.log(prob_pred_X + 1e-8), prob_true_X, reduction='sum')
            kl_e = F.kl_div(torch.log(prob_pred_E + 1e-8), prob_true_E, reduction='sum')
        else:
            kl_x = F.kl_div(torch.log(prob_pred_X + 1e-8), prob_true_X, reduction='sum')
            kl_e = F.kl_div(torch.log(prob_pred_E + 1e-8), prob_true_E, reduction='sum')
            
        return self.T * (kl_x + kl_e)


    def reconstruction_logp(
        self, 
        batch_data: BatchData, 
        embedding_data: EmbeddingData
    ) -> torch.Tensor:
        """
        Compute reconstruction probability at t=0.
        Adapted from DiGress/DiffMS for our data structure.
        """
        if self.phase == "pretrain_target" or batch_data.graph_data is None:
            return torch.tensor(0.0, device=self.device)
            
        X = batch_data.graph_data.X
        E = batch_data.graph_data.E
        node_mask = batch_data.graph_data.node_mask
        t = batch_data.graph_data.noise_params['t']
        
        # Compute noise values for t = 0
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)
        
        sampled0 = sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        # Create temporary graph data for prediction at t=0
        temp_graph_data = GraphData(
            X_t=X0,
            E_t=E0,
            node_mask=node_mask,
            noise_params={'t': t_zeros}
        )
        
        # Get extra features for t=0
        temp_graph_data.X_extra, temp_graph_data.y_extra = self.get_extra_features(
            X0, E0, {'t': t_zeros}, node_mask)

        # Create temporary batch data
        temp_batch_data = BatchData(
            raw_batch=batch_data.raw_batch,
            graph_data=temp_graph_data
        )

        # Predictions at t=0
        pred0 = self.denoise_drug_graph(embedding_data, temp_batch_data)

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        # proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        # Compute log probabilities
        X_logp = (X * probX0.log()).sum()
        E_logp = (E * probE0.log()).sum()

        # TODO: log these individually & return total logp?
        
        return X_logp + E_logp


    def compute_val_loss(
        self, 
        batch_data: BatchData, 
        embedding_data: EmbeddingData, 
        prediction_data: PredictionData,
        test: bool = False
    ) -> torch.Tensor:
        """
        Computes an estimator for the variational lower bound (NLL).
        Adapted from DiGress/DiffMS for our data structure.
        
        NLL = -log_pN + kl_prior + loss_all_t - loss_term_0
        """
        if self.phase == "pretrain_target" or batch_data.graph_data is None:
            return torch.tensor(0.0, device=self.device)
            
        node_mask = batch_data.graph_data.node_mask
        
        # 1. Log probability of number of nodes (we want to penalize too few or too many nodes)
        N = node_mask.sum(1).long()
        log_pN = self.nodes_dist.log_prob(N)

        # 2. KL between q(z_T | x) and p(z_T) = limit distribution
        kl_prior = self.kl_prior(batch_data)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(batch_data, prediction_data, test)

        # 4. Reconstruction loss at t=0
        loss_term_0 = self.reconstruction_logp(batch_data, embedding_data)

        # Combine terms
        nlls = -log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Return average NLL over batch
        nll = nlls.mean()

        # TODO: we want to log the NLL components here & return total nll
        
        return nll


    @torch.no_grad()
    def sample_batch(
        self, 
        drug_embeddings: torch.Tensor, 
        num_nodes: torch.Tensor = None,
        num_samples_per_embedding: int = 1
    ) -> list:
        """
        Sample molecules by iteratively denoising from limit distribution, conditioned on drug embeddings.
        
        Args:
            drug_embeddings: Drug embeddings to condition on [batch_size, embedding_dim]
            num_nodes: Number of nodes for each molecule [batch_size] or None for sampling from distribution
            num_samples_per_embedding: Number of molecular samples to generate per drug embedding
            
        Returns:
            List of RDKit molecule objects (or None for invalid molecules)
        """
        if self.phase == "pretrain_target":
            return []
            
        batch_size = drug_embeddings.size(0)
        
        # Sample number of nodes if not provided
        if num_nodes is None:
            n_nodes = self.nodes_dist.sample_n(batch_size, device=self.device)
        else:
            n_nodes = num_nodes.to(self.device)
        
        # Repeat drug embeddings and node counts for multiple samples per embedding
        if num_samples_per_embedding > 1:
            drug_embeddings = drug_embeddings.repeat_interleave(num_samples_per_embedding, dim=0)
            n_nodes = n_nodes.repeat_interleave(num_samples_per_embedding)
            batch_size = batch_size * num_samples_per_embedding
        
        n_max = torch.max(n_nodes).item()
        
        # Build node masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        
        # Sample initial noise from limit distribution
        z_T = sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E = z_T.X, z_T.E # we don't use sampled y but drug_embeddings
        
        assert (E == torch.transpose(E, 1, 2)).all()
        
        # Iteratively sample p(z_s | z_t) for t = T, T-1, ..., 1 with s = t - 1
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T
            
            # Sample z_s given z_t
            sampled_s = self.sample_p_zs_given_zt(
                s_norm, t_norm, X, E, drug_embeddings, node_mask
            )
            X, E = sampled_s.X, sampled_s.E

            # If we do sampled_s.mask(node_mask, collapse=True) here,
            # we can save the intermediate denoised sampled
            # to create a cool gif ?!
        
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E = sampled_s.X, sampled_s.E
        
        # Convert to RDKit molecules
        molecules = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            
            try:
                mol = self.visualization_tools.mol_from_graphs(atom_types, edge_types)
                molecules.append(mol)
            except Exception:
                # If molecule conversion fails, append None
                molecules.append(None)
        
        return molecules


    def sample_p_zs_given_zt(
        self,
        s: torch.Tensor,
        t: torch.Tensor, 
        X_t: torch.Tensor,
        E_t: torch.Tensor,
        drug_embeddings: torch.Tensor,
        node_mask: torch.Tensor
    ) -> tuple:
        """
        Sample from p(z_s | z_t) for one denoising step, conditioned on drug embeddings.
        Adapted from DiGress/DiffMS for our conditional setup.
        
        Args:
            s: Normalized timestep s [batch_size, 1]
            t: Normalized timestep t [batch_size, 1] 
            X_t: Node features at time t [batch_size, max_nodes, node_features]
            E_t: Edge features at time t [batch_size, max_nodes, max_nodes, edge_features]
            drug_embeddings: Drug embeddings for conditioning [batch_size, embedding_dim]
            node_mask: Node mask [batch_size, max_nodes]
            
        Returns:
            PlaceHolder object with graph G_s (E, X) given G_t (already masked)
            Note: G_s.y is just torch.zeros(batch_size, 0)
        """
        if self.phase == "pretrain_target":
            return None
            
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)
        
        # Get transition matrices
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)
        
        # Prepare graph data for neural network prediction
        graph_data = GraphData(
            X_t=X_t,
            E_t=E_t,
            node_mask=node_mask,
            noise_params={'t': t}
        )
        
        # Get extra features
        graph_data.X_extra, graph_data.y_extra = self.get_extra_features(
            X_t, E_t, {'t': t}, node_mask)
        
        # Create batch data with graph data
        batch_data = BatchData(graph_data=graph_data)
        
        # Create embedding data with drug embeddings
        embedding_data = EmbeddingData(drug_embedding=drug_embeddings)
        
        # Get neural network predictions
        pred = self.denoise_drug_graph(embedding_data, batch_data)
        
        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)  # [bs, n, d0]
        pred_E = F.softmax(pred.E, dim=-1)  # [bs, n, n, d0]
        
        # Compute posterior distributions p(z_s, z_t | z_0)
        p_s_and_t_given_0_X = compute_batched_over0_posterior_distribution(
            X_t=X_t, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
        )
        p_s_and_t_given_0_E = compute_batched_over0_posterior_distribution(
            X_t=E_t, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E  
        ) # both bs, N, d0, d_t-1
        
        # Combine predictions with posterior distributions
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X  # [bs, n, d0, d_t-1]
        unnormalized_prob_X = weighted_X.sum(dim=2)              # [bs, n, d_t-1]
        unnormalized_prob_X[torch.sum(
            unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(
            unnormalized_prob_X, dim=-1, keepdim=True)            # [bs, n, d_t-1]
        
        # Same for edges
        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E    # [bs, N, d0, d_t-1]
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])
        
        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()
        
        # Sample discrete features from the computed probabilities
        sampled_s = sample_discrete_features(prob_X, prob_E, node_mask=node_mask)
        
        # Convert to one-hot
        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()
        
        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)
        
        return PlaceHolder(
            X=X_s, E=E_s, y=torch.zeros(drug_embeddings.shape[0], 0)
        ).mask(node_mask).type_as(drug_embeddings)

        # # Create output placeholders
        # out_one_hot = PlaceHolder(X=X_s, E=E_s, y=torch.zeros(drug_embeddings.shape[0], 0))
        # out_discrete = PlaceHolder(X=X_s, E=E_s, y=torch.zeros(drug_embeddings.shape[0], 0))
        # return (
        #     out_one_hot.mask(node_mask).type_as(drug_embeddings),
        #     out_discrete.mask(node_mask, collapse=True).type_as(drug_embeddings)
        # )


    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        batch_data, embedding_data, prediction_data, loss_data = self._common_step(batch)
        
        # Update metrics & log loss(es) based on phase
        if self.phase not in ["pretrain_drug", "pretrain_target"]:
            if prediction_data.dti_scores is not None:
                self.train_metrics.update(
                    prediction_data.dti_scores, # were masked in _common_step
                    batch_data.dti_targets
                )
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
        
        self.log("train/loss_accuracy", loss_data.accuracy) if loss_data.accuracy is not None else None
        self.log("train/loss_complexity", loss_data.complexity) if loss_data.complexity is not None else None
        self.log("train/loss_contrastive", loss_data.contrastive) if loss_data.contrastive is not None else None
        self.log("train/loss_reconstruction", loss_data.reconstruction) if loss_data.reconstruction is not None else None
        self.log("train/loss", loss)
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        batch_data, embedding_data, prediction_data, loss_data = self._common_step(batch)
        
        # Update metrics & log loss(es) based on phase
        if self.phase not in ["pretrain_drug", "pretrain_target"]:
            if prediction_data.dti_scores is not None:
                self.val_metrics.update(
                    prediction_data.dti_scores, # were masked in _common_step
                    batch_data.dti_targets
                )
            for name, value in loss_data.components.items():
                self.log(f"val/loss_{name}", value)
        
        # Compute NLL for diffusion validation (when graph data is available)
        if self.phase != "pretrain_target" and batch_data.graph_data is not None:
            nll = self.compute_val_loss(
                batch_data, embedding_data, 
                prediction_data, test=False)
            self.log("val/nll", nll)
            
            # Sample molecules periodically for molecular metrics
            # This is expensive, so we only do it every few epochs
            # TODO: instead of every few epochs, perhaps (also) every n validation steps?
            # that way we don't do this for every batch in the validation set
            if hasattr(self, 'current_epoch') and self.current_epoch % self.sample_every_val == 0:
                generated_molecules = self.sample_batch(
                    drug_embeddings=embedding_data.drug_embedding,
                    num_samples_per_embedding=self.val_samples_per_embedding
                )
                target_molecules = [
                    Chem.MolFromSmiles(smiles) for smiles in batch_data.smiles
                ]
                # TODO: Look into these molecular metrics
                # valid/invalid, tanimoto sim w/ ground truth, exact match w/ ground truth
                # self.val_diffusion_metrics.update_all(generated_molecules, target_molecules)
        
        # if self.val_diffusion_metrics is not None:
        #     self.val_diffusion_metrics.update(
        #         prediction_data.graph_reconstruction, 
        #         batch_data.graph_data
        #     )

        # Compute total loss and log individual components
        loss = loss_data.compute_loss(self.weights)
        
        self.log("val/loss_accuracy", loss_data.accuracy) if loss_data.accuracy is not None else None
        self.log("val/loss_complexity", loss_data.complexity) if loss_data.complexity is not None else None
        self.log("val/loss_contrastive", loss_data.contrastive) if loss_data.contrastive is not None else None
        self.log("val/loss_reconstruction", loss_data.reconstruction) if loss_data.reconstruction is not None else None
        self.log("val/loss", loss)
        
        return loss
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        batch_data, embedding_data, prediction_data, loss_data = self._common_step(batch)
        
        # Update metrics & log loss(es) based on phase
        if self.phase not in ["pretrain_drug", "pretrain_target"]:
            if prediction_data.dti_scores is not None:
                self.test_metrics.update(
                    prediction_data.dti_scores, # were masked in _common_step
                    batch_data.dti_targets
                )
            for name, value in loss_data.components.items():
                self.log(f"test/loss_{name}", value)
        
        # Compute NLL for diffusion testing (when graph data is available)
        if self.phase != "pretrain_target" and batch_data.graph_data is not None:
            nll = self.compute_val_loss(batch_data, embedding_data, prediction_data, test=True)
            self.log("test/nll", nll)
            
            # Sample molecules for molecular metrics during testing
            # We always sample during testing since it's less frequent than validation
            generated_molecules = self.sample_batch(
                drug_embeddings=embedding_data.drug_embedding,
                num_samples_per_embedding=self.test_samples_per_embedding
            )
            target_molecules = [
                Chem.MolFromSmiles(smiles) for smiles in batch_data.smiles
            ]
            # TODO: we may need to repeat the target molecules because we generate test_samples_per_embedding per true molecule
            # TODO: Look into these molecular metrics
            # self.test_diffusion_metrics.update_all(generated_molecules, target_molecules)
        
        # Update diffusion metrics if available (already handled above)
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