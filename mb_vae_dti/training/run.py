#!/usr/bin/env python3
"""
Unified training script for all DTI model variants and training phases.

Model types:
- baseline: Single drug/target feature, single DTI score prediction
- multi_modal: Multiple drug/target features, single DTI score prediction
- multi_output: Single drug/target feature, multiple DTI score prediction
- full: Multiple drug/target features, multiple DTI scores, pre-training support

Training phases:
- pretrain: Pre-train drug or target branch on unlabeled data
- train: Train on combined DTI dataset (multi-score prediction)
- finetune: Fine-tune on single DTI benchmark dataset

Usage:
    # Single training run
    python run.py --model baseline --phase finetune --dataset DAVIS --split rand
    python run.py --model multi_modal --phase finetune --dataset DAVIS --split cold
    
    # With overrides
    python run.py --model baseline --phase finetune --dataset DAVIS --split rand --override training.max_epochs=2
    
    # Gridsearch
    python run.py --model baseline --phase finetune --dataset DAVIS --split rand --gridsearch
    
    # HPC batch processing
    python run.py --model baseline --phase finetune --dataset DAVIS --split rand --gridsearch --batch_index 0 --total_batches 5

    # Ensemble
    python run.py --model baseline --phase finetune --dataset DAVIS --split rand --gridsearch --ensemble
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Literal
import argparse
import gc

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from omegaconf import OmegaConf, DictConfig
import wandb

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from mb_vae_dti.training.datasets import DTIDataModule, PretrainDataModule

from mb_vae_dti.training.modules import (
    BaselineDTIModel,
    MultiModalDTIModel,
    MultiOutputDTIModel,
    MultiHybridDTIModel,
    FullDTIModel
)

from mb_vae_dti.training.utils import (
    ConfigManager, save_config, get_config_summary,
    setup_callbacks,
    setup_logging, generate_experiment_name,
    collect_validation_results, save_gridsearch_results,
    collect_test_results, save_ensemble_member_results, aggregate_ensemble_results
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Model type definitions
ModelType = Literal["baseline", "multi_modal", "multi_output", "multi_hybrid", "full"]
TrainingPhase = Literal["pretrain", "train", "finetune"]
Dataset = Literal["DAVIS", "KIBA"]
Split = Literal["rand", "cold"]
PretrainTarget = Literal["drug", "target"]


########################################################

def validate_combination(
    model: ModelType,
    phase: TrainingPhase,
    dataset: Optional[Dataset] = None,
    split: Optional[Split] = None,
    pretrain_target: Optional[PretrainTarget] = None
) -> None:
    """
    Validate that the combination of model, phase, dataset, and split is valid.
    
    Args:
        model: Model type
        phase: Training phase
        dataset: Dataset (required for finetune phase)
        split: Split type (required for finetune and train phases)
        pretrain_target: Pretrain target (required for pretrain phase)
    """
    # Pretrain phase validation
    if phase == "pretrain":
        if model in ["baseline", "multi_modal", "multi_output"]:
            raise ValueError("Only 'full' and 'multi_hybrid' models can be pre-trained")
        if pretrain_target is None:
            raise ValueError("Pretrain phase requires --pretrain_target (drug or target)")
        if dataset is not None or split is not None:
            raise ValueError("Pretrain phase should not specify dataset or split")
    
    # Train phase validation
    elif phase == "train":
        if model in ["baseline", "multi_modal"]:
            raise ValueError("Only 'full', 'multi_hybrid' and 'multi_output' models can do general DTI training")
        if split is None:
            raise ValueError("Train phase requires --split (rand or cold)")
        if pretrain_target is not None or dataset is not None:
            raise ValueError("Train phase should not specify pretrain_target or dataset")
    
    # Finetune phase validation
    elif phase == "finetune":
        if dataset is None:
            raise ValueError("Finetune phase requires --dataset (DAVIS or KIBA)")
        if split is None:
            raise ValueError("Finetune phase requires --split (rand or cold)")
        if pretrain_target is not None:
            raise ValueError("Finetune phase should not specify pretrain_target")


def get_config_path(
    model: ModelType,
    phase: TrainingPhase,
    dataset: Optional[Dataset] = None,
    split: Optional[Split] = None,
    pretrain_target: Optional[PretrainTarget] = None
) -> str:
    """
    Get the config file path based on model type, phase, and other parameters.
    
    Args:
        model: Model type
        phase: Training phase
        dataset: Dataset (for finetune phase)
        split: Split type (for finetune phase)
        pretrain_target: Pretrain target (for pretrain phase)
        
    Returns:
        Config file path relative to training/configs/
    """
    if phase == "pretrain":
        return f"{model}/pretrain_{pretrain_target}.yaml"
    elif phase == "train":
        return f"{model}/pretrain_{split}.yaml"
    elif phase == "finetune":
        return f"{model}/{dataset}_{split}.yaml"
    else:
        raise ValueError(f"Unknown phase: {phase}")


def get_model_class(model: ModelType):
    """Get the model class for the specified model type."""
    if model == "baseline":
        return BaselineDTIModel
    elif model == "multi_modal":
        return MultiModalDTIModel
    elif model == "multi_output":
        return MultiOutputDTIModel
    elif model == "multi_hybrid":
        return MultiHybridDTIModel
    elif model == "full":
        return FullDTIModel
    else:
        raise ValueError(f"Unknown model type: {model}")


def get_datamodule_class(phase: TrainingPhase):
    """Get the datamodule class for the specified training phase."""
    if phase == "pretrain":
        return PretrainDataModule
    elif phase in ["train", "finetune"]:
        return DTIDataModule
    else:
        raise ValueError(f"Unknown phase: {phase}")


def setup_model_baseline(
    config: DictConfig,
    feature_dims: Dict[str, Dict[str, int]],
    dataset: Dataset = None
) -> BaselineDTIModel:
    """Setup baseline model with single drug/target features."""

    model = BaselineDTIModel(
        embedding_dim=config.model.embedding_dim,
        drug_input_dim=feature_dims['drug'][config.data.drug_feature],
        target_input_dim=feature_dims['target'][config.data.target_feature],
        encoder_type=config.model.encoder_type,
        encoder_kwargs=OmegaConf.to_container(config.model.encoder_kwargs),
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        scheduler=config.training.scheduler,
        finetune_score="Y_pKd" if dataset == "DAVIS" else "Y_KIBA",
        drug_feature=config.data.drug_feature,
        target_feature=config.data.target_feature
    )
    
    return model


def setup_model_multi_modal(
    config: DictConfig,
    feature_dims: Dict[str, Dict[str, int]],
    dataset: Dataset = None
) -> MultiModalDTIModel:
    """Setup multi-modal model with multiple drug/target features."""
    # Build feature dimension dictionaries
    drug_features = {}
    target_features = {}
    
    for feat_name in config.data.drug_features:
        if feat_name in feature_dims['drug']:
            drug_features[feat_name] = feature_dims['drug'][feat_name]
        else:
            raise ValueError(f"Drug feature '{feat_name}' not found in dataset")
    
    for feat_name in config.data.target_features:
        if feat_name in feature_dims['target']:
            target_features[feat_name] = feature_dims['target'][feat_name]
        else:
            raise ValueError(f"Target feature '{feat_name}' not found in dataset")

    model = MultiModalDTIModel(
        embedding_dim=config.model.embedding_dim,
        drug_features=drug_features,
        target_features=target_features,
        encoder_type=config.model.encoder_type,
        encoder_kwargs=OmegaConf.to_container(config.model.encoder_kwargs),
        aggregator_type=config.model.aggregator_type,
        aggregator_kwargs=OmegaConf.to_container(config.model.aggregator_kwargs),
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        scheduler=config.training.scheduler,
        finetune_score="Y_pKd" if dataset == "DAVIS" else "Y_KIBA",
    )
    
    return model


def setup_model_multi_output(
    config: DictConfig,
    feature_dims: Dict[str, Dict[str, int]],
    dataset: Dataset = None,
    phase: TrainingPhase = None
) -> MultiOutputDTIModel:
    """Setup multi-output model with single drug/target features and DTI prediction head."""

    # Set finetune score based on phase & dataset
    if phase == "finetune":
        finetune_score = "Y_pKd" if dataset == "DAVIS" else "Y_KIBA"
    else:
        finetune_score = None
    
    model = MultiOutputDTIModel(
        embedding_dim=config.model.embedding_dim,
        drug_input_dim=feature_dims['drug'][config.data.drug_feature],
        target_input_dim=feature_dims['target'][config.data.target_feature],
        encoder_type=config.model.encoder_type,
        encoder_kwargs=OmegaConf.to_container(config.model.encoder_kwargs),
        fusion_kwargs=OmegaConf.to_container(config.model.fusion_kwargs),
        dti_head_kwargs=OmegaConf.to_container(config.model.dti_head_kwargs),
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        scheduler=config.training.scheduler,
        drug_feature=config.data.drug_feature,
        target_feature=config.data.target_feature,
        phase=phase,
        finetune_score=finetune_score
    )

    # Load pretrained weights if specified
    if config.model.get('checkpoint_path') is not None:
        model.load_pretrained_weights(checkpoint_path=config.model.checkpoint_path)
    
    return model


def setup_model_multi_hybrid(
    config: DictConfig,
    feature_dims: Dict[str, Dict[str, int]],
    dataset: Dataset = None,
    phase: TrainingPhase = None,
    pretrain_target: PretrainTarget = None
) -> MultiHybridDTIModel:
    """Setup multi-hybrid model with multi-modal inputs and multi-output predictions."""
    # Build feature dimension dictionaries
    drug_features = {}
    target_features = {}
    
    if config.data.drug_features is not None:
        for feat_name in config.data.drug_features:
            if feat_name in feature_dims['drug']:
                drug_features[feat_name] = feature_dims['drug'][feat_name]
            else:
                raise ValueError(f"Drug feature '{feat_name}' not found in dataset")
    
    if config.data.target_features is not None:
        for feat_name in config.data.target_features:
            if feat_name in feature_dims['target']:
                target_features[feat_name] = feature_dims['target'][feat_name]
            else:
                raise ValueError(f"Target feature '{feat_name}' not found in dataset")
    
    # Construct the correct phase value for pretrain
    if phase == "pretrain":
        if pretrain_target is None:
            raise ValueError("pretrain_target must be specified for pretrain phase")
        model_phase = f"pretrain_{pretrain_target}"
    else:
        model_phase = phase
    
    # Set finetune score based on phase & dataset
    if phase == "finetune":
        finetune_score = "Y_pKd" if dataset == "DAVIS" else "Y_KIBA"
    else:
        finetune_score = None            
    
    model = MultiHybridDTIModel(
        drug_features=drug_features,
        target_features=target_features,
        embedding_dim=config.model.embedding_dim,
        encoder_type=config.model.encoder_type,
        encoder_kwargs=OmegaConf.to_container(config.model.encoder_kwargs),
        aggregator_type=config.model.aggregator_type,
        aggregator_kwargs=OmegaConf.to_container(config.model.aggregator_kwargs),
        fusion_kwargs=OmegaConf.to_container(config.model.fusion_kwargs),
        dti_head_kwargs=OmegaConf.to_container(config.model.dti_head_kwargs),
        infonce_head_kwargs=OmegaConf.to_container(config.model.infonce_head_kwargs),
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        scheduler=config.training.scheduler,
        phase=model_phase,
        finetune_score=finetune_score,
        contrastive_weight=config.model.get('contrastive_weight', 1.0),
        temperature=config.model.get('temperature', 0.07),
    )
    
    # Load pretrained weights if specified
    if config.model.get('checkpoint_path') is not None:
        model.load_pretrained_weights(checkpoint_path=config.model.checkpoint_path)
    elif config.model.get('drug_checkpoint_path') is not None or config.model.get('target_checkpoint_path') is not None:
        model.load_pretrained_weights(
            drug_checkpoint_path=config.model.get('drug_checkpoint_path'),
            target_checkpoint_path=config.model.get('target_checkpoint_path')
        )
    
    return model


def setup_model_full(
    config: DictConfig,
    feature_dims: Dict[str, Dict[str, int]],
    dataset: Dataset = None,
    phase: TrainingPhase = None,
    pretrain_target: PretrainTarget = None
):
    """Setup full model with VAE drug branch and diffusion decoder."""
    # Build feature dimension dictionaries
    drug_features = {}
    target_features = {}
    
    if config.data.drug_features is not None:
        for feat_name in config.data.drug_features:
            if feat_name in feature_dims['drug']:
                drug_features[feat_name] = feature_dims['drug'][feat_name]
            else:
                raise ValueError(f"Drug feature '{feat_name}' not found in dataset")
    
    if config.data.target_features is not None:
        for feat_name in config.data.target_features:
            if feat_name in feature_dims['target']:
                target_features[feat_name] = feature_dims['target'][feat_name]
            else:
                raise ValueError(f"Target feature '{feat_name}' not found in dataset")
    
    # Construct the correct phase value for pretrain
    if phase == "pretrain":
        if pretrain_target is None:
            raise ValueError("pretrain_target must be specified for pretrain phase")
        model_phase = f"pretrain_{pretrain_target}"
    else:
        model_phase = phase
    
    # Set finetune score based on phase & dataset
    if phase == "finetune":
        finetune_score = "Y_pKd" if dataset == "DAVIS" else "Y_KIBA"
    else:
        finetune_score = None
    
    # Prepare dataset statistics for diffusion decoder
    dataset_statistics = config.model.get('dataset_statistics', None)
    if dataset_statistics is not None:
        # Convert OmegaConf DictConfig to a regular dict before further processing
        dataset_statistics = OmegaConf.to_container(dataset_statistics, resolve=True)
        # Convert marginals to tensors if present
        if 'x_marginals' in dataset_statistics:
            dataset_statistics['x_marginals'] = torch.tensor(dataset_statistics['x_marginals'])
        if 'e_marginals' in dataset_statistics:
            dataset_statistics['e_marginals'] = torch.tensor(dataset_statistics['e_marginals'])
    
    model = FullDTIModel(
        drug_features=drug_features,
        target_features=target_features,
        embedding_dim=config.model.embedding_dim,
        encoder_type=config.model.encoder_type,
        encoder_kwargs=OmegaConf.to_container(config.model.encoder_kwargs),
        aggregator_type=config.model.aggregator_type,
        aggregator_kwargs=OmegaConf.to_container(config.model.aggregator_kwargs),
        fusion_kwargs=OmegaConf.to_container(config.model.get('fusion_kwargs', {})),
        dti_head_kwargs=OmegaConf.to_container(config.model.get('dti_head_kwargs', {})),
        infonce_head_kwargs=OmegaConf.to_container(config.model.get('infonce_head_kwargs', {})),
        variational_head_kwargs=OmegaConf.to_container(config.model.get('variational_head_kwargs', {})),
        diffusion_decoder_kwargs=OmegaConf.to_container(config.model.get('diffusion_decoder_kwargs', {})),
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        scheduler=config.training.scheduler,
        phase=model_phase,
        finetune_score=finetune_score,
        checkpoint_path=config.model.get('checkpoint_path'),
        contrastive_weight=config.model.get('contrastive_weight', 1.0),
        kl_weight=config.model.get('kl_weight', 0.1),
        reconstruction_weight=config.model.get('reconstruction_weight', 1.0),
        temperature=config.model.get('temperature', 0.07),
        dataset_statistics=dataset_statistics,
    )
    
    # Load pretrained weights if specified
    if config.model.get('checkpoint_path') is not None:
        model.load_pretrained_weights(checkpoint_path=config.model.checkpoint_path)
    elif config.model.get('drug_checkpoint_path') is not None or config.model.get('target_checkpoint_path') is not None:
        model.load_pretrained_weights(
            drug_checkpoint_path=config.model.get('drug_checkpoint_path'),
            target_checkpoint_path=config.model.get('target_checkpoint_path')
        )
    
    return model


def setup_model(
    model_type: ModelType,
    config: DictConfig,
    feature_dims: Dict[str, Dict[str, int]],
    dataset: Dataset = None,
    phase: TrainingPhase = None,
    pretrain_target: PretrainTarget = None
):
    """Setup model based on type."""
    if model_type == "baseline":
        return setup_model_baseline(config, feature_dims, dataset)
    elif model_type == "multi_modal":
        return setup_model_multi_modal(config, feature_dims, dataset)
    elif model_type == "multi_output":
        return setup_model_multi_output(config, feature_dims, dataset, phase)
    elif model_type == "multi_hybrid":
        return setup_model_multi_hybrid(config, feature_dims, dataset, phase, pretrain_target)
    elif model_type == "full":
        return setup_model_full(config, feature_dims, dataset, phase, pretrain_target)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


########################################################


def train_single_config(
    config: DictConfig,
    model_type: ModelType,
    dataset: Dataset,
    split: Split,
    phase: TrainingPhase,
    pretrain_target: PretrainTarget = None,
    config_index: Optional[int] = None,
    batch_index: Optional[int] = None,
    is_ensemble: bool = False,
) -> None:
    """
    Train a single configuration.
    
    Args:
        config: Configuration to train
        model_type: Type of model to train
        phase: Training phase
        config_index: Optional index of the config in the gridsearch experiment batch
        batch_index: Optional index of the batch in the gridsearch experiment
        is_ensemble: Whether this is part of an ensemble run
    """
    # Determine if we're in gridsearch mode
    is_gridsearch = config_index is not None and not is_ensemble
    
    # Update experiment name if this is a gridsearch or ensemble run
    if is_gridsearch:
        original_name = OmegaConf.select(config, "logging.experiment_name", default="experiment")
        config.logging.experiment_name = generate_experiment_name(
            original_name, batch_index, config_index
        )
    elif is_ensemble:
        original_name = OmegaConf.select(config, "logging.experiment_name", default="experiment")
        config.logging.experiment_name = f"{original_name}_ensemble_{config_index+1}"
    
    # Initialize variables to None for proper cleanup
    model = None
    data_module = None
    trainer = None
    loggers = None
    callbacks = None
    
    try:
        # Set random seed for reproducibility
        if config.hardware.get('deterministic', False) and config.hardware.gpus > 0:
            pl.seed_everything(config.hardware.seed, workers=True)
        elif config.hardware.gpus == 0:
            pl.seed_everything(config.hardware.seed, workers=False)
        
        # Setup data module
        logger.info("Setting up data module...")
        datamodule_class = get_datamodule_class(phase)
        
        if phase == "pretrain":
            # Pretrain datamodule (drugs.h5torch or targets.h5torch)
            data_module = datamodule_class(
                h5_path=config.data.h5_path,
                batch_size=config.data.batch_size,
                num_workers=config.data.num_workers,
                pin_memory=config.data.pin_memory,
                shuffle_train=config.data.shuffle_train,
                drop_last=config.data.drop_last,
                load_in_memory=config.data.load_in_memory
            )
        else:
            # DTI training/finetuning
            split_type = "split_rand" if split == "rand" else "split_cold"
            if phase == "finetune":
                provenance_cols = ["in_DAVIS"] if dataset == "DAVIS" else ["in_KIBA"]
            else:
                provenance_cols = None
            data_module = datamodule_class(
                h5_path=config.data.h5_path,
                batch_size=config.data.batch_size,
                num_workers=config.data.num_workers,
                pin_memory=config.data.pin_memory,
                shuffle_train=config.data.shuffle_train,
                drop_last=config.data.drop_last,
                split_type=split_type,
                provenance_cols=provenance_cols,
                load_in_memory=config.data.load_in_memory
            )
        
        # Setup lightning module
        data_module.setup("fit")
        raw_feature_dims = data_module.get_feature_dims()
        
        # Transform feature dimensions for pretraining phases
        if phase == "pretrain":
            # For pretraining, we need to map flat feature dims to drug/target structure
            if pretrain_target == "drug":
                feature_dims = {'drug': raw_feature_dims, 'target': {}}
            elif pretrain_target == "target":
                feature_dims = {'drug': {}, 'target': raw_feature_dims}
            else:
                raise ValueError(f"Unknown pretrain phase: {pretrain_target}")
        else:
            # For DTI training/finetuning, feature_dims is already structured
            feature_dims = raw_feature_dims
        
        logger.info(f"Setting up {model_type} model...")
        model = setup_model(model_type, config, feature_dims, dataset, phase, pretrain_target)
        
        # Setup logging & callbacks
        save_dir = Path(config.logging.save_dir) / config.logging.experiment_name
        loggers = setup_logging(config, save_dir, phase)
        callbacks = setup_callbacks(config, save_dir, is_gridsearch=is_gridsearch)
        
        # Setup trainer
        logger.info("Setting up trainer...")
        trainer = Trainer(
            max_epochs=config.training.max_epochs,
            accelerator="gpu" if config.hardware.gpus > 0 else "cpu",
            devices=config.hardware.gpus if config.hardware.gpus > 0 else 1,
            precision=config.hardware.precision,
            gradient_clip_val=config.training.get('gradient_clip_val'),
            deterministic=config.hardware.get('deterministic', False),
            logger=loggers,
            callbacks=callbacks,
            log_every_n_steps=config.logging.log_every_n_steps,
            enable_checkpointing=not is_gridsearch,
            default_root_dir=str(save_dir),
            # Debug mode limits - set from config if debug mode is enabled
            limit_train_batches=config.training.get('limit_train_batches', 1.0),
            limit_val_batches=config.training.get('limit_val_batches', 1.0),
            limit_test_batches=config.training.get('limit_test_batches', 1.0)
        )
        
        # Save configuration
        save_config(config, save_dir / "config.yaml")
        
        # Train model
        logger.info("Starting training...")
        trainer.fit(model, data_module)
        
        # Test model on best checkpoint
        if not (is_gridsearch or is_ensemble) and not phase == "pretrain":
            logger.info("Testing on best checkpoint...")
            trainer.test(model, data_module, ckpt_path="best")

            final_model_path = save_dir / "final_model.pt"
            torch.save(model.state_dict(), final_model_path)
            logger.info(f"Saved final model to {final_model_path}")
        elif is_gridsearch:
            logger.info("Collecting gridsearch results...")
            results = collect_validation_results(config, trainer)
            save_gridsearch_results(results, Path(config.logging.save_dir) / "gridsearch_results")
        elif is_ensemble and not phase == "pretrain":
            logger.info("Testing ensemble model on best checkpoint...")
            trainer.test(model, data_module, ckpt_path="best")
            logger.info("Collecting ensemble test results...")
            test_results = collect_test_results(config, trainer, config_index)
            save_ensemble_member_results(test_results, Path(config.logging.save_dir), config_index)
        
        logger.info("Training completed!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Comprehensive cleanup - this is critical for gridsearch
        logger.info("Starting cleanup...")
        
        # Close wandb run if it was initialized
        try:
            wandb.finish()
        except Exception as e:
            logger.warning(f"Failed to close wandb run: {e}")
        

def main(args):
    """Main training function."""
    model_type = args.model
    phase = args.phase
    
    # Validate combination
    validate_combination(
        model=model_type,
        phase=phase,
        dataset=args.dataset,
        split=args.split,
        pretrain_target=args.pretrain_target
    )
    
    # Validate gridsearch and ensemble are mutually exclusive
    if args.gridsearch and args.ensemble:
        logger.error("Cannot specify both --gridsearch and --ensemble")
        return
    
    if args.gridsearch:
        if args.batch_index is None or args.total_batches is None:
            logger.error("--batch_index and --total_batches must be used together when running gridsearch")
            return
    else:
        if args.batch_index is not None or args.total_batches is not None:
            logger.error("--batch_index and --total_batches can only be used when running gridsearch")
            return
    
    # Get config path
    config_path = get_config_path(
        model=model_type,
        phase=phase,
        dataset=args.dataset,
        split=args.split,
        pretrain_target=args.pretrain_target
    )
    
    logger.info(f"Using config: {config_path}")
    
    # Initialize configuration manager
    config_manager = ConfigManager(config_root="configs")
    
    # Load configuration(s)
    configs = config_manager.load_config(
        config_path=config_path,
        overrides=args.override,
        gridsearch=args.gridsearch,
        ensemble=args.ensemble
    )
    
    # Convert to list if single config
    if isinstance(configs, DictConfig):
        configs = [configs]
    
    # Apply debug settings if debug mode is enabled
    if args.debug:
        logger.info("Debug mode enabled - applying debug settings to all configs")
        for config in configs:
            # Limit batches to 5 for each split
            config.training.limit_train_batches = 5
            config.training.limit_val_batches = 5
            config.training.limit_test_batches = 5
            
            # Reduce epochs to speed up testing
            config.training.max_epochs = min(config.training.max_epochs, 2)
            
            # Reduce batch size if it's too large
            config.data.batch_size = min(config.data.batch_size, 8)
            
            # Disable some expensive operations
            config.hardware.deterministic = False
            config.data.pin_memory = False
            config.data.num_workers = 0
            config.hardware.gpus = 0
    
    # Handle batch processing for gridsearch
    if args.gridsearch:
        if args.batch_index is not None and args.total_batches is not None:
            # Shuffle configs deterministically
            configs = config_manager.shuffle_configs(configs)
            
            # Create batches
            batches = config_manager.batch_configs(configs, total_batches=args.total_batches)
            
            # Get the specific batch
            if args.batch_index >= len(batches):
                logger.error(f"Batch index {args.batch_index} >= number of batches {len(batches)}")
                return
            
            configs = batches[args.batch_index]
            logger.info(f"Running batch {args.batch_index + 1}/{len(batches)} with {len(configs)} configurations")
        else:
            logger.info(f"Running full gridsearch with {len(configs)} configurations")
    elif args.ensemble:
        logger.info(f"Running ensemble with {len(configs)} configurations")
    
    # Train each configuration
    for i, config in enumerate(configs):
        config_index = i if (args.gridsearch or args.ensemble) else None

        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Training configuration {i+1}/{len(configs)}")
        if args.gridsearch:
            logger.info(f"Config index: {config_index}")
            if args.batch_index is not None:
                logger.info(f"Batch index: {args.batch_index}")
        elif args.ensemble:
            logger.info(f"Ensemble member: {config_index + 1}")
        logger.info(f"Configuration summary: {get_config_summary(config)}")
        logger.info(f"{'='*50}\n")

        try:
            train_single_config(
                config, 
                model_type, 
                args.dataset,
                args.split,
                phase, 
                args.pretrain_target,
                config_index, 
                args.batch_index,
                args.ensemble
            )
        except Exception as e:
            logger.error(f"Failed to train configuration {i+1}: {e}")
            logger.error(f"Configuration summary: {get_config_summary(config)}")
            
            if args.gridsearch or args.ensemble:
                logger.warning("Continuing with next configuration...")
                continue
            else:
                raise
    
    logger.info(f"\n{'='*50}")
    logger.info("All training runs completed!")
    logger.info(f"Total configurations processed: {len(configs)}")
    logger.info(f"{'='*50}")
    
    # Aggregate ensemble results if this was an ensemble run
    if args.ensemble and not phase == "pretrain":
        logger.info("\n" + "="*50)
        logger.info("Aggregating ensemble results...")
        logger.info("="*50)
        
        try:
            # Get the base save directory from the first config
            base_save_dir = Path(configs[0].logging.save_dir)
            
            # Aggregate results
            aggregated_results = aggregate_ensemble_results(
                save_dir=base_save_dir,
                expected_members=len(configs)
            )
            
            logger.info("Ensemble aggregation completed successfully!")
            logger.info(f"Results saved to: {base_save_dir}/ensemble_results/aggregated_results.json")
            
            # Print summary of key metrics
            test_metrics = aggregated_results.get("test_metrics", {})
            if test_metrics:
                logger.info("\nEnsemble Test Results Summary:")
                for metric_name, stats in test_metrics.items():
                    logger.info(f"  {metric_name}: {stats['mean']:.6f} ± {stats['std']:.6f} "
                               f"(min: {stats['min']:.6f}, max: {stats['max']:.6f})")
            
        except Exception as e:
            logger.error(f"Failed to aggregate ensemble results: {e}")
            logger.error("Individual member results should still be available in ensemble_results/ directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified DTI model training script")
    
    # Model specification
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "multi_modal", "multi_output", "multi_hybrid", "full"],
        help="Model type"
    )
    
    # Phase specification
    parser.add_argument(
        "--phase",
        type=str,
        choices=["pretrain", "train", "finetune"],
        help="Training phase (required when using --model)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["DAVIS", "KIBA", "Metz", "BindingDB_Kd", "BindingDB_Ki"],
        help="Dataset for finetuning (required when phase=finetune)"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["rand", "cold"],
        help="Split type for finetuning (required when phase=finetune)"
    )
    parser.add_argument(
        "--pretrain_target",
        type=str,
        choices=["drug", "target"],
        help="Target for pretraining (required when phase=pretrain)"
    )
    
    # Common arguments
    parser.add_argument(
        "--override",
        nargs="+",
        default=None,
        help="Override config parameters (e.g., model.embedding_dim=512)"
    )
    parser.add_argument(
        "--gridsearch",
        action="store_true",
        help="Run gridsearch over all parameter combinations in config"
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Run ensemble over predefined configurations in config"
    )
    parser.add_argument(
        "--batch_index",
        type=int,
        default=None,
        help="Index of batch to run (0-based, for HPC usage)"
    )
    parser.add_argument(
        "--total_batches",
        type=int,
        default=None,
        help="Total number of batches to split gridsearch into"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: limit to 5 batches per split, reduce epochs, disable checkpointing"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.model is None or args.phase is None:
        parser.error("--model and --phase must be provided")
    
    main(args) 