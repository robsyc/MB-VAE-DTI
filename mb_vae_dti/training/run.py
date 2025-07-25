#!/usr/bin/env python3
"""
Unified training script for all DTI model variants and training phases.

Model types:
- baseline: Single drug/target feature, single DTI score prediction (finetune only)
- multi_modal: Multiple drug/target features, single DTI score prediction
- multi_output: Single drug/target feature, multiple DTI score prediction (+ train)
- multi_hybrid: Combination prior w/ contrastive loss (+ pretrain)
- full: Adds discrete diffusion decoder to the drug branch

Training phases:
- pretrain: Pre-train drug or target branch on unlabeled data
- train: Train on combined DTI dataset (multi-score prediction)
- finetune: Fine-tune on single DTI benchmark dataset

Usage:
    # Single training run
    CUDA_VISIBLE_DEVICES=0 ...
    python run.py --model baseline --phase finetune --dataset DAVIS --split rand

    python run.py --model multi_modal --phase finetune --dataset DAVIS --split cold
    
    # With overrides
    python run.py --model baseline --phase finetune --dataset DAVIS --split rand --override training.max_epochs=2
    
    # Gridsearch
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
import os
import shutil

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from omegaconf import OmegaConf, DictConfig
import wandb
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

MOLECULAR_STATISTICS_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "molecular_statistics.json"
CONFIG_ROOT = Path(__file__).parent / "configs"

from mb_vae_dti.training.datasets import DTIDataModule, PretrainDataModule

from mb_vae_dti.training.modules import (
    BaselineDTIModel,
    MultiModalDTIModel,
    MultiOutputDTIModel,
    MultiHybridDTIModel,
    FullDTIModel
)

from mb_vae_dti.training.utils import (
    ConfigManager, get_config_summary,
    setup_callbacks, BestMetricsCallback,
    setup_logging, generate_experiment_name,
    collect_results, save_results
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
    pretrain_target: Optional[PretrainTarget] = None,
    config_root: Optional[Path] = CONFIG_ROOT
) -> Path:
    """
    Get the config file path based on model type, phase, and other parameters.
    
    Args:
        model: Model type
        phase: Training phase
        dataset: Dataset (for finetune phase)
        split: Split type (for finetune and train phases)
        pretrain_target: Pretrain target (for pretrain phase)
        
    Returns:
        Config file path relative to training/configs/
    """
    if phase == "pretrain":
        return config_root / f"{model}/pretrain_{pretrain_target}.yaml"
    elif phase == "train":
        return config_root / f"{model}/pretrain_{split}.yaml"
    elif phase == "finetune":
        return config_root / f"{model}/{dataset}_{split}.yaml"
    else:
        raise ValueError(f"Unknown phase: {phase}")


def setup_model_baseline(
    config: DictConfig,
    feature_dims: Dict[str, Dict[str, int]],
    dataset: Dataset
) -> BaselineDTIModel:
    phase = "finetune"
    finetune_score = "Y_pKd" if dataset == "DAVIS" else "Y_KIBA"
    drug_feats = {name: dim for name, dim in feature_dims['drug'].items() if name in config.data.drug_features}
    target_feats = {name: dim for name, dim in feature_dims['target'].items() if name in config.data.target_features}

    logger.info(f"Drug features: {drug_feats}")
    logger.info(f"Target features: {target_feats}")
    logger.info(f"Phase: {phase}")
    logger.info(f"Finetune score: {finetune_score}")

    model = BaselineDTIModel(
        phase=phase,
        finetune_score=finetune_score,

        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        scheduler=config.training.scheduler,

        drug_features=drug_feats,
        target_features=target_feats,

        encoder_type=config.model.encoder_type,
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        factor=config.model.factor,
        n_layers=config.model.n_layers,
        activation=config.model.activation,
        dropout=config.model.dropout,
        bias=config.model.bias,
    )
    
    return model


def setup_model_multi_modal(
    config: DictConfig,
    feature_dims: Dict[str, Dict[str, int]],
    dataset: Dataset
) -> MultiModalDTIModel:
    phase = "finetune"
    finetune_score = "Y_pKd" if dataset == "DAVIS" else "Y_KIBA"
    drug_feats = {name: dim for name, dim in feature_dims['drug'].items() if name in config.data.drug_features}
    target_feats = {name: dim for name, dim in feature_dims['target'].items() if name in config.data.target_features}

    logger.info(f"Drug features: {drug_feats}")
    logger.info(f"Target features: {target_feats}")
    logger.info(f"Phase: {phase}")
    logger.info(f"Finetune score: {finetune_score}")

    model = MultiModalDTIModel(
        phase=phase,
        finetune_score=finetune_score,

        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        scheduler=config.training.scheduler,

        drug_features=drug_feats,
        target_features=target_feats,

        encoder_type=config.model.encoder_type,
        aggregator_type=config.model.aggregator_type,
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        factor=config.model.factor,
        n_layers=config.model.n_layers,
        activation=config.model.activation,
        dropout=config.model.dropout,
        bias=config.model.bias,
    )
    
    return model


def setup_model_multi_output(
    config: DictConfig,
    feature_dims: Dict[str, Dict[str, int]],
    dataset: Dataset,
    phase: TrainingPhase
) -> MultiOutputDTIModel:
    # Set finetune score based on phase & dataset
    if phase == "finetune":
        finetune_score = "Y_pKd" if dataset == "DAVIS" else "Y_KIBA"
    else:
        finetune_score = None
    drug_feats = {name: dim for name, dim in feature_dims['drug'].items() if name in config.data.drug_features}
    target_feats = {name: dim for name, dim in feature_dims['target'].items() if name in config.data.target_features}
    
    logger.info(f"Drug features: {drug_feats}")
    logger.info(f"Target features: {target_feats}")
    logger.info(f"Phase: {phase}")
    logger.info(f"Finetune score: {finetune_score}")
    
    model = MultiOutputDTIModel(
        phase=phase,
        finetune_score=finetune_score,
        dti_weights=config.loss.dti_weights if phase != "finetune" else None,

        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        scheduler=config.training.scheduler,

        drug_features=drug_feats,
        target_features=target_feats,

        encoder_type=config.model.encoder_type,
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        factor=config.model.factor,
        n_layers=config.model.n_layers,
        activation=config.model.activation,
        dropout=config.model.dropout,
        bias=config.model.bias,
    )

    # Load pretrained weights if specified
    if config.model.get('checkpoint_path') is not None:
        try:
            model.load_pretrained_weights(checkpoint_path=config.model.checkpoint_path)
        except Exception as e:
            logger.warning(f"Error loading pretrained weights: {e}")
            logger.warning("Continuing without pretrained weights")

    return model


def setup_model_multi_hybrid(
        # TODO: UPDATE TO NEW MULTI-HYBRID MODEL & CONFIGS
    config: DictConfig,
    feature_dims: Dict[str, Dict[str, int]],
    dataset: Dataset = None,
    phase: TrainingPhase = None,
    pretrain_target: PretrainTarget = None
) -> MultiHybridDTIModel:
    # Set finetune score based on phase & dataset
    if phase == "finetune":
        finetune_score = "Y_pKd" if dataset == "DAVIS" else "Y_KIBA"
    else:
        finetune_score = None
    
    drug_features = {name: dim for name, dim in feature_dims['drug'].items() if name in config.data.drug_features}
    target_features = {name: dim for name, dim in feature_dims['target'].items() if name in config.data.target_features}

    logger.info(f"Drug features: {drug_features}")
    logger.info(f"Target features: {target_features}")
    logger.info(f"Phase: {phase}")
    logger.info(f"Finetune score: {finetune_score}")
    
    # Construct the correct phase value for pretrain
    if phase == "pretrain":
        if pretrain_target is None:
            raise ValueError("pretrain_target must be specified for pretrain phase")
        model_phase = f"pretrain_{pretrain_target}"
    else:
        model_phase = phase
    
    model = MultiHybridDTIModel(
        phase=phase,
        finetune_score=finetune_score,

        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        scheduler=config.training.scheduler,

        drug_features=drug_features,
        target_features=target_features,

        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        factor=config.model.factor,
        n_layers=config.model.n_layers,
        activation=config.model.activation,
        dropout=config.model.dropout,
        bias=config.model.bias,

        encoder_type=config.model.encoder_type,
        aggregator_type=config.model.aggregator_type,
        weights=config.loss.weights,
        dti_weights=config.loss.dti_weights if phase != "finetune" else None,
        contrastive_temp=config.loss.contrastive_temp,
    )
    
    # Load pretrained weights if specified
    if config.model.get('drug_checkpoint_path') is not None:
        if os.path.exists(config.model.drug_checkpoint_path):
            model.load_pretrained_weights(checkpoint_path=config.model.drug_checkpoint_path, prefix="drug_")
        else:
            logger.warning(f"Checkpoint file not found: {config.model.drug_checkpoint_path}")
            logger.warning("Continuing without pretrained drug weights")
    
    if config.model.get('target_checkpoint_path') is not None:
        if os.path.exists(config.model.target_checkpoint_path):
            model.load_pretrained_weights(checkpoint_path=config.model.target_checkpoint_path, prefix="target_")
        else:
            logger.warning(f"Checkpoint file not found: {config.model.target_checkpoint_path}")
            logger.warning("Continuing without pretrained target weights")
    
    if config.model.get('checkpoint_path') is not None:
        if os.path.exists(config.model.checkpoint_path):
            model.load_pretrained_weights(checkpoint_path=config.model.checkpoint_path)
        else:
            logger.warning(f"Checkpoint file not found: {config.model.checkpoint_path}")
            logger.warning("Continuing without pretrained weights")

    return model


def setup_model_full(
    config: DictConfig,
    feature_dims: Dict[str, Dict[str, int]],
    dataset: Dataset = None,
    split: Split = None,
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
        model_phase = f"pretrain_{pretrain_target}"
    else:
        model_phase = phase
    
    # Set finetune score based on phase & dataset
    if phase == "finetune":
        finetune_score = "Y_pKd" if dataset == "DAVIS" else "Y_KIBA"
    else:
        finetune_score = None
    
    # Fetch dataset statistics
    if model_phase != "pretrain_target":
        with open(MOLECULAR_STATISTICS_PATH, "r") as f:
            molecular_statistics = json.load(f)
            dataset_key = "drugs_"
            if phase == "pretrain":
                dataset_key += "pretrain"
            else:
                dataset_key += split
            if phase == "finetune":
                dataset_key += f"_{dataset.lower()}"
            dataset_statistics = {
                "general": molecular_statistics["general"],
                "dataset": molecular_statistics["datasets"][dataset_key]
            }
    else:
        dataset_statistics = None
    
    model = FullDTIModel(
        embedding_dim=config.model.embedding_dim,
        drug_features=drug_features,
        target_features=target_features,
        encoder_type=config.model.encoder_type,
        encoder_kwargs=OmegaConf.to_container(config.model.encoder_kwargs),
        aggregator_type=config.model.aggregator_type,
        aggregator_kwargs=OmegaConf.to_container(config.model.aggregator_kwargs),
        fusion_kwargs=OmegaConf.to_container(config.model.get('fusion_kwargs', {})),
        dti_head_kwargs=OmegaConf.to_container(config.model.get('dti_head_kwargs', {})),
        infonce_head_kwargs=OmegaConf.to_container(config.model.get('infonce_head_kwargs', {})),
        diffusion_steps=config.diffusion.diffusion_steps,
        num_samples_to_generate=config.diffusion.num_samples_to_generate,
        graph_transformer_kwargs=OmegaConf.to_container(config.model.get('graph_transformer_kwargs', {})),
        dataset_infos=dataset_statistics,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        scheduler=config.training.scheduler,
        phase=model_phase,
        finetune_score=finetune_score,
        contrastive_weight=config.model.get('contrastive_weight', 0.1),
        complexity_weight=config.model.get('complexity_weight', 0.001),
        accuracy_weight=config.model.get('accuracy_weight', 1.0),
        reconstruction_weight=config.model.get('reconstruction_weight', 1.0),
        lambda_train=config.model.get('lambda_train', [1, 5, 0]),
    )
    
    # Load pretrained weights if specified
    if config.model.get('checkpoint_path') is not None:
        if os.path.exists(config.model.checkpoint_path):
            model.load_pretrained_weights(checkpoint_path=config.model.checkpoint_path)
        else:
            logger.warning(f"Checkpoint file not found: {config.model.checkpoint_path}")
            logger.warning("Continuing without pretrained weights")
    elif config.model.get('drug_checkpoint_path') is not None and config.model.get('target_checkpoint_path') is not None:
        if os.path.exists(config.model.drug_checkpoint_path) and os.path.exists(config.model.target_checkpoint_path):
            model.load_pretrained_weights(
                drug_checkpoint_path=config.model.drug_checkpoint_path,
                target_checkpoint_path=config.model.target_checkpoint_path
            )
        else:
            logger.warning(f"Checkpoint file not found: {config.model.drug_checkpoint_path}")
            logger.warning(f"Checkpoint file not found: {config.model.target_checkpoint_path}")
            logger.warning("Continuing without pretrained weights")
    
    return model


########################################################


def train_single_config(
    config: DictConfig,
    model_type: ModelType,
    phase: TrainingPhase,
    dataset: Optional[Dataset] = None,
    split: Optional[Split] = None,
    pretrain_target: Optional[PretrainTarget] = None,
    cleanup_checkpoints: bool = False
) -> None:
    """
    Train a single configuration.
    
    Args:
        config: Configuration to train
        model_type: Type of model to train
        phase: Training phase
        dataset: Dataset (for finetune phase)
        split: Split type (for finetune and train phases)
        pretrain_target: Pretrain target (for pretrain phase)
        cleanup_checkpoints: Whether to delete checkpoint files after training
                           (useful for gridsearch to save disk space)
    """
    logger.info(f"Experiment name: {config.logging.experiment_name}")
    save_dir = Path(config.logging.save_dir) / config.logging.experiment_name
    logger.info(f"Save directory: {save_dir}")
    logger.info(get_config_summary(config))

    # Initialize variables to None for proper cleanup
    data_module = None
    model = None
    trainer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        # Set random seed for reproducibility
        if config.hardware.get('deterministic', False) and config.hardware.gpus > 0:
            pl.seed_everything(config.hardware.seed, workers=True)
        elif config.hardware.gpus == 0:
            pl.seed_everything(config.hardware.seed, workers=False)
        
        ###############################################################
        # Setup data module
        ###############################################################
        logger.info("Setting up data module...")        
        if phase == "pretrain":
            # Pretrain datamodule (drugs.h5torch or targets.h5torch)
            data_module = PretrainDataModule(
                h5_path=config.data.h5_path,
                batch_size=config.data.batch_size,
                num_workers=config.data.num_workers,
                pin_memory=config.data.pin_memory,
                shuffle_train=config.data.shuffle_train,
                drop_last=config.data.drop_last,
                load_in_memory=config.data.load_in_memory,
                return_pyg=(model_type == "full" and pretrain_target != "target")
            )
        else:
            # DTI training/finetuning
            split_type = "split_rand" if split == "rand" else "split_cold"
            if phase == "finetune":
                provenance_cols = ["in_DAVIS"] if dataset == "DAVIS" else ["in_KIBA"]
            else:
                provenance_cols = None
            data_module = DTIDataModule(
                h5_path=config.data.h5_path,
                batch_size=config.data.batch_size,
                num_workers=config.data.num_workers,
                pin_memory=config.data.pin_memory,
                shuffle_train=config.data.shuffle_train,
                drop_last=config.data.drop_last,
                split_type=split_type,
                provenance_cols=provenance_cols,
                load_in_memory=config.data.load_in_memory,
                return_pyg=(model_type == "full" and pretrain_target != "target")
            )
        
        ###############################################################
        # Setup lightning module
        ###############################################################
        data_module.setup("fit")
        raw_feature_dims = data_module.get_feature_dims()
        
        # Transform feature dimensions for pretraining phases
        if phase == "pretrain":
            feature_dims = {
                'drug': raw_feature_dims if pretrain_target == "drug" else {},
                'target': raw_feature_dims if pretrain_target == "target" else {}
            }
        else:
            feature_dims = raw_feature_dims
        
        # Setup model based on type
        logger.info(f"Setting up {model_type} model...")
        model_setup_functions = {
            "baseline": lambda: setup_model_baseline(config, feature_dims, dataset),
            "multi_modal": lambda: setup_model_multi_modal(config, feature_dims, dataset),
            "multi_output": lambda: setup_model_multi_output(config, feature_dims, dataset, phase),
            "multi_hybrid": lambda: setup_model_multi_hybrid(config, feature_dims, dataset, phase, pretrain_target),
            "full": lambda: setup_model_full(config, feature_dims, dataset, split, phase, pretrain_target)
        }
        
        if model_type not in model_setup_functions:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = model_setup_functions[model_type]()
        
        ###############################################################
        # Setup trainer
        ###############################################################
        logger.info("Setting up trainer...")
        trainer = Trainer(
            max_epochs=config.training.max_epochs,
            accelerator="gpu" if config.hardware.gpus > 0 else "cpu",
            devices=config.hardware.gpus if config.hardware.gpus > 0 else 1,
            precision=config.hardware.precision,
            gradient_clip_val=config.training.get('gradient_clip_val'),
            deterministic=config.hardware.get('deterministic', False),
            logger=setup_logging(config, save_dir, model_type, phase, dataset, split, pretrain_target),
            callbacks=setup_callbacks(config, save_dir, monitor="val/loss"),
            log_every_n_steps=config.logging.log_every_n_steps,
            enable_checkpointing=True,
            default_root_dir=str(save_dir),
            # Debug mode limits - set from config if debug mode is enabled
            limit_train_batches=config.training.get('limit_train_batches', 1.0),
            limit_val_batches=config.training.get('limit_val_batches', 1.0),
            limit_test_batches=config.training.get('limit_test_batches', 1.0)
        )

        ###############################################################
        # Train/Test loop
        ###############################################################
        logger.info("Starting training...")
        trainer.fit(model, data_module)
        
        # Test model on best checkpoint
        if phase != "pretrain":
            logger.info("Testing on best checkpoint...")
            trainer.test(model, data_module, ckpt_path="best")
            
            # Cleanup checkpoint files if requested (for gridsearch/ensemble)
            if cleanup_checkpoints:
                checkpoint_dir = save_dir / "checkpoints"
                if checkpoint_dir.exists():
                    shutil.rmtree(checkpoint_dir)
                    logger.info(f"Cleaned up checkpoint directory: {checkpoint_dir}")
                wandb_dir = save_dir / "wandb"
                if wandb_dir.exists():
                    shutil.rmtree(wandb_dir)
                    logger.info(f"Cleaned up wandb directory: {wandb_dir}")
            else:
                best_model_path = save_dir / "checkpoints" / "best_model.ckpt"
                if best_model_path.exists():
                    logger.info(f"Saved best model to {best_model_path}")
                else:
                    logger.warning("Best model not found")

        ###############################################################
        # Collect results
        ###############################################################
        logger.info("Collecting results...")
        results = collect_results(config, trainer)
        save_results(results, save_dir)

        logger.info("Training completed!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    finally:
        # Comprehensive cleanup - this is critical for gridsearch
        logger.info("Starting cleanup...")
        
        # Print system memory info before cleanup
        if torch.cuda.is_available():
            logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        if model is not None:
            del model
        if data_module is not None:
            del data_module
        if trainer is not None:
            del trainer
        gc.collect()
        
        # Print memory info after cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"After cleanup - CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            logger.info(f"After cleanup - CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
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
    config_manager = ConfigManager()
    
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
        logger.info("Debug mode enabled - only keeping max 3 configs and applying debug settings")
        configs = configs[:3]
        for config in configs:
            # Limit batches to 5 for each split
            config.training.limit_train_batches = 5
            config.training.limit_val_batches = 5
            config.training.limit_test_batches = 5
            
            # Ease computational load for local CPU testing
            config.logging.log_every_n_steps = 2
            config.training.max_epochs = 2
            config.model.diffusion_steps = 5
            config.model.num_samples_to_generate = 2
            config.data.batch_size = min(config.data.batch_size, 8)
            config.data.pin_memory = False
            config.data.num_workers = 0
            config.data.load_in_memory = False
            config.hardware.gpus = 0
            config.hardware.deterministic = False
    
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

        config.logging.experiment_name = generate_experiment_name(
            model_type, phase, args.dataset, args.split, args.pretrain_target, config_index, args.batch_index, args.ensemble
        )
        
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
                phase,
                args.dataset,
                args.split,
                args.pretrain_target,
                cleanup_checkpoints=(args.gridsearch or args.ensemble)
            )
        except Exception as e:
            logger.error(f"Failed to train configuration {i+1}: {e}")
            logger.error(f"Configuration summary: {get_config_summary(config)}")
            
            if args.gridsearch or args.ensemble:
                if args.debug:
                    raise e
                else:
                    logger.warning("Continuing with next configuration...")
                    continue
            else:
                raise
    
    logger.info(f"\n{'='*50}")
    logger.info("All training runs completed!")
    logger.info(f"Total configurations processed: {len(configs)}")
    logger.info(f"{'='*50}")

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