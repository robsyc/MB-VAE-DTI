"""
Runs perturbation experiments

Baseline:
python mb_vae_dti/validating/perturbation.py \
  --dataset DAVIS --split split_rand --model baseline --branch drug \
  --steps 11 --output /root/MB-VAE-DTI/data/results/perturbation/baseline_davis_rand_drug.json

Multi-modal on all or single feature:
python mb_vae_dti/validating/perturbation.py \
  --dataset DAVIS --split split_rand --model multi_modal --branch drug \
  --steps 11 --output /root/MB-VAE-DTI/data/results/perturbation/multi_modal_davis_rand_drug.json

python mb_vae_dti/validating/perturbation.py \
  --dataset DAVIS --split split_rand --model multi_modal --branch drug --feature EMB-BiomedGraph \
  --steps 11 --output /root/MB-VAE-DTI/data/results/perturbation/multi_modal_davis_rand_drug_emb_biomedgraph.json
"""

import sys
sys.path.append("/root/MB-VAE-DTI")

from pathlib import Path
from typing import Dict, List, Optional, Literal, Any
import argparse
import json
from tqdm import tqdm

import torch
import pytorch_lightning as pl

from mb_vae_dti.training.datasets.datamodules import DTIDataModule
from mb_vae_dti.training.utils import ConfigManager
from mb_vae_dti.training.metrics.dti_metrics import RealDTIMetrics

# Reuse model setup utilities from the training script
from mb_vae_dti.training.run import (
    get_config_path,
    setup_model_baseline,
    setup_model_multi_modal,
)

Dataset = Literal["DAVIS", "KIBA"]
Split = Literal["split_rand", "split_cold"]
ModelType = Literal["baseline", "multi_modal"]
Branch = Literal["drug", "target"]


def _build_datamodule(dataset: Dataset, split: Split, batch_size: int = 32, num_workers: int = 4) -> DTIDataModule:
    split_type = split
    provenance_cols = ["in_DAVIS"] if dataset == "DAVIS" else ["in_KIBA"]
    dm = DTIDataModule(
        h5_path="/root/MB-VAE-DTI/data/input/dti.h5torch",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle_train=False,
        drop_last=False,
        split_type=split_type,
        provenance_cols=provenance_cols,
        load_in_memory=False,
        return_pyg=False,
    )
    dm.prepare_data()
    dm.setup("test")
    return dm


def _load_model(model_type: ModelType, dataset: Dataset, split: Split, feature_dims: Dict[str, Dict[str, int]]):
    CONFIG_ROOT = Path("/root/MB-VAE-DTI/mb_vae_dti/training/configs")
    split_str = "rand" if split == "split_rand" else "cold"
    config_path = CONFIG_ROOT / f"{model_type}/{dataset}_{split_str}.yaml"

    config_manager = ConfigManager()
    config = config_manager.load_config(config_path=config_path)

    # Select features as in training
    drug_feats = {name: dim for name, dim in feature_dims['drug'].items() if name in config.data.drug_features}
    target_feats = {name: dim for name, dim in feature_dims['target'].items() if name in config.data.target_features}

    if model_type == "baseline":
        model = setup_model_baseline(config, feature_dims, dataset)
    elif model_type == "multi_modal":
        model = setup_model_multi_modal(config, feature_dims, dataset)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.load_pretrained_weights(Path("/root/MB-VAE-DTI") / config.model.checkpoint_path)

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # Set deterministic seed for reproducibility
    pl.seed_everything(config.hardware.seed, workers=True)
    return model, config


def _expand_mean_to_batch(mean_vec: torch.Tensor, batch_shape: torch.Size) -> torch.Tensor:
    # mean_vec has shape of a single example, e.g., (D,) or (1, D) matching feature_tensor[0].shape
    # We need to expand to (B, ...) to match feature_tensor.shape
    if mean_vec.dim() == len(batch_shape) - 1:
        mean_vec = mean_vec.unsqueeze(0)
    return mean_vec.expand(batch_shape)


def _get_feature_names_for_branch(model, branch: Branch) -> List[str]:
    if branch == "drug":
        return list(model.hparams.drug_features.keys())
    else:
        return list(model.hparams.target_features.keys())


def _move_tensors_to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_tensors_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_tensors_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_tensors_to_device(v, device) for v in obj)
    return obj


def run_perturbation(
    dataset: Dataset,
    split: Split,
    model_type: ModelType,
    branch: Branch,
    steps: int = 11,
    feature: Optional[str] = None,
    output_json: Optional[Path] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    ):
    dm = _build_datamodule(dataset, split, batch_size=batch_size, num_workers=num_workers)

    # Compute test-set feature means once
    means = dm.get_test_feature_means(as_numpy=False)

    # Build model using config
    feature_dims = dm.get_feature_dims()
    model, config = _load_model(model_type, dataset, split, feature_dims)

    # Determine finetune score for metrics
    finetune_score = "Y_pKd" if dataset == "DAVIS" else "Y_KIBA"

    # Decide features to perturb
    available_feature_names = _get_feature_names_for_branch(model, branch)

    if feature is not None:
        if feature not in available_feature_names:
            raise ValueError(f"Feature '{feature}' not used by model in {branch} branch. Available: {available_feature_names}")
        feature_names = [feature]
    else:
        feature_names = available_feature_names

    device = next(model.parameters()).device

    # Steps from 0 to 1 inclusive
    alphas = [i / (steps - 1) for i in range(steps)] if steps > 1 else [1.0]
    print(f"Perturbing features with {steps} steps: {alphas}")

    results: Dict[str, Any] = {
        "dataset": dataset,
        "split": split,
        "model": model_type,
        "branch": branch,
        "feature": feature,
        "steps": steps,
        "metrics": []
    }

    torch.set_grad_enabled(False)

    for alpha in tqdm(alphas, desc="Perturbing features"):
        # Fresh metrics for each alpha
        metrics = RealDTIMetrics(prefix=f"{finetune_score}_")

        for batch in tqdm(dm.test_dataloader(), desc=f"Running model on alpha={alpha}"):
            # Move entire batch tensors to device first for device consistency
            batch = _move_tensors_to_device(batch, device)

            # Apply perturbation on the selected branch/features
            if branch == "drug":
                for fname in feature_names:
                    feat_tensor = batch['drug']['features'][fname]
                    mean_vec = means['drug']['features'][fname].to(device)
                    mean_batch = _expand_mean_to_batch(mean_vec, feat_tensor.shape)
                    batch['drug']['features'][fname] = (1.0 - alpha) * feat_tensor + alpha * mean_batch
            else:  # target
                for fname in feature_names:
                    feat_tensor = batch['target']['features'][fname]
                    mean_vec = means['target']['features'][fname].to(device)
                    mean_batch = _expand_mean_to_batch(mean_vec, feat_tensor.shape)
                    batch['target']['features'][fname] = (1.0 - alpha) * feat_tensor + alpha * mean_batch

            # Run model common step to get masked preds/targets
            batch_data, embedding_data, prediction_data, loss_data = model._common_step(batch)
            if prediction_data.dti_scores is not None and batch_data.dti_targets is not None:
                metrics.update(
                    prediction_data.dti_scores.detach().cpu(),
                    batch_data.dti_targets.detach().cpu()
                )

        # Compute and record metrics for this alpha
        computed = {k: float(v.detach().cpu()) for k, v in metrics.compute().items()}
        results["metrics"].append({
            "alpha": alpha,
            **computed
        })
        print({"alpha": alpha, **computed})

    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {output_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perturbation evaluation on test set without Trainer.")
    parser.add_argument("--dataset", type=str, choices=["DAVIS", "KIBA"], required=True)
    parser.add_argument("--split", type=str, choices=["split_rand", "split_cold"], required=True)
    parser.add_argument("--model", type=str, choices=["baseline", "multi_modal"], required=True)
    parser.add_argument("--branch", type=str, choices=["drug", "target"], required=True)
    parser.add_argument("--feature", type=str, default=None, help="Optional single feature to perturb (only for multi_modal). If omitted, perturbs all features in the branch.")
    parser.add_argument("--steps", type=int, default=11, help="Number of steps from 0 to 1 (inclusive). Default: 11")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output", type=str, default=None, help="Optional path to save JSON results")

    args = parser.parse_args()

    run_perturbation(
        dataset=args.dataset, 
        split=args.split, 
        model_type=args.model, 
        branch=args.branch,
        steps=args.steps,
        feature=args.feature,
        output_json=Path(args.output) if args.output else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

