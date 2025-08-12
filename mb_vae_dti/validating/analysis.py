"""
Analysis utilities for gridsearch experiment results.

This module provides functions to analyze the performance of different
hyperparameter combinations from gridsearch experiments.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def load_gridsearch_results(
    results_dir: Union[str, Path],
    pattern: str = "*_results.json"
    ) -> pd.DataFrame:
    """
    Load all gridsearch results from a directory into a pandas DataFrame.
    
    Args:
        results_dir: Directory containing JSON result files
        pattern: Glob pattern to match result files (default: "*_results.json")
        
    Returns:
        DataFrame with all results, with metadata columns flattened
    """
    results_dir = Path(results_dir)
    result_files = list(results_dir.glob(pattern))
    results = []

    logger.info(f"Loading {len(result_files)} result files from {results_dir}")
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to load {result_file}: {e}")

    df = pd.DataFrame(results)

    if 'val_metrics' in df.columns:
        val_metrics_df = pd.json_normalize(df['val_metrics'])
        for col in val_metrics_df.columns:
            df[col] = val_metrics_df[col]
        df = df.drop('val_metrics', axis=1)
    
    if 'test_metrics' in df.columns:
        test_metrics_df = pd.json_normalize(df['test_metrics'])
        for col in test_metrics_df.columns:
            df[col] = test_metrics_df[col]
        df = df.drop('test_metrics', axis=1)
    
    if "timing" in df.columns:
        timing_df = pd.json_normalize(df['timing'])
        for col in timing_df.columns:
            df[col] = timing_df[col]
        df = df.drop('timing', axis=1)
    
    if "config" in df.columns:
        config_df = pd.json_normalize(df['config'])
        for col in config_df.columns:
            df[f"config.{col}"] = config_df[col]
        df = df.drop('config', axis=1)
    
    return df

def get_test_averages(df: pd.DataFrame) -> pd.DataFrame:
    test_cols = [col for col in df.columns if col.startswith("test/")]
    test_df = df[test_cols]
    test_df = test_df.mean(axis=0)
    return test_df


def set_plotting_style():
    """Set the default plotting style for consistent visualizations."""
    plt.style.use('tableau-colorblind10')  # Use Matplotlib style sheet
    sns.set_style("whitegrid")  # Apply seaborn style on top
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['axes.labelpad'] = 10


# --- Perturbation analysis helpers ---

_ALLOWED_METRICS = {"ci", "mse", "pearson", "r2", "rmse"}


def _get_metric_prefix(dataset: str) -> str:
    dataset = dataset.upper()
    if dataset == "DAVIS":
        return "Y_pKd"
    if dataset == "KIBA":
        return "Y_KIBA"
    raise ValueError(f"Unsupported dataset: {dataset}")

def _get_metric_key(dataset: str, metric: str) -> str:
    metric = metric.lower()
    if metric not in _ALLOWED_METRICS:
        raise ValueError(f"Unsupported metric: {metric}. Choose one of {_ALLOWED_METRICS}")
    return f"{_get_metric_prefix(dataset)}_{metric}"

def _iter_matching_perturbation_files(
    results_dir: Union[str, Path],
    dataset: str,
    split: str,
    model: str,
    ) -> List[Path]:
    """
    Find all perturbation result JSON files in a directory that match the given
    dataset, split and model by inspecting the file contents (robust to filenames).
    """
    results_dir = Path(results_dir)
    candidates = list(results_dir.glob("*.json"))
    matched: List[Path] = []
    for path in candidates:
        try:
            with open(path, "r") as f:
                obj = json.load(f)
            if not isinstance(obj, dict):
                continue
            if obj.get("dataset", "").upper() != dataset.upper():
                continue
            if obj.get("split", "") != split:
                continue
            if obj.get("model", "") != model:
                continue
            matched.append(path)
        except Exception as e:
            logger.warning(f"Skipping {path} due to error: {e}")
    return matched

def load_perturbation_metric_df(
    results_dir: Union[str, Path] = "/root/MB-VAE-DTI/data/results/perturbation",
    *,
    dataset: str,
    split: str,
    model: str,
    metric: str,
    ) -> pd.DataFrame:
    """
    Load and flatten perturbation results into a DataFrame for a specific metric.
    Metrics: ci, mse, pearson, r2, rmse
    
    Returns a DataFrame with columns:
      - dataset, split, model, branch, feature, steps
      - alpha: float
      - value: metric value at that alpha
    """
    metric_key = _get_metric_key(dataset, metric)
    files = _iter_matching_perturbation_files(results_dir, dataset, split, model)

    records: List[Dict[str, Any]] = []
    for path in files:
        try:
            with open(path, "r") as f:
                obj = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}")
            continue

        meta_dataset = obj.get("dataset")
        meta_split = obj.get("split")
        meta_model = obj.get("model")
        branch = obj.get("branch")
        feature = obj.get("feature")
        steps = obj.get("steps")
        metrics_list = obj.get("metrics", [])

        for entry in metrics_list:
            alpha = entry.get("alpha")
            if metric_key not in entry:
                # Metric not present in this file; skip gracefully
                continue
            value = entry[metric_key]
            records.append({
                "dataset": meta_dataset,
                "split": meta_split,
                "model": meta_model,
                "branch": branch,
                "feature": feature,
                "steps": steps,
                "alpha": alpha,
                "value": value,
            })

    df = pd.DataFrame.from_records(records)
    return df

def plot_perturbation_overview(
    model: str,
    metric: str,
    results_dir: Union[str, Path] = "/root/MB-VAE-DTI/data/results/perturbation",
    datasets: List[str] = ("DAVIS", "KIBA"),
    splits: List[str] = ("split_rand", "split_cold"),
    ):
    """Create a 2x2 overview of perturbation curves for a model/metric.

    - Solid lines: entity-level (drug vs target)
    - Dotted lines: per-feature (for multi_modal runs)

    Args:
        model: "baseline" or "multi_modal"
        metric: one of {ci, mse, pearson, r2, rmse}
        results_dir: directory containing perturbation JSONs
        datasets: iterable of datasets to include
        splits: iterable of splits to include

    Returns:
        The created Matplotlib Figure
    """
    set_plotting_style()

    results_dir = Path(results_dir)
    pretty_metric = metric.upper() if metric != "r2" else "R2"

    # Consistent branch colors
    branch_colors = {"drug": "#1f77b4", "target": "#2ca02c"}
    # Short, human-friendly feature labels and consistent linestyles across subplots
    feature_label_map = {
        "EMB-BiomedGraph": "graph",
        "EMB-BiomedImg": "img",
        "EMB-BiomedText": "text",
        "EMB-ESM": "ESM",
        "EMB-NT": "NT",
    }
    feature_linestyle_map = {
        "graph": ":",   # fine dotted
        "img": "--",     # dashed
        "text": "-.",    # dash-dot
        "ESM": "--",
        "NT": "-.",
    }
 
    # Arrange: rows = splits (rand on top, cold below); columns = datasets (DAVIS left, KIBA right)
    fig, axes = plt.subplots(len(splits), len(datasets), sharex=True, sharey=False, figsize=(12, 8))
    if len(splits) == 1 and len(datasets) == 1:
        axes = [[axes]]  # normalize indexing

    # Track y-values per row (split) to set dynamic limits
    row_yvalues: List[List[float]] = [[] for _ in range(len(splits))]

    for i, split in enumerate(splits):
        for j, dataset in enumerate(datasets):
            ax = axes[i][j]

            df = load_perturbation_metric_df(
                results_dir=results_dir,
                dataset=dataset,
                split=split,
                model=model,
                metric=metric,
            )

            if df.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                ax.grid(True, alpha=0.3)
                continue

            # Entity-level (feature is None): solid lines per branch
            entity_df = df[df["feature"].isna()].copy()
            if not entity_df.empty:
                for branch_name, sub_df in entity_df.groupby("branch"):
                    sub_df = sub_df.sort_values("alpha")
                    row_yvalues[i].extend(sub_df["value"].tolist())
                    ax.plot(
                        sub_df["alpha"],
                        sub_df["value"],
                        label=branch_name,
                        color=branch_colors.get(branch_name, None),
                        marker="o",
                    )

            # Per-feature (for multi_modal): dotted lines, color by branch
            feature_df = df[~df["feature"].isna()].copy()
            if not feature_df.empty:
                # Shorten feature names and assign consistent linestyles
                feature_df["feature_short"] = feature_df["feature"].map(feature_label_map).fillna(feature_df["feature"])
                feature_df["series"] = feature_df.apply(lambda r: f"{r['branch']}:{r['feature_short']}", axis=1)
                for series_name, sub_df in feature_df.groupby("series"):
                    branch_name, feature_short = series_name.split(":", 1)
                    sub_df = sub_df.sort_values("alpha")
                    row_yvalues[i].extend(sub_df["value"].tolist())
                    ax.plot(
                        sub_df["alpha"],
                        sub_df["value"],
                        label=series_name,
                        color=branch_colors.get(branch_name, None),
                        linestyle=feature_linestyle_map.get(feature_short, ":"),
                        linewidth=1.5,
                        alpha=0.9,
                        marker=None,
                    )

            # Labels: y only on first column, x only on bottom row; y shows only metric name
            if j == 0:
                ax.set_ylabel(pretty_metric)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", left=False, labelleft=False)

            if i == len(splits) - 1:
                ax.set_xlabel("alpha (towards feature-wise test mean)")
            else:
                ax.set_xlabel("")

            ax.grid(True, alpha=0.3)

    # Column titles (datasets) across the top
    def _dataset_title(name: str) -> str:
        return "Davis (pKd)" if name.upper() == "DAVIS" else "KIBA"

    for j, dataset in enumerate(datasets):
        axes[0][j].set_title(_dataset_title(dataset), loc="center", y=1.02)

    # Dynamic y-limits per row (utilize vertical space better)
    for i in range(len(splits)):
        if not row_yvalues[i]:
            continue
        y_min = min(row_yvalues[i])
        y_max = max(row_yvalues[i])
        if y_max == y_min:
            y_max = y_min + 1e-3
        pad = 0.08 * (y_max - y_min)
        y_lower = y_min - pad
        y_upper = y_max + pad
        for ax in axes[i]:
            ax.set_ylim(y_lower, y_upper)

    # Vertical split labels on the right of the rightmost subplots
    split_title_map = {"split_rand": "Random split", "split_cold": "Cold-drug split"}
    for i, split in enumerate(splits):
        ax_right = axes[i][-1]
        ax_right.text(
            1.02,
            0.5,
            split_title_map.get(split, split.replace("split_", "")),
            rotation="vertical",
            va="center",
            ha="left",
            transform=ax_right.transAxes,
            fontsize=14,
        )

    # Legend inside bottom-left subplot
    handles, labels = [], []
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                labels.append(label)
                handles.append(handle)
    if handles:
        axes[-1][0].legend(handles, labels, title=None, loc="lower left", frameon=True)

    fig.tight_layout(rect=[0, 0, 0.95, 0.98])
    return fig