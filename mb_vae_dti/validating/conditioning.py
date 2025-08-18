import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from rdkit import Chem
import pandas as pd
from mb_vae_dti.loading import *
from mb_vae_dti.processing.split import *

import json
import numpy as np
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
import os
from datetime import datetime

from external.ESPF.script import get_target_fingerprint
from mb_vae_dti.training.utils import ConfigManager
from mb_vae_dti.training.modules.full import FullDTIModel

logger = logging.getLogger(__name__)

class ConditionalDrugGenerator:
    """
    Conditional drug generation pipeline that optimizes drug latent representations 
    for positive interactions with specific protein targets.
    
    Pipeline steps:
    1. Encode protein targets using ESP fingerprints
    2. Process targets through target encoders
    3. Sample initial drug latents from VAE
    4. Optimize latents for positive Y prediction using gradients
    5. Generate molecules via diffusion sampling
    6. Convert to SMILES and save with tracking metadata
    """
    
    def __init__(
        self, 
        model: FullDTIModel,
        device: torch.device,
        config: Dict,
        output_dir: str = "data/results/generated_conditional",
        max_batch_size: int = 512
    ):
        self.model = model
        self.device = device
        self.config = config
        self.output_dir = output_dir
        self.max_batch_size = max_batch_size
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup detailed logging for the generation process."""
        log_file = os.path.join(self.output_dir, f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.info("Conditional drug generation pipeline initialized")

    def encode_targets(self, target_sequences: List[str]) -> torch.Tensor:
        """
        Step 1: Encode protein targets using ESP fingerprint function.
        
        Args:
            target_sequences: List of protein amino acid sequences
            
        Returns:
            target_fingerprints: Tensor of shape [n_targets, 4170] (ESP fingerprint size)
        """
        logger.info(f"Encoding {len(target_sequences)} protein targets using ESP fingerprints")
        
        fingerprints = []
        for i, seq in enumerate(target_sequences):
            try:
                fp = get_target_fingerprint(seq)
                fingerprints.append(fp)
                logger.debug(f"Encoded target {i+1}/{len(target_sequences)}")
            except Exception as e:
                logger.error(f"Failed to encode target {i+1}: {e}")
                # Use zero vector as fallback
                fingerprints.append(np.zeros(4170, dtype=np.float32))
        
        # Convert to tensor
        target_fingerprints = torch.tensor(np.array(fingerprints), dtype=torch.float32, device=self.device)
        logger.info(f"Target encoding complete. Shape: {target_fingerprints.shape}")
        
        return target_fingerprints

    def process_targets(self, target_fingerprints: torch.Tensor) -> torch.Tensor:
        """
        Step 2: Process target fingerprints through target encoders.
        
        Args:
            target_fingerprints: Tensor of shape [n_targets, 4170]
            
        Returns:
            processed_targets: Tensor of shape [n_targets, embedding_dim]
        """
        logger.info("Processing targets through target encoders")
        
        with torch.no_grad():
            # Use the target branch of the model
            target_features = [target_fingerprints]  # ESP fingerprints
            
            # Encode targets
            target_embeddings = [
                self.model.target_encoders[feat_name](feat)
                for feat_name, feat in zip(self.model.hparams.target_features, target_features)
            ]
            
            # Aggregate if multiple features (in our case, just ESP)
            if len(target_embeddings) == 1:
                processed_targets = target_embeddings[0]
            else:
                processed_targets = self.model.target_aggregator(target_embeddings)
        
        logger.info(f"Target processing complete. Shape: {processed_targets.shape}")
        return processed_targets

    def sample_initial_latents(
        self, 
        n_targets: int, 
        n_samples_per_target: int = 16,
        reference_smiles: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step 3: Sample initial drug latents using hybrid strategy (random + learned).
        
        Args:
            n_targets: Number of unique targets
            n_samples_per_target: Number of latent samples per target (split 50/50)
            reference_smiles: SMILES from training set for learned sampling
            
        Returns:
            initial_latents: Tensor of shape [n_targets * n_samples_per_target, embedding_dim]
            sampling_metadata: Tensor of shape [n_targets * n_samples_per_target] 
                              (0 = random, 1 = learned)
        """
        logger.info(f"Sampling {n_samples_per_target} initial latents per target ({n_targets} targets)")
        logger.info("Using hybrid strategy: 50% random sampling, 50% learned distribution")
        
        total_samples = n_targets * n_samples_per_target
        embedding_dim = self.model.hparams.embedding_dim
        
        # Split samples 50/50 between random and learned
        n_random_per_target = n_samples_per_target // 2
        n_learned_per_target = n_samples_per_target - n_random_per_target
        
        total_random = n_targets * n_random_per_target
        total_learned = n_targets * n_learned_per_target
        
        logger.info(f"Random samples: {total_random}, Learned samples: {total_learned}")
        
        # Sample random latents from standard normal prior
        random_latents = torch.randn(
            total_random, embedding_dim, 
            device=self.device, dtype=torch.float32
        )
        
        # Sample learned latents from VAE distribution
        if reference_smiles is not None and len(reference_smiles) > 0:
            learned_latents = self._sample_from_learned_distribution(
                total_learned, reference_smiles
            )
        else:
            logger.warning("No reference SMILES provided, using random sampling for 'learned' samples")
            learned_latents = torch.randn(
                total_learned, embedding_dim,
                device=self.device, dtype=torch.float32
            )
        
        # Create metadata to track sampling strategy
        random_metadata = torch.zeros(total_random, device=self.device)  # 0 = random
        learned_metadata = torch.ones(total_learned, device=self.device)  # 1 = learned
        
        # Interleave random and learned samples per target for balanced distribution
        initial_latents = []
        sampling_metadata = []
        
        for target_idx in range(n_targets):
            # Get samples for this target
            random_start = target_idx * n_random_per_target
            random_end = random_start + n_random_per_target
            learned_start = target_idx * n_learned_per_target
            learned_end = learned_start + n_learned_per_target
            
            target_random = random_latents[random_start:random_end]
            target_learned = learned_latents[learned_start:learned_end]
            target_random_meta = random_metadata[random_start:random_end]
            target_learned_meta = learned_metadata[learned_start:learned_end]
            
            # Combine and shuffle for this target
            target_latents = torch.cat([target_random, target_learned], dim=0)
            target_meta = torch.cat([target_random_meta, target_learned_meta], dim=0)
            
            # Shuffle to mix random and learned samples
            perm = torch.randperm(target_latents.shape[0])
            target_latents = target_latents[perm]
            target_meta = target_meta[perm]
            
            initial_latents.append(target_latents)
            sampling_metadata.append(target_meta)
        
        # Concatenate all targets
        initial_latents = torch.cat(initial_latents, dim=0)
        sampling_metadata = torch.cat(sampling_metadata, dim=0)
        
        logger.info(f"Hybrid sampling complete. Shape: {initial_latents.shape}")
        logger.info(f"Random samples: {(sampling_metadata == 0).sum().item()}")
        logger.info(f"Learned samples: {(sampling_metadata == 1).sum().item()}")
        
        return initial_latents, sampling_metadata

    def _sample_from_learned_distribution(
        self, 
        n_samples: int, 
        reference_smiles: List[str]
    ) -> torch.Tensor:
        """Sample from VAE's learned latent distribution using reference molecules."""
        logger.debug(f"Sampling {n_samples} latents from learned VAE distribution")
        
        # Randomly sample reference molecules to avoid bias
        if len(reference_smiles) > n_samples:
            sampled_smiles = np.random.choice(reference_smiles, size=n_samples, replace=False).tolist()
        else:
            # Use all available and repeat if necessary
            sampled_smiles = reference_smiles * (n_samples // len(reference_smiles) + 1)
            sampled_smiles = sampled_smiles[:n_samples]
        
        # Convert SMILES to Morgan fingerprints
        from external.MorganFP.script import get_drug_fingerprint
        
        fingerprints = []
        valid_count = 0
        
        for smiles in sampled_smiles:
            try:
                fp = get_drug_fingerprint(smiles)
                fingerprints.append(fp)
                valid_count += 1
            except Exception as e:
                logger.debug(f"Failed to process SMILES '{smiles}': {e}")
                # Use zero vector as fallback
                fingerprints.append(np.zeros(2048, dtype=np.float32))
        
        logger.debug(f"Successfully processed {valid_count}/{len(sampled_smiles)} reference molecules")
        
        # Convert to tensor
        fp_tensor = torch.tensor(np.array(fingerprints), dtype=torch.float32, device=self.device)
        
        # Encode through drug branch to get latent samples
        with torch.no_grad():
            drug_features = [fp_tensor]
            
            # Encode through drug encoders
            drug_embeddings = [
                self.model.drug_encoders[feat_name](feat)
                for feat_name, feat in zip(self.model.hparams.drug_features, drug_features)
            ]
            
            # Aggregate if multiple features
            if len(drug_embeddings) == 1:
                drug_embedding = drug_embeddings[0]
            else:
                drug_embedding = self.model.drug_aggregator(drug_embeddings)
            
            # Sample from VAE distribution using reparameterization trick
            latent_samples, mu, logvar = self.model.drug_kl_head(drug_embedding)
            
            # Add small amount of noise for diversity while staying on manifold
            noise_scale = 0.05  # Small noise to maintain manifold proximity
            latent_samples += torch.randn_like(latent_samples) * noise_scale
        
        logger.debug(f"Generated {latent_samples.shape[0]} learned latent samples")
        return latent_samples

    def _flatten_grouped_stats(self, grouped_df):
        """
        Helper method to flatten pandas groupby aggregation results for JSON serialization.
        Converts tuple column names to string format.
        """
        # Flatten the multi-level column names
        grouped_df.columns = ['_'.join(col).strip() for col in grouped_df.columns.values]
        return grouped_df.to_dict('index')

    def optimize_latents_for_interaction(
        self,
        initial_latents: torch.Tensor,
        sampling_metadata: torch.Tensor,
        processed_targets: torch.Tensor,
        n_samples_per_target: int = 8,
        n_optimization_steps: int = 100,
        learning_rate: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step 4: Optimize drug latents for positive Y prediction using gradient ascent.
        
        Args:
            initial_latents: [n_targets * n_samples_per_target, embedding_dim]
            sampling_metadata: [n_targets * n_samples_per_target] - sampling strategy (0=random, 1=learned)
            processed_targets: [n_targets, embedding_dim]
            n_samples_per_target: Number of initial samples per target
            n_optimization_steps: Number of gradient steps per optimization
            learning_rate: Learning rate for optimization
            
        Returns:
            optimized_latents: [total_optimized, embedding_dim]
            metadata: [total_optimized, 4] - (target_idx, sample_idx, sampling_strategy, final_score)
        """
        logger.info("Starting gradient-based latent optimization for positive interactions")
        
        n_targets = processed_targets.shape[0]
        total_optimized = n_targets * n_samples_per_target
        
        optimized_latents = []
        metadata = []
        
        for target_idx in range(n_targets):
            logger.info(f"Optimizing for target {target_idx + 1}/{n_targets}")
            target_embedding = processed_targets[target_idx:target_idx+1]  # [1, embedding_dim]
            
            for sample_idx in range(n_samples_per_target):
                # Get the initial latent for this target-sample combination
                latent_idx = target_idx * n_samples_per_target + sample_idx
                initial_latent = initial_latents[latent_idx:latent_idx+1]  # [1, embedding_dim]
                sampling_strategy = sampling_metadata[latent_idx].item()  # 0=random, 1=learned
                
                # Create optimizable copy
                latent = initial_latent.clone()
                latent.requires_grad_(True)
                
                # Setup optimizer
                optimizer = Adam([latent], lr=learning_rate)
                
                best_latent = latent.clone()
                best_score = float('-inf')
                
                # Optimization loop
                for step in range(n_optimization_steps):
                    optimizer.zero_grad()
                    
                    # Forward pass through fusion and DTI head
                    with torch.enable_grad():
                        # Expand target to match batch size
                        target_batch = target_embedding.expand(latent.shape[0], -1)
                        
                        # Fusion
                        fused_embedding, _ = self.model.fusion(latent, target_batch)
                        
                        # DTI prediction - get binary interaction score
                        dti_scores = self.model.dti_head(fused_embedding)
                        y_logit = dti_scores['Y']  # Binary interaction logit
                        
                        # We want to maximize the probability of positive interaction
                        # Use sigmoid to convert logit to probability, then optimize
                        y_prob = torch.sigmoid(y_logit)
                        loss = -y_prob.mean()  # Negative because we want to maximize
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Track best latent
                    current_score = y_prob.item()
                    if current_score > best_score:
                        best_score = current_score
                        best_latent = latent.clone().detach()
                    
                    # Log progress occasionally
                    if step % 20 == 0:
                        logger.debug(f"Target {target_idx}, Sample {sample_idx}"
                                    f"Step {step}: Score = {current_score:.4f}")
                    
                # Store optimized latent and metadata
                optimized_latents.append(best_latent)
                metadata.append([target_idx, sample_idx, sampling_strategy, best_score])
                
                sampling_type = "learned" if sampling_strategy == 1 else "random"
                logger.info(f"Optimization complete. Final score: {best_score:.4f} (from {sampling_type} sampling)")
        
        # Concatenate all optimized latents
        optimized_latents = torch.cat(optimized_latents, dim=0)  # [total_optimized, embedding_dim]
        metadata = torch.tensor(metadata, device=self.device)  # [total_optimized, 4]
        
        logger.info(f"Latent optimization complete. Generated {optimized_latents.shape[0]} optimized latents")
        logger.info(f"Score statistics - Mean: {metadata[:, 3].mean():.4f}, "
                   f"Std: {metadata[:, 3].std():.4f}, Min: {metadata[:, 3].min():.4f}, Max: {metadata[:, 3].max():.4f}")
        
        # Log statistics by sampling strategy
        random_mask = metadata[:, 2] == 0
        learned_mask = metadata[:, 2] == 1
        
        if random_mask.any():
            random_scores = metadata[random_mask, 3]
            logger.info(f"Random sampling scores - Mean: {random_scores.mean():.4f}, "
                       f"Std: {random_scores.std():.4f}, Min: {random_scores.min():.4f}, Max: {random_scores.max():.4f}")
        
        if learned_mask.any():
            learned_scores = metadata[learned_mask, 3]
            logger.info(f"Learned sampling scores - Mean: {learned_scores.mean():.4f}, "
                       f"Std: {learned_scores.std():.4f}, Min: {learned_scores.min():.4f}, Max: {learned_scores.max():.4f}")
        
        return optimized_latents, metadata

    def generate_molecules_batch(
        self,
        optimized_latents: torch.Tensor,
        metadata: torch.Tensor,
        n_molecules_per_latent: int = 16
    ) -> List[Dict]:
        """
        Step 5: Generate molecules using diffusion sampling with batch management.
        
        Args:
            optimized_latents: [n_latents, embedding_dim]
            metadata: [n_latents, 4] - tracking info (target_idx, sample_idx, sampling_strategy, score)
            n_molecules_per_latent: Number of molecules to generate per latent
            
        Returns:
            results: List of dicts with generation results and metadata
        """
        logger.info(f"Generating {n_molecules_per_latent} molecules per optimized latent")
        
        n_latents = optimized_latents.shape[0]
        results = []
        
        # Calculate batch size for diffusion (memory management)
        # Each latent will generate n_molecules_per_latent samples
        # We need to ensure total_samples <= max_batch_size
        effective_batch_size = min(self.max_batch_size // n_molecules_per_latent, n_latents)
        
        logger.info(f"Processing {n_latents} latents in batches of {effective_batch_size}")
        
        for batch_start in tqdm(range(0, n_latents, effective_batch_size), desc="Generating molecules"):
            batch_end = min(batch_start + effective_batch_size, n_latents)
            batch_latents = optimized_latents[batch_start:batch_end]
            batch_metadata = metadata[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start}-{batch_end}")
            
            try:
                # Generate molecules for this batch
                with torch.no_grad():
                    # sample_batch expects [batch_size, embedding_dim] and returns 
                    # List[List[Optional[Chem.Mol]]] with shape [batch_size][n_molecules_per_latent]
                    generated_molecules = self.model.sample_batch(
                        drug_embeddings=batch_latents,
                        num_samples_per_embedding=n_molecules_per_latent
                    )
                
                # Process results for this batch
                for i, (latent_metadata, mol_group) in enumerate(zip(batch_metadata, generated_molecules)):
                    target_idx, sample_idx, sampling_strategy, final_score = latent_metadata.cpu().numpy()
                    
                    for diff_idx, mol in enumerate(mol_group):
                        result = {
                            'target_idx': int(target_idx),
                            'sample_idx': int(sample_idx),
                            'sampling_strategy': int(sampling_strategy),  # 0=random, 1=learned
                            'diff_idx': diff_idx,
                            'optimization_score': float(final_score),
                            'molecule': mol,
                            'smiles': None,
                            'valid': False
                        }
                        
                        # Convert to SMILES if valid
                        if mol is not None:
                            try:
                                smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
                                mol = Chem.MolFromSmiles(smiles) if smiles is not None else None
                                result['smiles'] = smiles
                                result['valid'] = True if mol is not None else False
                            except Exception as e:
                                logger.debug(f"Failed to convert molecule to SMILES: {e}")
                        
                        results.append(result)
                
                logger.debug(f"Batch {batch_start}-{batch_end} complete")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_start}-{batch_end}: {e}")
                # Add empty results for this batch to maintain indexing
                for i in range(batch_end - batch_start):
                    latent_metadata = batch_metadata[i]
                    target_idx, sample_idx, sampling_strategy, final_score = latent_metadata.cpu().numpy()
                    
                    for diff_idx in range(n_molecules_per_latent):
                        result = {
                            'target_idx': int(target_idx),
                            'sample_idx': int(sample_idx),
                            'sampling_strategy': int(sampling_strategy),
                            'diff_idx': diff_idx,
                            'optimization_score': float(final_score),
                            'molecule': None,
                            'smiles': None,
                            'valid': False,
                            'error': str(e)
                        }
                        results.append(result)
        
        # Calculate statistics
        total_generated = len(results)
        valid_molecules = sum(1 for r in results if r['valid'])
        validity_rate = valid_molecules / total_generated if total_generated > 0 else 0
        
        logger.info(f"Molecule generation complete. Total: {total_generated}, "
                   f"Valid: {valid_molecules}, Validity rate: {validity_rate:.2%}")
        
        return results

    def save_results(self, results: List[Dict], target_sequences: List[str]):
        """
        Step 6: Save generation results with comprehensive tracking.
        
        Args:
            results: List of generation results from generate_molecules_batch
            target_sequences: Original target sequences for reference
        """
        logger.info("Saving generation results")
        
        # Create comprehensive results dataframe
        df_data = []
        for result in results:
            row = {
                'target_idx': result['target_idx'],
                'sample_idx': result['sample_idx'],
                'sampling_strategy': result['sampling_strategy'],
                'sampling_type': 'learned' if result['sampling_strategy'] == 1 else 'random',
                'diff_idx': result['diff_idx'],
                'optimization_score': result['optimization_score'],
                'smiles': result['smiles'],
                'valid': result['valid'],
                'error': result.get('error', None)
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Save main results
        results_file = os.path.join(self.output_dir, "conditional_generation_results.csv")
        df.to_csv(results_file, index=False)
        logger.info(f"Results saved to {results_file}")
        
        # Save target sequences for reference
        target_file = os.path.join(self.output_dir, "target_sequences.txt")
        with open(target_file, 'w') as f:
            for i, seq in enumerate(target_sequences):
                f.write(f"Target_{i}: {seq}\n")
        
        # Save valid SMILES separately for easy access
        valid_smiles = df[df['valid'] == True]['smiles'].tolist()
        smiles_file = os.path.join(self.output_dir, "generated_smiles.txt")
        with open(smiles_file, 'w') as f:
            for smiles in valid_smiles:
                f.write(f"{smiles}\n")
        
        # Save summary statistics
        summary = {
            'total_generated': len(results),
            'valid_molecules': df['valid'].sum(),
            'validity_rate': df['valid'].mean(),
            'unique_smiles': df[df['valid']]['smiles'].nunique(),
            'optimization_score_stats': {
                'mean': df['optimization_score'].mean(),
                'std': df['optimization_score'].std(),
                'min': df['optimization_score'].min(),
                'max': df['optimization_score'].max()
            },
            'sampling_strategy_stats': {
                'random_count': (df['sampling_strategy'] == 0).sum(),
                'learned_count': (df['sampling_strategy'] == 1).sum(),
                'random_validity': df[df['sampling_strategy'] == 0]['valid'].mean(),
                'learned_validity': df[df['sampling_strategy'] == 1]['valid'].mean(),
                'random_score_mean': df[df['sampling_strategy'] == 0]['optimization_score'].mean(),
                'learned_score_mean': df[df['sampling_strategy'] == 1]['optimization_score'].mean()
            },
            'per_target_stats': self._flatten_grouped_stats(df.groupby('target_idx').agg({
                'valid': ['sum', 'count', 'mean'],
                'optimization_score': ['mean', 'std', 'max']
            })),
            'per_sampling_strategy_stats': self._flatten_grouped_stats(df.groupby('sampling_type').agg({
                'valid': ['sum', 'count', 'mean'],
                'optimization_score': ['mean', 'std', 'max']
            }))
        }
        
        summary_file = os.path.join(self.output_dir, "generation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Summary statistics saved to {summary_file}")
        logger.info(f"Generated {summary['total_generated']} total molecules, "
                   f"{summary['valid_molecules']} valid ({summary['validity_rate']:.2%})")

    def run_full_pipeline(
        self,
        target_sequences: List[str],
        n_samples_per_target: int = 16,
        n_molecules_per_latent: int = 16,
        optimization_config: Optional[Dict] = None,
        reference_smiles: Optional[List[str]] = None
    ) -> Dict:
        """
        Run the complete conditional drug generation pipeline.
        
        Args:
            target_sequences: List of protein amino acid sequences
            n_samples_per_target: Number of initial VAE samples per target
            n_molecules_per_latent: Number of diffusion samples per optimized latent
            optimization_config: Optional config for gradient optimization
            reference_smiles: SMILES from training set for learned sampling
            
        Returns:
            summary: Dictionary with pipeline results and statistics
        """
        logger.info("Starting full conditional drug generation pipeline")
        logger.info(f"Targets: {len(target_sequences)}, "
                   f"VAE samples/target: {n_samples_per_target}, "
                   f"Molecules/latent: {n_molecules_per_latent}")
        
        total_expected = (len(target_sequences) * n_samples_per_target * n_molecules_per_latent)
        logger.info(f"Expected total molecules: {total_expected}")
        
        # Set default optimization config
        if optimization_config is None:
            optimization_config = {
                'n_optimization_steps': 100,
                'learning_rate': 0.01
            }
        
        try:
            # Step 1: Encode targets
            target_fingerprints = self.encode_targets(target_sequences)
            
            # Step 2: Process targets
            processed_targets = self.process_targets(target_fingerprints)
            
            # Step 3: Sample initial latents (hybrid strategy)
            initial_latents, sampling_metadata = self.sample_initial_latents(
                len(target_sequences), n_samples_per_target, reference_smiles
            )
            
            # Step 4: Optimize latents
            optimized_latents, metadata = self.optimize_latents_for_interaction(
                initial_latents, sampling_metadata, processed_targets,
                n_samples_per_target,
                **optimization_config
            )
            
            # Step 5: Generate molecules
            results = self.generate_molecules_batch(
                optimized_latents, metadata, n_molecules_per_latent
            )
            
            # Step 6: Save results
            self.save_results(results, target_sequences)
            
            # Compile summary
            summary = {
                'pipeline_completed': True,
                'n_targets': len(target_sequences),
                'n_samples_per_target': n_samples_per_target,
                'n_molecules_per_latent': n_molecules_per_latent,
                'total_expected': total_expected,
                'total_generated': len(results),
                'valid_molecules': sum(1 for r in results if r['valid']),
                'validity_rate': sum(1 for r in results if r['valid']) / len(results) if results else 0,
                'output_directory': self.output_dir
            }
            
            logger.info("Pipeline completed successfully!")
            logger.info(f"Summary: {summary}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


# Load data and setup (existing code)
df = pd.read_csv("data/processed/dti.csv")
df = add_cold_drug_split(df)

df_test_true = df[((df["split_cold"] == "test") | (df["split_cold"] == "val")) & (df["Y"] == True)]
df_train_true = df[(df["split_cold"] == "train") & (df["Y"] == True)]

def get_heavy_atom_count(smiles):
    """Get heavy atom count from SMILES string"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return mol.GetNumHeavyAtoms()
    except:
        return None

unique_drugs = df_test_true['Drug_SMILES'].unique()
heavy_atom_counts = {smiles: get_heavy_atom_count(smiles) for smiles in unique_drugs}
valid_drugs = {smiles for smiles, count in heavy_atom_counts.items() 
               if count is not None and 22 <= count <= 35}

df_filtered = df_test_true[df_test_true['Drug_SMILES'].isin(valid_drugs)]

aa_counts = df_filtered["Target_AA"].value_counts()
df_sub = df_filtered[df_filtered["Target_AA"].isin(aa_counts[aa_counts.between(64, 128)].index)]

df_train_true_sub = df_train_true[df_train_true["Target_AA"].isin(df_sub["Target_AA"].unique())]
targets = df_train_true_sub["Target_AA"].value_counts().sort_values(ascending=True).head(32).index

df = df[df["Target_AA"].isin(targets)]

# Extract unique target sequences
target_sequences = targets.tolist()

# Get reference SMILES from cold training set (avoid leakage)
df_train_cold = df[df["split_cold"] == "train"]
reference_smiles = df_train_cold["Drug_SMILES"].unique().tolist()
logger.info(f"Found {len(reference_smiles)} unique reference SMILES from training set")

# Load model
config_path = "./mb_vae_dti/training/configs/full/pretrain_cold.yaml"
ckpt_path = "./data/checkpoints/full_cold.ckpt"
mol_stats_path = "./data/processed/molecular_statistics.json"

config_manager = ConfigManager()
config = config_manager.load_config(config_path=config_path)

with open(mol_stats_path, "r") as f:
    mol_stats = json.load(f)
    key = "drugs_cold"
    dataset_statistics = {
        "general": mol_stats["general"],
        "dataset": mol_stats["datasets"][key]
    }

drug_feats = {"FP-Morgan": 2048}
target_feats = {"FP-ESP": 4170}

model = FullDTIModel(
    phase="train",
    finetune_score=None,
    learning_rate=config.training.learning_rate,
    weight_decay=config.training.weight_decay,
    scheduler=config.training.scheduler,
    weights=config.loss.weights,
    dti_weights=config.loss.dti_weights,
    diff_weights=config.loss.diff_weights,
    contrastive_temp=config.loss.contrastive_temp,
    drug_features=drug_feats,
    target_features=target_feats,
    embedding_dim=config.model.embedding_dim,
    hidden_dim=config.model.hidden_dim,
    factor=config.model.factor,
    n_layers=config.model.n_layers,
    activation=config.model.activation,
    dropout=config.model.dropout,
    bias=config.model.bias,
    encoder_type=config.model.encoder_type,
    aggregator_type=config.model.aggregator_type,

    diffusion_steps=config.model.diffusion_steps,
    sample_every_val=config.model.sample_every_val,
    val_samples_per_embedding=config.model.val_samples_per_embedding,
    test_samples_per_embedding=config.model.test_samples_per_embedding,
    graph_transformer_kwargs=OmegaConf.to_container(config.model.get('graph_transformer_kwargs', {})),
    dataset_infos=dataset_statistics,
)

model.load_pretrained_weights(ckpt_path)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
pl.seed_everything(config.hardware.seed, workers=True)

# Initialize and run the conditional generation pipeline
if __name__ == "__main__":
    # Initialize the generator
    generator = ConditionalDrugGenerator(
        model=model,
        device=device,
        config=config,
        output_dir="data/results/generated_conditional",
        max_batch_size=512
    )
    
    # Run the full pipeline
    summary = generator.run_full_pipeline(
        target_sequences=target_sequences,
        n_samples_per_target=32,
        n_molecules_per_latent=16,
        optimization_config={
            'n_optimization_steps': 100,
            'learning_rate': 0.01
        },
        reference_smiles=reference_smiles
    )
    
    print(f"Pipeline completed! Summary: {summary}")