"""
Script to embed protein sequences using ESM-C 6B (using API) -> size 2560 (80 layers)
See: https://github.com/evolutionaryscale/esm
Alternatively, we could use `esmc-600m-2024-12` (smaller open source model with 36 layers)
"""

import os
import dotenv
import json
import argparse
import sys
import hashlib
import pickle

import numpy as np
from pathlib import Path
import tqdm

from esm.sdk import client
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinError,
    LogitsConfig,
    LogitsOutput,
    ProteinType,
)
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, List, Dict, Optional

# Constants
BATCH_SIZE = 100  # Default batch size for processing

dotenv.load_dotenv()
ESM_TOKEN = os.getenv("ESM_TOKEN")

model = client(
    model="esmc-6b-2024-12", url="https://forge.evolutionaryscale.ai", token=ESM_TOKEN
)

EMBEDDING_CONFIG = LogitsConfig(
    sequence=True, return_hidden_states=True, ith_hidden_layer=80
) # See: https://github.com/evolutionaryscale/esm/issues/176#issuecomment-2784146081

# Cache directory path
CACHE_FILE = Path(os.path.dirname(os.path.abspath(__file__))) / "embedding_cache.pkl"

def load_cache() -> Dict[str, np.ndarray]:
    """Load the embedding cache from disk"""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
    return {}

def save_cache(cache: Dict[str, np.ndarray]) -> None:
    """Save the embedding cache to disk"""
    # Ensure cache directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
    except Exception as e:
        print(f"Error saving cache: {e}")

# Initialize the cache
_EMBEDDING_CACHE = load_cache()

def esm_embed_sequence(
    sequence: str, model: ESM3InferenceClient = model, use_cache: bool = True
) -> np.ndarray:
    """
    Embed a protein sequence using the ESM-C (6B with 80 layers) model.
    Returns the mean of the hidden states over the residues.
    
    Args:
        sequence: Protein sequence as string
        model: ESM model client
        use_cache: Whether to use caching (default: True)
    
    Returns:
        numpy.ndarray: The embedding vector
    """
    global _EMBEDDING_CACHE
    
    # Check cache if enabled
    if use_cache and sequence in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[sequence]
    
    # Generate embedding if not in cache
    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)
    output = model.logits(protein_tensor, EMBEDDING_CONFIG)
    emb = output.hidden_states.mean(dim=-2) # average over residues
    embedding = emb.squeeze().numpy()
    
    # Update cache if enabled
    if use_cache:
        _EMBEDDING_CACHE[sequence] = embedding
        # Save cache periodically (every 10 new entries)
        if len(_EMBEDDING_CACHE) % 10 == 0:
            save_cache(_EMBEDDING_CACHE)
    
    return embedding

def esm_batch_embed(
    inputs: Sequence[ProteinType], model: ESM3InferenceClient = model, use_cache: bool = True
) -> List[np.ndarray]:
    """Forge supports auto-batching. So batch_embed() is as simple as running a collection
    of embed calls in parallel using asyncio.
    """
    global _EMBEDDING_CACHE
    
    # Filter sequences that are already in cache
    if use_cache:
        to_compute = [seq for seq in inputs if seq not in _EMBEDDING_CACHE]
        cached_results = [_EMBEDDING_CACHE[seq] for seq in inputs if seq in _EMBEDDING_CACHE]
        
        if not to_compute:
            return cached_results
    else:
        to_compute = inputs
    
    # Compute embeddings for sequences not in cache
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(esm_embed_sequence, protein, model, False) for protein in to_compute
        ]
        computed_results = []
        for i, future in enumerate(futures):
            try:
                result = future.result()
                computed_results.append(result)
                # Update cache for computed results
                if use_cache:
                    _EMBEDDING_CACHE[to_compute[i]] = result
            except Exception as e:
                computed_results.append(ESMProteinError(500, str(e)))
    
    # Save updated cache
    if use_cache and computed_results:
        save_cache(_EMBEDDING_CACHE)
    
    # Combine cached and computed results in original order
    if use_cache:
        results = []
        cache_idx = 0
        compute_idx = 0
        for seq in inputs:
            if seq in _EMBEDDING_CACHE and cache_idx < len(cached_results):
                results.append(cached_results[cache_idx])
                cache_idx += 1
            else:
                results.append(computed_results[compute_idx])
                compute_idx += 1
        return results
    else:
        return computed_results

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate ESM embeddings for protein sequences")
    parser.add_argument("--input", required=True, help="TXT file with one amino acid sequence per line e.g. dti_aa.txt")
    parser.add_argument("--output", required=True, help="Output numpy file for embeddings")
    return parser.parse_args()

def main():
    """Main function to handle file I/O and call the embedding functions"""
    args = parse_args()
    use_cache = True
    
    # Print cache statistics
    if use_cache:
        print(f"Using embedding cache: {len(_EMBEDDING_CACHE)} entries found")
    
    # Check if input is a file or a single sequence
    if os.path.isfile(args.input):
        with open(args.input, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]
    else:
        # Treat input as a single sequence string
        sequences = [args.input]
    
    # Generate embeddings in batches
    if len(sequences) > 1:
        embeddings = []
        batches = [sequences[i:i+BATCH_SIZE] for i in range(0, len(sequences), BATCH_SIZE)]
        print(f"Batch processing {len(batches)} batches with size {BATCH_SIZE}...")
        
        for batch in tqdm.tqdm(batches, desc="Processing batches"):
            batch_embeddings = esm_batch_embed(batch, use_cache=use_cache)
            embeddings.extend(batch_embeddings)
    else:
        print(f"Processing single sequence...")
        embeddings = esm_embed_sequence(sequences[0], use_cache=use_cache)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings)
    
    # Save embeddings
    np.save(args.output, embeddings_array)
    print(f"Saved embeddings with shape {embeddings_array.shape} to {args.output}")
    
    # Save final cache
    if use_cache:
        save_cache(_EMBEDDING_CACHE)
        print(f"Updated cache with {len(_EMBEDDING_CACHE)} entries")

if __name__ == "__main__":
    main()