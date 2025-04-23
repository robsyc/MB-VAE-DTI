"""
Script to generate multiview embeddings for drugs using Biomed-multiview model.
This script generates graph, image and text embeddings for SMILES strings.
See: https://github.com/BiomedSciAI/biomed-multi-view
"""

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import h5py
from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel
from bmfm_sm.core.data_modules.namespace import LateFusionStrategy
from bmfm_sm.predictive.data_modules.graph_finetune_dataset import Graph2dFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.image_finetune_dataset import ImageFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.text_finetune_dataset import TextFinetuneDataPipeline

# Add the parent directory to the Python path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import parse_args

# Get the current script directory and find the path to the model checkpoint
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_PATH = PROJECT_ROOT / "data" / "checkpoints" / "Biomed-smmv" / "biomed-smmv-base.pth"

class BiomedMultiViewMoleculeEncoder(nn.Module):
    def __init__(self):
        super(BiomedMultiViewMoleculeEncoder, self).__init__()
        # Initialize the pretrained model
        biomed_smmv_pretrained = SmallMoleculeMultiViewModel.from_pretrained(
            LateFusionStrategy.ATTENTIONAL,
            model_path=str(MODEL_PATH),
            huggingface=False,
            inference_mode=True
        )
        # Initialize the model subcomponents
        self.model_graph = biomed_smmv_pretrained.model_graph # output dim: 512
        self.model_image = biomed_smmv_pretrained.model_image # output dim: 512
        self.model_text = biomed_smmv_pretrained.model_text   # output dim: 768

    def collate_graph_data(self, graph_data_list):
        # Helper function for collating the individual processed graph samples into a batch
        collated = {}
        collated["node_num"] = torch.cat([sample['node_num'] for sample in graph_data_list])
        collated["node_data"] = torch.cat([sample['node_data'] for sample in graph_data_list])
        collated["edge_num"] = torch.cat([sample['edge_num'] for sample in graph_data_list])
        collated["edge_data"] = torch.cat([sample['edge_data'] for sample in graph_data_list])
        collated["edge_index"] = torch.cat([sample['edge_index'] for sample in graph_data_list], dim=1)

        max_node_num = max(collated["node_num"])
        collated["lap_eigvec"] = torch.cat(
                [
                    pad(i, (0, max_node_num - i.size(1)), value=float("0"))
                    for i in [sample["lap_eigvec"] for sample in graph_data_list]
                ]
            )
        return collated
    
    def forward_graph(self, smiles: list) -> np.ndarray:
        graph_data_list = [Graph2dFinetuneDataPipeline.smiles_to_graph_format(sm) for sm in smiles]
        graph_batch = self.collate_graph_data(graph_data_list)
        return self.model_graph(graph_batch)
    
    def forward_image(self, smiles: list) -> np.ndarray:
        img_data = [ImageFinetuneDataPipeline.smiles_to_image_format(sm)['img'].squeeze(0) for sm in smiles]
        return self.model_image(torch.stack(img_data, dim=0))
    
    def forward_text(self, smiles: list) -> np.ndarray:
        txt_data = [TextFinetuneDataPipeline.smiles_to_text_format(sm) for sm in smiles]
        tokenized_smiles_batch = pad_sequence([i['smiles.tokenized'].squeeze(0) for i in txt_data], batch_first=True)
        attention_mask_batch = pad_sequence([i['attention_mask'].squeeze(0) for i in txt_data], batch_first=True)
        return self.model_text(tokenized_smiles_batch, attention_mask_batch)

    def forward(self, smiles: list):
        graph_embeddings = self.forward_graph(smiles)
        image_embeddings = self.forward_image(smiles)
        text_embeddings = self.forward_text(smiles)
        return graph_embeddings, image_embeddings, text_embeddings

def batch_get_drug_embeddings(smiles_list, batch_size=32):
    """
    Get embeddings for a list of SMILES strings in batches.
    
    Args:
        smiles_list (list): List of SMILES strings
        batch_size (int): Size of batches to process
        
    Returns:
        dict: Dictionary with keys 'graph', 'image', 'text' containing the respective embeddings
    """
    # Initialize model once
    model = BiomedMultiViewMoleculeEncoder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_graph_embeddings = []
    all_image_embeddings = []
    all_text_embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Processing batches"):
        batch = smiles_list[i:i+batch_size]
        
        with torch.no_grad():
            embeddings = model(batch)
            
            # Extract the embeddings from the three views
            graph_emb = embeddings[0].cpu().numpy() # shape: (batch_size, 512)
            image_emb = embeddings[1].cpu().numpy() # shape: (batch_size, 512)
            text_emb = embeddings[2].cpu().numpy() # shape: (batch_size, 768)
            
            all_graph_embeddings.append(graph_emb)
            all_image_embeddings.append(image_emb)
            all_text_embeddings.append(text_emb)
    
    # Concatenate all batches
    if len(all_graph_embeddings) > 0:
        return {
            'graph': np.vstack(all_graph_embeddings),
            'image': np.vstack(all_image_embeddings),
            'text': np.vstack(all_text_embeddings)
        }
    else:
        return {
            'graph': np.array([]),
            'image': np.array([]),
            'text': np.array([])
        }

def process_embeddings_batch(h5_file_path, batch_size=32):
    """
    Process batches of SMILES strings and save their embeddings to the H5 file
    
    Args:
        h5_file_path: Path to the H5 file containing SMILES data
        batch_size: Number of SMILES to process in each batch
    """
    # Initialize model once
    model = BiomedMultiViewMoleculeEncoder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    with h5py.File(h5_file_path, 'r+') as f:
        # Access the data and ensure it exists
        if 'data' not in f:
            raise KeyError("HDF5 file must contain a 'data' dataset with SMILES strings")
        
        data = f['data']
        num_items = len(data)
        
        # Create embeddings group if it doesn't exist
        embeddings_group = f.require_group('embeddings')
        
        # Delete existing embeddings if they exist
        for embedding_name in ['graph', 'image', 'text']:
            if embedding_name in embeddings_group:
                del embeddings_group[embedding_name]
        
        # Process the first batch to determine embedding shape and dtype
        first_batch_size = min(batch_size, num_items)
        first_batch_data = [data[j].decode('utf-8') for j in range(first_batch_size)]
        
        # Get first batch embeddings
        with torch.no_grad():
            first_batch_embeddings = model(first_batch_data)
            graph_emb = first_batch_embeddings[0].cpu().numpy()
            image_emb = first_batch_embeddings[1].cpu().numpy()
            text_emb = first_batch_embeddings[2].cpu().numpy()
        
        # Create datasets with appropriate shapes
        graph_dataset = embeddings_group.create_dataset(
            'graph', shape=(num_items, graph_emb.shape[1]), dtype=graph_emb.dtype)
        image_dataset = embeddings_group.create_dataset(
            'image', shape=(num_items, image_emb.shape[1]), dtype=image_emb.dtype)
        text_dataset = embeddings_group.create_dataset(
            'text', shape=(num_items, text_emb.shape[1]), dtype=text_emb.dtype)
        
        # Save first batch
        graph_dataset[:first_batch_size] = graph_emb
        image_dataset[:first_batch_size] = image_emb
        text_dataset[:first_batch_size] = text_emb
        
        # Calculate number of batches for progress bar
        num_batches = (num_items + batch_size - 1) // batch_size
        
        print(f"Processing {num_items} items in {num_batches} batches...")
        
        # Process remaining batches
        with tqdm(total=num_items, desc="Generating multiview embeddings") as pbar:
            # Update progress bar for the first batch we already processed
            pbar.update(first_batch_size)
            
            # Process remaining batches
            for batch_start in range(first_batch_size, num_items, batch_size):
                # Calculate end of current batch
                batch_end = min(batch_start + batch_size, num_items)
                batch_size_actual = batch_end - batch_start
                
                # Get batch data
                batch_data = [data[j].decode('utf-8') for j in range(batch_start, batch_end)]
                
                # Process batch
                with torch.no_grad():
                    batch_embeddings = model(batch_data)
                    graph_emb = batch_embeddings[0].cpu().numpy()
                    image_emb = batch_embeddings[1].cpu().numpy()
                    text_emb = batch_embeddings[2].cpu().numpy()
                
                # Save batch
                graph_dataset[batch_start:batch_end] = graph_emb
                image_dataset[batch_start:batch_end] = image_emb
                text_dataset[batch_start:batch_end] = text_emb
                
                # Update progress bar
                pbar.update(batch_size_actual)
                
        # Add metadata about the embeddings
        for embedding_name, dataset in [
            ('graph', graph_dataset), 
            ('image', image_dataset), 
            ('text', text_dataset)
        ]:
            dataset.attrs['name'] = embedding_name
            dataset.attrs['shape'] = dataset.shape[1:]
            dataset.attrs['dtype'] = str(dataset.dtype)
        
        print(f"Successfully added multiview embeddings to {h5_file_path}")
        print(f"  - Graph embeddings shape: {graph_dataset.shape}")
        print(f"  - Image embeddings shape: {image_dataset.shape}")
        print(f"  - Text embeddings shape: {text_dataset.shape}")

def main():
    """Main function to handle file I/O and call the embedding functions"""
    args = parse_args()
    
    # Process the HDF5 file
    print(f"Processing embeddings for {args.input}")
    print(f"Using model at: {MODEL_PATH}")
    
    # Generate embeddings
    process_embeddings_batch(args.input)
    
if __name__ == "__main__":
    main()