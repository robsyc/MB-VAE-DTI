"""
Script to generate multiview embeddings for drugs using Biomed-multi-view model.
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
from pathlib import Path
from typing import List, Dict, Any

from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel
from bmfm_sm.core.data_modules.namespace import LateFusionStrategy
from bmfm_sm.predictive.data_modules.graph_finetune_dataset import Graph2dFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.image_finetune_dataset import ImageFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.text_finetune_dataset import TextFinetuneDataPipeline

# Add the parent directory to the Python path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
# Import necessary functions from utils
from utils import parse_args, add_embeddings_to_hdf5

# --- Constants ---
MODEL_NAME = "biomed-smmv-base"
# Define specific embedding names for each view
EMBEDDING_NAME_GRAPH = "EMB-BiomedGraph"
EMBEDDING_NAME_IMAGE = "EMB-BiomedImg"
EMBEDDING_NAME_TEXT = "EMB-BiomedText"
BATCH_SIZE = 32 # Adjust based on GPU memory for the largest view (likely text)

# Get the current script directory and find the path to the model checkpoint
SCRIPT_DIR = Path(__file__).resolve().parent
# Construct path robustly
MODEL_PATH = SCRIPT_DIR.joinpath(f"{MODEL_NAME}.pth")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_PATH}")

# --- Model Definition ---
class BiomedMultiViewMoleculeEncoder(nn.Module):
    def __init__(self, model_path: str):
        super(BiomedMultiViewMoleculeEncoder, self).__init__()
        print(f"Loading pretrained Biomed-Multi-View model from: {model_path}...")
        # Initialize the pretrained model
        biomed_smmv_pretrained = SmallMoleculeMultiViewModel.from_pretrained(
            LateFusionStrategy.ATTENTIONAL, # Strategy doesn't matter for inference on sub-models
            model_path=model_path,
            huggingface=False,
            inference_mode=True
        )
        # Initialize the model subcomponents
        self.model_graph = biomed_smmv_pretrained.model_graph # output dim: 512
        self.model_image = biomed_smmv_pretrained.model_image # output dim: 512
        self.model_text = biomed_smmv_pretrained.model_text   # output dim: 768
        print("Biomed-Multi-View sub-models loaded.")

    # Helper for graph batching (static method as it doesn't depend on model state)
    @staticmethod
    def collate_graph_data(graph_data_list: List[Dict]) -> Dict:
        # Helper function for collating the individual processed graph samples into a batch
        collated = {}
        # Handle cases where a list might be empty
        if not graph_data_list:
            # Return structure expected by model_graph, potentially with zero tensors
            # This needs careful handling based on model_graph's expectations for empty batches
            # For now, assume non-empty batches are handled by caller
            raise ValueError("collate_graph_data received an empty list.")
            
        collated["node_num"] = torch.cat([sample['node_num'] for sample in graph_data_list])
        collated["node_data"] = torch.cat([sample['node_data'] for sample in graph_data_list])
        collated["edge_num"] = torch.cat([sample['edge_num'] for sample in graph_data_list])
        collated["edge_data"] = torch.cat([sample['edge_data'] for sample in graph_data_list])
        collated["edge_index"] = torch.cat([sample['edge_index'] for sample in graph_data_list], dim=1)

        # Find max_node_num within the current batch for padding lap_eigvec
        current_max_node_num = max(sample['node_num'].item() for sample in graph_data_list) # Safely get item
        
        # Pad lap_eigvec correctly
        lap_eigvecs = [sample["lap_eigvec"] for sample in graph_data_list]
        collated["lap_eigvec"] = torch.cat(
            [
                pad(eigvec, (0, current_max_node_num - eigvec.size(1)), value=0.0) 
                for eigvec in lap_eigvecs
            ], 
            dim=0 # Concatenate along batch dimension
        )
        return collated
    
    def forward_graph(self, smiles_list: List[str], device: torch.device) -> List[np.ndarray]:
        if not smiles_list:
            return []
        graph_data_list = [Graph2dFinetuneDataPipeline.smiles_to_graph_format(sm) for sm in smiles_list]
        # Filter out None results if smiles_to_graph_format can fail
        valid_graph_data = [g for g in graph_data_list if g is not None]
        if not valid_graph_data:
            print("Warning: No valid graphs generated for this batch.")
            # Return list of zero arrays matching expected output shape/type
            # Need to know graph_model output size (512)
            return [np.zeros(512, dtype=np.float32) for _ in smiles_list]
            
        graph_batch = self.collate_graph_data(valid_graph_data)
        # Move collated data to device
        graph_batch = {k: v.to(device) for k, v in graph_batch.items()}
        with torch.no_grad():
            embeddings = self.model_graph(graph_batch)
        return [emb.cpu().numpy() for emb in embeddings]
    
    def forward_image(self, smiles_list: List[str], device: torch.device) -> List[np.ndarray]:
        if not smiles_list:
            return []
        img_data_list = [ImageFinetuneDataPipeline.smiles_to_image_format(sm) for sm in smiles_list]
        # Filter out None results
        valid_img_data = [img['img'].squeeze(0) for img in img_data_list if img is not None]
        if not valid_img_data:
            print("Warning: No valid images generated for this batch.")
            # Return list of zero arrays matching expected output shape/type (512)
            return [np.zeros(512, dtype=np.float32) for _ in smiles_list]
            
        img_batch = torch.stack(valid_img_data, dim=0).to(device)
        with torch.no_grad():
            embeddings = self.model_image(img_batch)
        return [emb.cpu().numpy() for emb in embeddings]
    
    def forward_text(self, smiles_list: List[str], device: torch.device) -> List[np.ndarray]:
        if not smiles_list:
            return []
        txt_data_list = [TextFinetuneDataPipeline.smiles_to_text_format(sm) for sm in smiles_list]
        # Filter out None results
        valid_txt_data = [txt for txt in txt_data_list if txt is not None]
        if not valid_txt_data:
            print("Warning: No valid text data generated for this batch.")
            # Return list of zero arrays matching expected output shape/type (768)
            return [np.zeros(768, dtype=np.float32) for _ in smiles_list]
            
        tokenized_smiles = [i['smiles.tokenized'].squeeze(0) for i in valid_txt_data]
        attention_masks = [i['attention_mask'].squeeze(0) for i in valid_txt_data]
        
        tokenized_batch = pad_sequence(tokenized_smiles, batch_first=True, padding_value=0).to(device) # Assuming 0 is pad token ID
        attention_mask_batch = pad_sequence(attention_masks, batch_first=True, padding_value=0).to(device) # Pad mask with 0
        
        with torch.no_grad():
            embeddings = self.model_text(tokenized_batch, attention_mask_batch)
        return [emb.cpu().numpy() for emb in embeddings]

# --- Global Model Initialization ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
MODEL = BiomedMultiViewMoleculeEncoder(str(MODEL_PATH))
MODEL = MODEL.to(DEVICE)
MODEL.eval()

# --- View-Specific Batch Processing Functions ---
def batch_process_graph(smiles_list: List[str]) -> List[np.ndarray]:
    """Processes a batch of SMILES for graph embeddings."""
    return MODEL.forward_graph(smiles_list, DEVICE)

def batch_process_image(smiles_list: List[str]) -> List[np.ndarray]:
    """Processes a batch of SMILES for image embeddings."""
    return MODEL.forward_image(smiles_list, DEVICE)

def batch_process_text(smiles_list: List[str]) -> List[np.ndarray]:
    """Processes a batch of SMILES for text embeddings."""
    return MODEL.forward_text(smiles_list, DEVICE)


# --- Main Execution Logic ---
def main():
    """Main function to parse args and call the HDF5 processing utility for each view."""
    args = parse_args() # Gets only --input
    print(f"Processing SMILES from HDF5 file: {args.input}")
    
    # Base metadata
    model_metadata = {
        'model_name': MODEL_NAME,
        # 'model_path': str(MODEL_PATH)
        # Add more metadata if needed, e.g., sub-model specifics if accessible
    }
    
    # --- Process Graph Embeddings ---
    add_embeddings_to_hdf5(
        h5_file_path=args.input,
        embedding_name=EMBEDDING_NAME_GRAPH,
        batch_processing_function=batch_process_graph,
        batch_size=BATCH_SIZE,
        model_metadata={**model_metadata, 'view': 'graph'} # Add view-specific metadata
    )
    
    # --- Process Image Embeddings ---
    add_embeddings_to_hdf5(
        h5_file_path=args.input,
        embedding_name=EMBEDDING_NAME_IMAGE,
        batch_processing_function=batch_process_image,
        batch_size=BATCH_SIZE,
        model_metadata={**model_metadata, 'view': 'image'}
    )
    
    # --- Process Text Embeddings ---
    add_embeddings_to_hdf5(
        h5_file_path=args.input,
        embedding_name=EMBEDDING_NAME_TEXT,
        batch_processing_function=batch_process_text,
        batch_size=BATCH_SIZE,
        model_metadata={**model_metadata, 'view': 'text'}
    )

    print("Finished processing all views for Biomed-Multi-View.")

if __name__ == "__main__":
    main()