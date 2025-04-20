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
import argparse
from pathlib import Path
from tqdm import tqdm
from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel
from bmfm_sm.core.data_modules.namespace import LateFusionStrategy
from bmfm_sm.predictive.data_modules.graph_finetune_dataset import Graph2dFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.image_finetune_dataset import ImageFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.text_finetune_dataset import TextFinetuneDataPipeline

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
        
    def forward(self, smiles: list):
        tokenized_smiles_list = []
        attention_mask_list = []
        graph_data_list = []
        
        image_tensors = []

        for sm in smiles:
            # Prepare image and text data in batch format
            img_data = ImageFinetuneDataPipeline.smiles_to_image_format(sm)
            image_tensors.append(img_data['img'].squeeze(0)) # Remove extra batch dimension if present

            txt_data = TextFinetuneDataPipeline.smiles_to_text_format(sm)
            tokenized_smiles_list.append(txt_data['smiles.tokenized'].squeeze(0))
            attention_mask_list.append(txt_data['attention_mask'].squeeze(0))

            # Run the graph model on individual smiles
            graph_data = Graph2dFinetuneDataPipeline.smiles_to_graph_format(sm)
            graph_data_list.append(graph_data)

        # Run the image and text models on the batched data
        image_batch = torch.stack(image_tensors, dim=0)
        tokenized_smiles_batch = pad_sequence(tokenized_smiles_list, batch_first=True)
        attention_mask_batch = pad_sequence(attention_mask_list, batch_first=True)
        graph_batch = self.collate_graph_data(graph_data_list)
        
        image_emb = self.model_image(image_batch)
        text_emb = self.model_text(tokenized_smiles_batch, attention_mask_batch)
        graph_emb = self.model_graph(graph_batch)

        return [graph_emb, image_emb, text_emb]

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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate multiview embeddings for drug SMILES strings")
    parser.add_argument("--input", required=True, help="TXT file with one SMILES string per line or a single SMILES string")
    parser.add_argument("--output", required=True, help="Output numpy file for embeddings")
    return parser.parse_args()

def main():
    """Main function to handle file I/O and call the embedding functions"""
    args = parse_args()
    
    # Check if input is a file or a single sequence
    if os.path.isfile(args.input):
        with open(args.input, 'r') as f:
            smiles_strings = [line.strip() for line in f if line.strip()]
    else:
        # Treat input as a single SMILES string
        smiles_strings = [args.input]
    
    print(f"Processing {len(smiles_strings)} SMILES strings...")
    print(f"Using model at: {MODEL_PATH}")
    
    # Generate embeddings
    embeddings_dict = batch_get_drug_embeddings(smiles_strings)
    
    # Create an output directory if needed (strip .npy extension)
    output_base = os.path.splitext(args.output)[0]
    
    # Save individual embeddings
    for embedding_type, embedding_data in embeddings_dict.items():
        output_file = f"{output_base}_{embedding_type}.npy"
        np.save(output_file, embedding_data)
        print(f"Saved {embedding_type} embeddings with shape {embedding_data.shape} to {output_file}")
    
if __name__ == "__main__":
    main()