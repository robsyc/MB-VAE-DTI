import pandas as pd
import numpy as np
import os

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from subword_nmt.apply_bpe import BPE
import codecs

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel
from bmfm_sm.core.data_modules.namespace import LateFusionStrategy
from bmfm_sm.predictive.data_modules.graph_finetune_dataset import Graph2dFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.image_finetune_dataset import ImageFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.text_finetune_dataset import TextFinetuneDataPipeline

from transformers import T5Tokenizer, T5EncoderModel
import re

import h5torch
from typing import Literal


HUGGING_FACE = True
DATASET_PATH = "./data/dataset/"
MODEL_PATH = "./data/model_saves/"
espf_folder = "./data/ESPF"
assert os.path.exists(DATASET_PATH), "Dataset directory does not exist."
assert any([f.endswith(".h5t") for f in os.listdir(DATASET_PATH)]), "No .h5t files found in dataset directory."
if not HUGGING_FACE:
    assert os.path.exists(MODEL_PATH), "Model directory does not exist."
else:
    print("Using HuggingFace model")
assert os.path.exists(espf_folder), "ESPF directory does not exist."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## FINGERPRINTS

def get_drug_fingerprint(s: str) -> np.ndarray:
    # See: https://greglandrum.github.io/rdkit-blog/posts/2023-01-18-fingerprint-generator-tutorial.html
    molecule = Chem.MolFromSmiles(s)
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return np.array(mfpgen.GetFingerprint(molecule))

vocab_path = espf_folder + '/codes_protein.txt'
bpe_codes_protein = codecs.open(vocab_path)
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
sub_csv = pd.read_csv(espf_folder + '/subword_units_map_protein.csv')

idx2word_p = sub_csv['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

def get_target_fingerprint(s: str) -> np.ndarray:
    # See: https://github.com/kexinhuang12345/ESPF
    t = pbpe.process_line(s).split()
    i = [words2idx_p[i] for i in t]
    v = np.zeros(len(idx2word_p), )
    v[i] = 1
    return v


## EMBEDDINGS

class BiomedMultiViewMoleculeEncoder(nn.Module):
    def __init__(
        self,
        inference_mode: bool = True,
        huggingface: bool = True
    ):
        super(BiomedMultiViewMoleculeEncoder, self).__init__()
        # Initialize the pretrained model
        if huggingface:
            path = 'ibm/biomed.sm.mv-te-84m'
        else:
            path = MODEL_PATH + "Biomed-smmv/biomed-smmv-base.pth"
        biomed_smmv_pretrained = SmallMoleculeMultiViewModel.from_pretrained(
            LateFusionStrategy.ATTENTIONAL,
            model_path=path,
            huggingface=huggingface,
            inference_mode=inference_mode
        )
        # Initialize the model subcomponents
        self.model_graph = biomed_smmv_pretrained.model_graph # output dim: 512
        self.model_image = biomed_smmv_pretrained.model_image # output dim: 512
        self.model_text = biomed_smmv_pretrained.model_text   # output dim: 768

        # Helper function for collating the individual processed graph samples:
    def collate_graph_data(self, graph_data_list):
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
        graph_emb = []

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

class T5ProstTargetEncoder(nn.Module):
    def __init__(
            self, 
            huggingface = True,
            cap_seq = False
            ):
        super(T5ProstTargetEncoder, self).__init__()
        if huggingface:
            path = "Rostlab/ProstT5"
        else:
            path = MODEL_PATH + "ProstT5"
        self.tokenizer = T5Tokenizer.from_pretrained(path, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(path).to(device)
        # only GPUs support half-precision (float16) currently; if you want to run on CPU use full-precision (float32) (not recommended, much slower)
        self.model.float() if device.type=='cpu' else self.model.half()
    
    def process(self, sequences):
        sequences = [sequence[:self.AA_SEQ_CAP] for sequence in sequences]
        sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))).upper() for sequence in sequences]
        sequences = ["<AA2fold> " + s for s in sequences]
        return sequences
    
    def tokenize(self, sequences):
        ids = self.tokenizer.batch_encode_plus(
            sequences,
            add_special_tokens=True,
            padding="longest",
            return_tensors='pt'
        )
        ids = {key: tensor.to(device) for key, tensor in ids.items()}
        return ids

    def forward(self, sequences):
        X = self.process(sequences)
        ids = self.tokenize(X)
        outputs = self.model(
            input_ids=ids['input_ids'],
            attention_mask=ids['attention_mask']
        ).last_hidden_state # (batch_size, seq_len, hidden_dim)
        outputs = outputs.mean(dim=1) # (batch_size, hidden_dim)
        return outputs

####################################################################################################################

def add_embeddings(name: Literal["BindingDB_Kd", "DAVIS", "KIBA"]) -> None:
    """
    Adds fingerprint and embedding columns to the specified dataset's existing h5torch file.
    """
    f = h5torch.File(DATASET_PATH + name + ".h5t", "a")
    drug_smiles = f["0/Drug_SMILES"][:]
    target_AA = f["1/Target_seq"][:]
    target_DNA = f["1/Target_seq_DNA"][:]

    drug_fingerprints = np.empty((len(drug_smiles), 2048), dtype=np.float32)
    for i, s in enumerate(drug_smiles):
        s = s.decode("utf-8")
        fp = get_drug_fingerprint(s)
        drug_fingerprints[i] = fp
    f.register(drug_fingerprints, mode="N-D", axis=0, name="Drug_fp", dtype_save="float32", dtype_load="float32")
    f.save()

    drug_emb_graph = np.empty((len(drug_smiles), 512), dtype=np.float32)
    drug_emb_image = np.empty((len(drug_smiles), 512), dtype=np.float32)
    drug_emb_text = np.empty((len(drug_smiles), 768), dtype=np.float32)
    model = BiomedMultiViewMoleculeEncoder(huggingface=HUGGING_FACE)
    for i, s in enumerate(drug_smiles):
        s = s.decode("utf-8")
        emb = model([s])    # [graph, image, text] TODO: maybe do batch processing?!
        drug_emb_graph[i] = emb[0]
        drug_emb_image[i] = emb[1]
        drug_emb_text[i] = emb[2]
    f.register(drug_emb_graph, mode="N-D", axis=0, name="Drug_emb_graph", dtype_save="float32", dtype_load="float32")
    f.register(drug_emb_image, mode="N-D", axis=0, name="Drug_emb_image", dtype_save="float32", dtype_load="float32")
    f.register(drug_emb_text, mode="N-D", axis=0, name="Drug_emb_text", dtype_save="float32", dtype_load="float32")
    f.save()

    target_fingerprints = np.empty((len(target_AA), 4170), dtype=np.float32)
    for i, s in enumerate(target_AA):
        s = s.decode("utf-8")
        fp = get_target_fingerprint(s)
        target_fingerprints[i] = fp
    f.register(target_fingerprints, mode="N-D", axis=1, name="Target_fp", dtype_save="float32", dtype_load="float32")
    f.save()

    target_emb_T5 = np.empty((len(target_AA), 1024), dtype=np.float32)
    # target_emb_ESM = np.empty((len(target_AA), 1280), dtype=np.float32)
    # target_emb_DNA = np.empty((len(target_DNA), 1280), dtype=np.float32)

    # f.register(drug_embeddings_1, mode="N-D", axis=0, name="drug_embeddings_1", dtype_save="float32", dtype_load="float32")