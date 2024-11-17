# Adopted from the BenevolentAI team and the MolBERT Repository - https://github.com/BenevolentAI/MolBERT

# Operations for Text Modality (SMILES codes)
import copy

import numpy as np
import torch
from fuse.data.ops.op_base import OpBase
from fuse.utils.ndict import NDict
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

import bmfm_sm.core.data_modules.namespace as ns
from bmfm_sm.core.data_modules.fuse_ops.text_tokenizer import TextTokenizer

# Molformer-specific global variables
max_length = 500
mod_length = 42
avg_length = 66
mlm_probability = 0.15
tokenizer = TextTokenizer()


class OpGenerateSmilesFingerprint(OpBase):
    def __call__(self, sample_dict: NDict, key_in, key_out) -> NDict:
        """Converts smiles code to a RDKit fingerprint."""
        smiles_codes = sample_dict[key_in]
        molecules = Chem.MolFromSmiles(smiles_codes)
        fingerprints = Chem.RDKFingerprint(molecules)

        sample_dict[key_out] = fingerprints

        return sample_dict


class OpGenerateRDKitDescriptors(OpBase):
    def __call__(
        self,
        sample_dict: NDict,
        key_in,
        key_out=ns.FIELD_DATA_LIGAND_FINGERPRINT,
        descriptor_list=None,
    ) -> NDict:
        """
        Generates the molecular descriptors for the molecule (based on its smiles codes) and saves these under the label key
        By default, will generate all the RDKit supported descriptors but a subset list can be passed in as well.
        """
        smiles_code = sample_dict[key_in]
        molecule = Chem.MolFromSmiles(smiles_code)

        if descriptor_list == None:
            descriptor_list = sorted([x[0] for x in Descriptors._descList])

        desc_calc = MolecularDescriptorCalculator(descriptor_list)
        mol_values = desc_calc.CalcDescriptors(molecule)

        for i in range(len(descriptor_list)):
            sample_dict[f"{key_out}.{descriptor_list[i]}"] = mol_values[i]

        return sample_dict


# Utility function for permutation generation
def standardise(smiles: str, canonicalise: bool | None = False) -> str | None:
    """
    Standardise a SMILES string if valid (canonical + kekulized)
    Returns: standard version the SMILES if valid, None otherwise.
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    except Exception as e:
        # invalid?
        print(f'Chem.MolFromSmiles failed smiles="{smiles}" error={e}')
        return None

    if mol is None:
        # invalid?
        print(f'Chem.MolFromSmiles failed smiles="{smiles}"')
        return None

    flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_CLEANUP
    Chem.SanitizeMol(mol, flags, catchErrors=True)

    if canonicalise:
        # bug where permuted smiles are not canonicalised to the same form. This is fixed by round tripping SMILES
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        if mol is None:
            print(f'Chem.MolFromSmiles failed after sanitization smiles="{smiles}"')
            return None

    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=canonicalise)
    except (ValueError, RuntimeError):
        print(f"SMILES failed Kekulization! {smiles}")
        return None

    return smiles


class OpGenerateSmilesPermutation(OpBase):
    def __call__(self, sample_dict: NDict, key_in, key_out=None) -> NDict:
        """Generate a valid permutation of a given SMILES code."""
        smiles_code = sample_dict[key_in]

        try:
            molecule = Chem.MolFromSmiles(smiles_code, sanitize=False)
        except Exception as e:
            print(f'Chem.MolFromSmiles failed smiles="{smiles_code}" error={e}')
            sample_dict[key_out] = None
            return sample_dict

        if molecule is None:
            print(f'Chem.MolFromSmiles failed smiles="{smiles_code}"')
            sample_dict[key_out] = None
            return sample_dict

        ans = list(range(molecule.GetNumAtoms()))
        np.random.shuffle(ans)

        new_smiles = Chem.MolToSmiles(
            Chem.RenumberAtoms(molecule, ans), canonical=False
        )
        new_smiles = standardise(new_smiles)

        sample_dict[key_out] = new_smiles

        return sample_dict


class OpGenerateCanonicalizedSmiles(OpBase):
    def __call__(self, sample_dict: NDict, key_in, key_out) -> NDict:
        """Generates canonicalized version of the SMILES code."""
        smiles_codes = sample_dict[key_in]
        canonical_smiles = Chem.CanonSmiles(smiles_codes)
        sample_dict[key_out] = canonical_smiles
        return sample_dict


class OpGenerateTokenizedSmiles(OpBase):
    def __call__(self, sample_dict: NDict, key_in, key_out) -> NDict:
        """Generates tokenized version of the SMILES code."""
        global tokenizer
        smiles_codes = sample_dict[key_in]
        tokens = tokenizer._tokenize(smiles_codes)
        tokens = tokenizer.build_inputs_with_special_tokens(tokens)
        tens = torch.tensor([tokenizer._convert_token_to_id(word) for word in tokens])
        sample_dict[key_out] = tens
        return sample_dict


class OpGenerateSmilesMasks(OpBase):
    """Generates masks for the Smiles dataset."""

    def mask_tokens(self, inputs, special_tokens_mask):
        """Prepare masked tokens inputs/labels for masked language modeling acccording to Molformer paper: 80% MASK, 10% random, 10% original."""
        global tokenizer, mlm_probability
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.size(), mlm_probability)
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.size(), 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = tokenizer.mask_token_id
        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.size(), 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            tokenizer.vocab_size, labels.size(), dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def pack_tensors(self, tokens):
        # array =  torch.nn.utils.rnn.pad_sequence([tokens])
        global tokenizer
        array = copy.deepcopy(tokens)
        special_token_mask = [
            [
                1 if x in [tokenizer.cls_token_id, tokenizer.sep_token_id] else 0
                for x in stuff
            ]
            for stuff in [array]
        ]
        masked_array, masked_labels = self.mask_tokens(array, special_token_mask[0])
        return masked_array, masked_labels  # , lengths

    def __call__(self, sample_dict: NDict, key_in, key_out) -> NDict:
        smiles = sample_dict[key_in]
        input, labels = self.pack_tensors(smiles)
        sample_dict[ns.FIELD_DATA_LIGAND_MLM_INPUT] = input
        sample_dict[ns.FIELD_DATA_LIGAND_MLM_LABEL] = labels

        return sample_dict
