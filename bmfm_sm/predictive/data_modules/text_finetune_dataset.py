import numpy as np
import torch
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.utils.ndict import NDict

import bmfm_sm.core.data_modules.namespace as ns
from bmfm_sm.core.data_modules.fuse_ops.ops_smiles import OpGenerateTokenizedSmiles
from bmfm_sm.core.data_modules.fuse_ops.text_tokenizer import TextTokenizer
from bmfm_sm.predictive.data_modules.mpp_finetune_dataset import MPPFinetuneDataset


class TextFinetuneDataPipeline(MPPFinetuneDataset):
    def __init__(self, data_dir, dataset_args, stage="train"):
        self.tokenizer = TextTokenizer()
        super().__init__(data_dir=data_dir, dataset_args=dataset_args, stage=stage)

    def get_ops_pipeline(self) -> PipelineDefault:
        superclass_ops = super().get_ops_pipeline()._ops_and_kwargs

        text_ops = [
            # Generating tokenized smiles from the smiles codes
            (
                OpGenerateTokenizedSmiles(),
                {"key_in": ns.FIELD_SMILES, "key_out": ns.FIELD_TOKENIZED_SMILES},
            ),
        ]

        return PipelineDefault("all_ops", superclass_ops + text_ops)

    def collate_fn(self, data: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokenized_smiles = [i[ns.FIELD_TOKENIZED_SMILES] for i in data]
        attention_mask = []
        for i in data:
            att_mask_arr = [1] * len(i[ns.FIELD_TOKENIZED_SMILES])
            attention_mask.append(torch.tensor(att_mask_arr))

        tokenized_smiles_padded = torch.nn.utils.rnn.pad_sequence(
            tokenized_smiles,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = torch.tensor(np.asarray([i[ns.FIELD_LABEL] for i in data]))

        return tokenized_smiles_padded, attention_mask_padded, labels

    # Used for adaptive sampling
    def get_feature_fn(self):
        def f(example):
            return len(example["smiles"])

        return f

    @staticmethod
    def smiles_to_text_format(smiles):
        op_smiles = OpGenerateTokenizedSmiles()

        sample_ndict = NDict({ns.FIELD_SMILES: smiles})
        sample_ndict = op_smiles(
            sample_ndict, key_in=ns.FIELD_SMILES, key_out=ns.FIELD_TOKENIZED_SMILES
        )

        sample_ndict["attention_mask"] = torch.tensor(
            [1] * len(sample_ndict[ns.FIELD_TOKENIZED_SMILES])
        )
        sample_ndict[ns.FIELD_TOKENIZED_SMILES] = torch.nn.utils.rnn.pad_sequence(
            [sample_ndict[ns.FIELD_TOKENIZED_SMILES]], batch_first=True, padding_value=2
        )
        sample_ndict["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
            [sample_ndict["attention_mask"]], batch_first=True, padding_value=0
        )

        return sample_ndict
