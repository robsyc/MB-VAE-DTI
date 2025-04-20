import torch
from fuse.data.ops.ops_common import OpLambda
from fuse.data.pipelines.pipeline_default import PipelineDefault

import bmfm_sm.core.data_modules.namespace as ns
from bmfm_sm.core.data_modules.fuse_ops.ops_ligand_target import OpLookupTargetEmbedding
from bmfm_sm.predictive.data_modules.mpp_finetune_dataset import MPPFinetuneDataset
from bmfm_sm.predictive.data_modules.multimodal_finetune_dataset import (
    MultiModalFinetuneDataPipeline,
)


class LigandTargetFinetuneDataPipeline(MPPFinetuneDataset):
    def __init__(self, data_dir, dataset_args, stage="train"):
        self.data_pipeline = MultiModalFinetuneDataPipeline(
            data_dir=data_dir, dataset_args=dataset_args, stage=stage
        )
        super().__init__(data_dir=data_dir, dataset_args=dataset_args, stage=stage)

    def get_ops_pipeline(self) -> PipelineDefault:
        multi_modal_ops = self.data_pipeline.get_ops_pipeline()._ops_and_kwargs
        self.protein_op = OpLookupTargetEmbedding
        self.target_embedding_file = self.dataset_args["target_embedding_file"]

        model_ops = [
            (
                self.protein_op(self.target_embedding_file),
                {"key_in": ns.FIELD_PROTEIN, "key_out": ns.FIELD_DATA_PROTEIN_EMB},
            ),
            (
                OpLambda(torch.from_numpy),
                {"key": ns.FIELD_DATA_PROTEIN_EMB, "op_id": None},
            ),
        ]

        return PipelineDefault("all_ops", multi_modal_ops + model_ops)

    def collate_fn(self, batch):
        collated = self.data_pipeline.collate_fn(batch)
        collated[ns.FIELD_DATA_PROTEIN_EMB] = torch.stack(
            [i[ns.FIELD_DATA_PROTEIN_EMB] for i in batch]
        )
        return collated
