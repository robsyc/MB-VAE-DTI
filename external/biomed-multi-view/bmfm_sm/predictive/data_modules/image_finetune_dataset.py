import numpy as np
import torch
import torchvision.transforms as transforms
from fuse.data.ops.ops_common import OpLambda
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.utils.ndict import NDict

import bmfm_sm.core.data_modules.namespace as ns
from bmfm_sm.core.data_modules.fuse_ops.ops_images import (
    OpNormalizeImage,
    OpPreprocessImage,
    OpSmilesToImage,
)
from bmfm_sm.predictive.data_modules.mpp_finetune_dataset import MPPFinetuneDataset


class ImageFinetuneDataPipeline(MPPFinetuneDataset):
    def __init__(self, data_dir, dataset_args, stage="train"):
        super().__init__(data_dir=data_dir, dataset_args=dataset_args, stage=stage)

    def get_ops_pipeline(self) -> PipelineDefault:
        superclass_ops = super().get_ops_pipeline()._ops_and_kwargs

        # Common operations for train, val and test
        image_ops = [
            (OpSmilesToImage(), {"key_in": ns.FIELD_SMILES, "key_out": ns.FIELD_IMAGE}),
        ]

        if self.stage == "train":
            image_ops.extend(
                [
                    #  Use the default image transformation pipeline (includes augmentation)
                    (
                        OpPreprocessImage(),
                        {"key_in": ns.FIELD_IMAGE, "key_out": ns.FIELD_IMAGE},
                    )
                ]
            )
        elif self.stage in ["val", "test"]:
            image_ops.extend(
                [
                    # Only crops the image (no augmentation)
                    (
                        OpPreprocessImage(),
                        {
                            "key_in": ns.FIELD_IMAGE,
                            "key_out": ns.FIELD_IMAGE,
                            "img_transf_func": transforms.CenterCrop(224),
                        },
                    )
                ]
            )

        image_ops.extend(
            [
                #  Convert the images into tensors
                (
                    OpLambda(transforms.ToTensor()),
                    {"key": ns.FIELD_IMAGE, "op_id": None},
                ),
                # Normalize the tensors
                (
                    OpNormalizeImage(),
                    {"key_in": ns.FIELD_IMAGE, "key_out": ns.FIELD_IMAGE},
                ),
            ]
        )

        return PipelineDefault("all_ops", superclass_ops + image_ops)

    def collate_fn(self, batch):
        images = torch.stack([item[ns.FIELD_IMAGE] for item in batch])
        labels = torch.tensor(np.asarray([item[ns.FIELD_LABEL] for item in batch]))

        return images, labels

    @staticmethod
    def smiles_to_image_format(smiles):
        op_smiles_to_image = OpSmilesToImage()
        op_preprocess_image = OpPreprocessImage()
        op_lambda = OpLambda(transforms.ToTensor())
        op_normalize = OpNormalizeImage()

        # Passing the NDict through the operations
        sample_ndict = NDict({ns.FIELD_SMILES: smiles})
        sample_ndict = op_smiles_to_image(
            sample_ndict, key_in=ns.FIELD_SMILES, key_out=ns.FIELD_IMAGE
        )
        sample_ndict = op_preprocess_image(
            sample_ndict,
            key_in=ns.FIELD_IMAGE,
            key_out=ns.FIELD_IMAGE,
            img_transf_func=transforms.CenterCrop(224),
        )

        sample_ndict = op_lambda(sample_ndict, key=ns.FIELD_IMAGE, op_id=None)
        sample_ndict = op_normalize(
            sample_ndict, key_in=ns.FIELD_IMAGE, key_out=ns.FIELD_IMAGE
        )

        output_dict = {ns.FIELD_IMAGE: sample_ndict[ns.FIELD_IMAGE].unsqueeze(0)}
        return output_dict
