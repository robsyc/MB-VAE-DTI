"""
- Superclass for the MPP Finetune Dataset
    - The text, image, graph2d, graph2dgnn, graph3d, and multimodality finetune MPP datasets will be subclasses
- Implements shared functionality such as:
    - Filtering the source CSV based on the split file and the current stage
    - Implements the next_item() required by the FuseFeaturizedDataset (passes back the next row of the filtered dataframe)
    - Contains the shared Ops used by all subclasses (format MPP label)
- Subclasses still need to:
    - Implement the __init__() if any extra variables or dataset args need to be stored
    - Implement the get_ops_pipeline() function if any other modality-specific featurization ops are needed
    - Implement collate_fn() if any collation needs to be done.
"""

import json
import os

import pandas as pd
from fuse.data.pipelines.pipeline_default import PipelineDefault

import bmfm_sm.core.data_modules.namespace as ns
from bmfm_sm.core.data_modules.base_datasets import FuseFeaturizedDataset
from bmfm_sm.core.data_modules.fuse_ops.ops_general import OpFormatMPPLabel


class MPPFinetuneDataset(FuseFeaturizedDataset):
    def __init__(self, data_dir, dataset_args, stage="train"):
        self.task_type = dataset_args.get("task_type")
        self.stage = stage
        self.dataset_args = dataset_args

        """
        - If split_file exists in the dataset args, assume there is 1 data CSVs that needs to be split based on the train/val/test idx in the split_file
        - If split_file is not in the dataset args, assume there is 3 separate data CSVs that we can read from (and we read the CSV for the current stage)
        """

        if "split_file" in self.dataset_args:
            with open(dataset_args.get("split_file")) as json_file:
                loaded_splits = json.load(json_file)
            source_file = os.path.expandvars(loaded_splits["source_file"])
            full_data = pd.read_csv(source_file)  # Loading in the source data file
            split_idx = loaded_splits[
                stage
            ]  # Loading the indices needed for the current stage (train/val/test)
            self.data = full_data[
                full_data["index"].isin(split_idx)
            ]  # Filter and only keep the rows needed for the current stage

            # Re-ordering the dataframe based on the order of the split indices
            order_mapping = {value: index for index, value in enumerate(split_idx)}
            self.data["order"] = self.data["index"].map(order_mapping)
            self.data = (
                self.data.sort_values("order")
                .drop(columns="order")
                .reset_index(drop=True)
            )
        else:
            self.data = pd.read_csv(os.path.join(data_dir, f"data_{stage}.csv"))

        super().__init__(data_dir=data_dir, dataset_args=dataset_args, stage=stage)

    # This is called by the super class' __getitem__. We pass back a single raw data sample that is then featurized by the Ops Pipeline
    def next_item(self, idx):
        return self.data.iloc[idx]

    def get_ops_pipeline(self) -> PipelineDefault:
        # Common operation used by all MPP Dataset classes
        ops = [
            (
                OpFormatMPPLabel(
                    include_label_index=self.dataset_args.get(
                        "include_label_index", None
                    )
                ),
                {
                    "key_in": ns.FIELD_LABEL,
                    "key_out": ns.FIELD_LABEL,
                    "task_type": self.task_type,
                },
            ),
        ]
        return PipelineDefault("all_ops", ops)

    def collate_fn(self, batch):
        return batch
