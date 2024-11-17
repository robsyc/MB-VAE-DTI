import importlib
import logging

import pytorch_lightning as pl
from litdata import StreamingDataLoader, StreamingDataset
from torch.utils.data import DataLoader

from bmfm_sm.core.data_modules.samplers import TokenBudgetSampler


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_class,
        data_dir,
        dataset_args,
        batch_size=32,
        pin_memory=False,
        num_workers=1,
    ):
        """
        params:
            - dataset_class --> StreamingDataset type class, can pass in the string path or the class object
            - data_dir --> Directory where the shards are stored (should contrain a 'train', 'val' and optionally a 'test' subdir with the shards)
            - dataset_args --> dictionary with any arguments needed by the dataset_class, Will be passed when initializing the dataset
            - batch_size / pin_memory / num_workers --> To be used by the Streaming Dataloaders.
        """
        super().__init__()

        self.dataset_class = BaseDataModule.import_class(dataset_class)

        # Core arguments
        self.data_dir = data_dir
        # Args to be used by dataloaders
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.dataloader_type = (
            StreamingDataLoader
            if issubclass(self.dataset_class, StreamingDataset)
            else DataLoader
        )

        # Dataset Arguments
        self.dataset_args = dataset_args

    def prepare_data(self):
        logging.info("prepare_data() got called.")
        return

    def setup(self, stage="fit"):
        logging.info("setup() got called.")
        if stage == "fit":
            self.train_dataset = self.dataset_class(
                self.data_dir, self.dataset_args, stage="train"
            )

            self.val_dataset = self.dataset_class(
                self.data_dir, self.dataset_args, stage="val"
            )
        elif stage == "test":
            self.test_dataset = self.dataset_class(
                self.data_dir, self.dataset_args, stage="test"
            )
        else:
            raise ValueError(
                f"Stage argument passed to setup must be fit or test. {stage} is not supported"
            )

    def train_dataloader(self):
        return self.build_dataloader(self.train_dataset, is_train=True)

    def val_dataloader(self):
        return self.build_dataloader(self.val_dataset, is_train=False)

    def test_dataloader(self):
        return self.build_dataloader(self.test_dataset, is_train=False)

    def build_dataloader(self, dataset, is_train):
        adaptive_sampling = self.dataset_args.get("adaptive_sampling", False)
        if adaptive_sampling:
            num_bins = self.dataset_args.get("adaptive_sampling_num_bins", None)
            budget = self.dataset_args.get("adaptive_sampling_budget", None)
            assert num_bins, "When adaptive sampling is enabled, num_bins cant be null"
            assert budget, "When adaptive sampling is enabled, budget cant be null"
            sampler = TokenBudgetSampler(dataset, num_bins, budget, is_train=is_train)

            return DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=self.num_workers,
                collate_fn=dataset.collate_fn,
                pin_memory=self.pin_memory,
            )
        else:
            return self.dataloader_type(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=dataset.collate_fn,
                pin_memory=self.pin_memory,
                shuffle=is_train,
            )

    @staticmethod
    def import_class(class_path):
        if type(class_path) is str:
            split_path = class_path.rsplit(".", 1)
            module_name, class_name = split_path[0], split_path[1]
            return getattr(importlib.import_module(module_name), class_name)
        else:
            return class_path
