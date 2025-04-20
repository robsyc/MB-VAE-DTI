import importlib
import logging
import random
import shutil
import tempfile
import unittest

from bmfm_sm.core.data_modules.base_data_module import BaseDataModule
from bmfm_sm.core.data_modules.base_datasets import FuseFeaturizedDataset
from bmfm_sm.core.data_modules.namespace import FIELD_IMAGE, FIELD_LABEL, FIELD_SMILES
from bmfm_sm.predictive.data_modules.image_finetune_dataset import (
    ImageFinetuneDataPipeline,
)


class TestImageFinetuneDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dirs = importlib.resources.files("bmfm_sm.predictive.tests.test-data")
        cls.data_dir = dirs.joinpath("finetune")
        cls.task_type = "classification"
        cls.batch_size = 4
        cls.num_tasks = 1

        cls.tmp_dir = tempfile.mkdtemp()
        cls.cache_path = cls.tmp_dir
        cls.dataset_args = {"task_type": cls.task_type}
        cls.pl = ImageFinetuneDataPipeline(
            data_dir=cls.data_dir, dataset_args=cls.dataset_args
        )

    @classmethod
    def tearDownClass(cls):
        logging.info("Teardown called. Deleting cache directory.")
        shutil.rmtree(cls.tmp_dir)

    def test_constructor(self):
        assert isinstance(self.pl, FuseFeaturizedDataset)

    def test_get_sample(self):
        assert self.pl is not None

        idx = random.randint(0, len(self.pl) - 1)
        sample = self.pl[idx]

        assert sample is not None

        assert sample[FIELD_IMAGE].shape == (3, 224, 224)
        assert sample[FIELD_LABEL].shape == (self.num_tasks,)
        assert type(sample[FIELD_SMILES]) == str

    def test_data_interface(self):
        """Test test_data_interface."""
        dm = BaseDataModule(
            ImageFinetuneDataPipeline,
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            pin_memory=False,
            num_workers=0,
            dataset_args=self.dataset_args,
        )
        dm.prepare_data()
        dm.setup("fit")

        train_dataloader = dm.train_dataloader()
        batch = next(iter(train_dataloader))
        assert batch[0].shape == (self.batch_size, 3, 224, 224)
        assert batch[1].shape == (self.batch_size, self.num_tasks)


if __name__ == "__main__":
    unittest.main()
