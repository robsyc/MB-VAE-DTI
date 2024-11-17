import importlib
import shutil
import tempfile
import unittest

import numpy
import numpy.testing
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

import bmfm_sm.core.data_modules.namespace as ns
from bmfm_sm.core.data_modules.base_data_module import BaseDataModule
from bmfm_sm.core.data_modules.base_datasets import FuseFeaturizedDataset
from bmfm_sm.predictive.data_modules.multimodal_finetune_dataset import (
    MultiModalFinetuneDataPipeline,
)


class TestMultiModalFinetuneDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dirs = importlib.resources.files("bmfm_sm.predictive.tests.test-data")
        cls.data_dir = dirs.joinpath("finetune")
        cls.batch_size = 3
        cls.task_type = "classification"
        cls.tmp_dir = tempfile.mkdtemp()
        cls.dataset_args = {
            "modalities": ["GRAPH_2D_MODEL", "IMAGE_MODEL", "TEXT_MODEL"],
            "task_type": "classification",
            "adaptive_sampling": False,
        }
        cls.pl = MultiModalFinetuneDataPipeline(
            data_dir=cls.data_dir, dataset_args=cls.dataset_args
        )
        cls.expected_keys = {
            "img",
            "label",
            "smiles.tokenized",
            "attention_mask",
            "node_num",
            "node_data",
            "edge_num",
            "edge_data",
            "edge_index",
            "lap_eigvec",
        }

        cls.num_node_features = 9
        cls.num_edge_features = 5

    @classmethod
    def tearDownClass(cls):
        print("Teardown called. Deleting cache directory.")
        shutil.rmtree(cls.tmp_dir)

    def test_constructor(self):
        """Test test_constructor."""
        assert isinstance(self.pl, FuseFeaturizedDataset)

    def test_getsample(self):
        assert self.pl is not None
        assert isinstance(self.pl, Dataset)

        sample_dict = self.pl[0]
        assert sample_dict is not None
        # Graph tests
        assert isinstance(sample_dict[ns.FIELD_LABEL], numpy.ndarray)
        assert isinstance(sample_dict[ns.FIELD_LABEL][0], numpy.int64)
        assert sample_dict[ns.FIELD_LABEL][0] == 0
        assert isinstance(sample_dict[ns.FIELD_GRAPH2D], Data)
        # Image tests
        assert isinstance(sample_dict[ns.FIELD_IMAGE], torch.Tensor)
        assert sample_dict[ns.FIELD_IMAGE].shape == (3, 224, 224)
        # Text tests
        assert (
            sample_dict[ns.FIELD_SMILES]
            == "CCC1=[O+][Cu-3]2([O+]=C(CC)C1)[O+]=C(CC)CC(CC)=[O+]2"
        )

        sample_dict2 = self.pl[3]
        assert sample_dict2 is not None
        # Graph tests
        assert isinstance(sample_dict2[ns.FIELD_LABEL], numpy.ndarray)
        assert isinstance(sample_dict2[ns.FIELD_LABEL][0], numpy.int64)
        assert sample_dict2[ns.FIELD_LABEL][0] == 0
        assert isinstance(sample_dict2[ns.FIELD_GRAPH2D], Data)
        # Image tests
        assert isinstance(sample_dict[ns.FIELD_IMAGE], torch.Tensor)
        assert sample_dict[ns.FIELD_IMAGE].shape == (3, 224, 224)
        # Text tests
        assert (
            sample_dict2[ns.FIELD_SMILES]
            == "Nc1ccc(C=Cc2ccc(N)cc2S(=O)(=O)O)c(S(=O)(=O)O)c1"
        )

    def test_collate(self):
        data_list1 = [self.pl[0]]
        assert len(data_list1) == 1
        c1 = self.pl.collate_fn(data_list1)
        assert c1["smiles.tokenized"].shape == (1, 37)
        assert c1["attention_mask"].shape == (1, 37)
        assert c1["img"].shape == (1, 3, 224, 224)
        assert c1 is not None

        # Check we have all expected fields
        self.assertSetEqual(set(c1.keys()), self.expected_keys)
        assert len(c1["node_num"]) == 1

        data_list2 = [self.pl[0], self.pl[1]]
        assert len(data_list2) == 2
        c2 = self.pl.collate_fn(data_list2)
        assert c2 is not None
        assert c2["smiles.tokenized"].shape == (2, 69)
        assert c2["attention_mask"].shape == (2, 69)
        assert c2["img"].shape == (2, 3, 224, 224)
        assert len(c2["node_num"]) == 2

        data_list3 = (self.pl[0], self.pl[1], self.pl[2])
        assert len(data_list3) == 3
        c3 = self.pl.collate_fn(data_list3)
        assert c3 is not None
        assert c3["smiles.tokenized"].shape == (3, 69)
        assert c3["attention_mask"].shape == (3, 69)
        assert c3["img"].shape == (3, 3, 224, 224)
        assert len(c3["node_num"]) == 3

    def test_data_interface(self):
        """Test test_data_interface."""
        dm = BaseDataModule(
            MultiModalFinetuneDataPipeline,
            self.data_dir,
            batch_size=self.batch_size,
            pin_memory=False,
            num_workers=1,
            dataset_args=self.dataset_args,
        )

        dm.prepare_data()

        dm.setup("fit")

        train_dataloader = dm.train_dataloader()
        b1 = next(iter(train_dataloader))
        assert b1 is not None
        assert isinstance(b1, dict)
        self.assertSetEqual(set(b1.keys()), self.expected_keys)
        assert len(b1["node_num"]) == self.batch_size
        # Check the number of node features
        assert b1["node_data"].shape[1] == self.num_node_features
        # Check the number of edge features
        assert b1["edge_data"].shape[1] == self.num_edge_features

        valid_dataloader = dm.val_dataloader()
        b2 = next(iter(valid_dataloader))
        assert b2 is not None
        assert isinstance(b2, dict)
        self.assertSetEqual(set(b2.keys()), self.expected_keys)
        assert len(b2["node_num"]) == self.batch_size
        # Check the number of node features
        assert b2["node_data"].shape[1] == self.num_node_features
        # Check the number of edge features
        assert b2["edge_data"].shape[1] == self.num_edge_features

        dm.setup("test")
        test_dataloader = dm.test_dataloader()
        b3 = next(iter(test_dataloader))
        assert b3 is not None
        assert isinstance(b3, dict)
        self.assertSetEqual(set(b3.keys()), self.expected_keys)
        assert len(b3["node_num"]) == self.batch_size
        # Check the number of node features
        assert b3["node_data"].shape[1] == self.num_node_features
        # Check the number of edge features
        assert b3["edge_data"].shape[1] == self.num_edge_features

        for k in self.expected_keys:
            assert isinstance(b1[k], torch.Tensor)
            assert isinstance(b2[k], torch.Tensor)
            assert isinstance(b3[k], torch.Tensor)

    def test_data_interface_with_betti_features(self):
        """Test test_data_interface."""
        # Update the dataset arguments to includde Betti numbers as atom/node features
        # We now have 11 nodes features instead of 9
        self.dataset_args = {
            "modalities": ["GRAPH_2D_MODEL", "IMAGE_MODEL", "TEXT_MODEL"],
            "task_type": "classification",
            "adaptive_sampling": False,
            "include_betti_01": True,
        }
        self.num_node_features = 11
        self.num_edge_features = 5

        dm = BaseDataModule(
            MultiModalFinetuneDataPipeline,
            self.data_dir,
            batch_size=self.batch_size,
            pin_memory=False,
            num_workers=1,
            dataset_args=self.dataset_args,
        )

        dm.prepare_data()

        dm.setup("fit")

        train_dataloader = dm.train_dataloader()
        b1 = next(iter(train_dataloader))
        assert b1 is not None
        assert isinstance(b1, dict)
        self.assertSetEqual(set(b1.keys()), self.expected_keys)
        assert len(b1["node_num"]) == self.batch_size
        # Check the number of node features
        assert b1["node_data"].shape[1] == self.num_node_features
        # Check the number of edge features
        assert b1["edge_data"].shape[1] == self.num_edge_features

        valid_dataloader = dm.val_dataloader()
        b2 = next(iter(valid_dataloader))
        assert b2 is not None
        assert isinstance(b2, dict)
        self.assertSetEqual(set(b2.keys()), self.expected_keys)
        assert len(b2["node_num"]) == self.batch_size
        # Check the number of node features
        assert b2["node_data"].shape[1] == self.num_node_features
        # Check the number of edge features
        assert b2["edge_data"].shape[1] == self.num_edge_features

        dm.setup("test")
        test_dataloader = dm.test_dataloader()
        b3 = next(iter(test_dataloader))
        assert b3 is not None
        assert isinstance(b3, dict)
        self.assertSetEqual(set(b3.keys()), self.expected_keys)
        assert len(b3["node_num"]) == self.batch_size
        # Check the number of node features
        assert b3["node_data"].shape[1] == self.num_node_features
        # Check the number of edge features
        assert b3["edge_data"].shape[1] == self.num_edge_features

        for k in self.expected_keys:
            assert isinstance(b1[k], torch.Tensor)
            assert isinstance(b2[k], torch.Tensor)
            assert isinstance(b3[k], torch.Tensor)


if __name__ == "__main__":
    unittest.main()
