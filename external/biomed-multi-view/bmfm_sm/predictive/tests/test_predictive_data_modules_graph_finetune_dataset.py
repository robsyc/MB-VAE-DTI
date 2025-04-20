import importlib
import logging
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch

import bmfm_sm.core.data_modules.namespace as ns
from bmfm_sm.core.data_modules.base_data_module import BaseDataModule
from bmfm_sm.core.data_modules.base_datasets import FuseFeaturizedDataset
from bmfm_sm.core.data_modules.samplers import TokenBudgetSampler
from bmfm_sm.predictive.data_modules.graph_finetune_dataset import (
    Graph2dFinetuneDataPipeline,
    Graph2dGNNFinetuneDataPipeline,
)


def identity(x):
    return x


class TestGraphFinetuneDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dirs = importlib.resources.files("bmfm_sm.predictive.tests.test-data")
        cls.data_dir = dirs.joinpath("finetune")
        cls.batch_size = 2
        cls.task_type = "classification"
        cls.dataset_args = {"task_type": "classification"}
        cls.pl = Graph2dGNNFinetuneDataPipeline(
            data_dir=cls.data_dir, dataset_args=cls.dataset_args
        )
        cls.num_node_features = 9
        cls.num_edge_features = 5

    def test_constructor(self):
        """Test test_constructor."""
        assert isinstance(self.pl, FuseFeaturizedDataset)

    def test_get_sample(self):
        sample = self.pl[0]
        assert sample is not None
        assert (
            sample[ns.FIELD_SMILES]
            == "CCC1=[O+][Cu-3]2([O+]=C(CC)C1)[O+]=C(CC)CC(CC)=[O+]2"
        )
        assert isinstance(sample[ns.FIELD_LABEL], np.ndarray)
        assert isinstance(sample[ns.FIELD_LABEL][0], np.int64)
        assert sample[ns.FIELD_LABEL][0] == 0
        assert isinstance(sample[ns.FIELD_GRAPH2D_GNN], Data)

        sample = self.pl[3]
        assert sample is not None
        assert (
            sample[ns.FIELD_SMILES] == "Nc1ccc(C=Cc2ccc(N)cc2S(=O)(=O)O)c(S(=O)(=O)O)c1"
        )
        assert isinstance(sample[ns.FIELD_LABEL], np.ndarray)
        assert isinstance(sample[ns.FIELD_LABEL][0], np.int64)
        assert sample[ns.FIELD_LABEL][0] == 0
        assert isinstance(sample[ns.FIELD_GRAPH2D_GNN], Data)

    def test_collate(self):
        # Checking collation with 1 sample
        data_list1 = [self.pl[0]]
        c1 = self.pl.collate_fn(data_list1)
        assert c1 is not None
        assert len(c1.y) == 1

        # Check collation with multiple samples
        data_list3 = [self.pl[0], self.pl[1], self.pl[2]]
        c3 = self.pl.collate_fn(data_list3)
        assert c3 is not None
        assert len(c3.y) == 3

        # Check that a sample with missing Graph data should be skipped
        fake_data = {
            ns.FIELD_SMILES: "SMILE",
            ns.FIELD_LABEL: np.array([0]),
            ns.FIELD_GRAPH2D_GNN: None,
        }
        c4 = self.pl.collate_fn([self.pl[0], self.pl[1], self.pl[2], fake_data])
        assert c4 is not None
        assert len(c4.y) == 3

    def test_data_interface(self):
        """Test test_data_interface."""
        dm = BaseDataModule(
            Graph2dGNNFinetuneDataPipeline,
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
        assert isinstance(b1, Batch)
        assert len(b1) == self.batch_size
        assert isinstance(b1[0], Data)
        # Check the number of node features
        assert b1[0].x.shape[1] == self.num_node_features
        # Check the number of edge features
        assert b1[0].edge_attr.shape[1] == self.num_edge_features

        valid_dataloader = dm.val_dataloader()
        b2 = next(iter(valid_dataloader))
        assert b2 is not None
        assert isinstance(b2, Batch)
        assert len(b2) == self.batch_size
        assert isinstance(b2[0], Data)
        # Check the number of node features
        assert b2[0].x.shape[1] == self.num_node_features
        # Check the number of edge features
        assert b2[0].edge_attr.shape[1] == self.num_edge_features

        dm.setup("test")
        test_dataloader = dm.test_dataloader()
        b3 = next(iter(test_dataloader))
        assert b3 is not None
        assert isinstance(b3, Batch)
        assert len(b3) == self.batch_size
        assert isinstance(b3[0], Data)
        # Check the number of node features
        assert b3[0].x.shape[1] == self.num_node_features
        # Check the number of edge features
        assert b3[0].edge_attr.shape[1] == self.num_edge_features

    def test_data_interface_with_betti_features(self):
        """Test test_data_interface."""
        # Update the dataset arguments to includde Betti numbers as atom/node features
        # We now have 11 nodes features instead of 9
        self.dataset_args = {
            "task_type": "classification",
            "include_betti_01": True,
        }
        self.num_node_features = 11
        self.num_edge_features = 5

        dm = BaseDataModule(
            Graph2dGNNFinetuneDataPipeline,
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
        assert isinstance(b1, Batch)
        assert len(b1) == self.batch_size
        assert isinstance(b1[0], Data)
        # Check the number of node features
        assert b1[0].x.shape[1] == self.num_node_features
        # Check the number of edge features
        assert b1[0].edge_attr.shape[1] == self.num_edge_features

        valid_dataloader = dm.val_dataloader()
        b2 = next(iter(valid_dataloader))
        assert b2 is not None
        assert isinstance(b2, Batch)
        assert len(b2) == self.batch_size
        assert isinstance(b2[0], Data)
        # Check the number of node features
        assert b2[0].x.shape[1] == self.num_node_features
        # Check the number of edge features
        assert b2[0].edge_attr.shape[1] == self.num_edge_features

        dm.setup("test")
        test_dataloader = dm.test_dataloader()
        b3 = next(iter(test_dataloader))
        assert b3 is not None
        assert isinstance(b3, Batch)
        assert len(b3) == self.batch_size
        assert isinstance(b3[0], Data)
        # Check the number of node features
        assert b3[0].x.shape[1] == self.num_node_features
        # Check the number of edge features
        assert b3[0].edge_attr.shape[1] == self.num_edge_features


#######################################################################################################################
class TestGraph2dFinetuneDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = importlib.resources.files(
            "bmfm_sm.predictive.tests.test-data"
        ).joinpath("finetune")
        cls.batch_size = 3
        cls.num_tasks = 1
        cls.task_type = "classification"
        cls.dataset_args = {"task_type": "classification"}
        cls.pl = Graph2dFinetuneDataPipeline(
            data_dir=cls.data_dir, dataset_args=cls.dataset_args
        )
        cls.expected_keys = {
            "y",
            "node_num",
            "node_data",
            "edge_index",
            "edge_num",
            "edge_data",
            "lap_eigvec",
        }
        cls.num_node_features = 9
        cls.num_edge_features = 5

    def test_constructor(self):
        """Test test_constructor."""
        assert isinstance(self.pl, FuseFeaturizedDataset)
        assert self.pl is not None

    def test_get_sample(self):
        sample = self.pl[0]
        assert sample is not None

        assert (
            sample[ns.FIELD_SMILES]
            == "CCC1=[O+][Cu-3]2([O+]=C(CC)C1)[O+]=C(CC)CC(CC)=[O+]2"
        )

        assert isinstance(sample[ns.FIELD_LABEL], np.ndarray)
        assert sample[ns.FIELD_LABEL].shape == (self.num_tasks,)
        assert isinstance(sample[ns.FIELD_LABEL][0], np.int64)
        assert sample[ns.FIELD_LABEL][0] == 0

        assert isinstance(sample[ns.FIELD_GRAPH2D], Data)

    def test_collate(self):
        # Checking collation with 1 sample
        data_list1 = [self.pl[0]]
        c1 = self.pl.collate_fn(data_list1)
        assert c1 is not None
        self.assertSetEqual(set(c1.keys()), self.expected_keys)
        assert len(c1["node_num"]) == 1

        # Check collation with multiple samples
        data_list3 = [self.pl[0], self.pl[1], self.pl[2]]
        c3 = self.pl.collate_fn(data_list3)
        assert c3 is not None
        self.assertSetEqual(set(c3.keys()), self.expected_keys)
        assert len(c3["node_num"]) == 3

        # Check that a sample with missing Graph data should be skipped
        fake_data = {
            ns.FIELD_SMILES: "SMILE",
            ns.FIELD_LABEL: np.array([0]),
            ns.FIELD_GRAPH2D: None,
        }
        c4 = self.pl.collate_fn([self.pl[0], self.pl[1], self.pl[2], fake_data])
        assert c4 is not None
        self.assertSetEqual(set(c4.keys()), self.expected_keys)
        assert len(c4["node_num"]) == 3

    def test_adaptive_sampling_feature_fn(self):
        # len(list) == 1
        data_list1 = [self.pl[0]]
        assert len(data_list1) == 1
        f = self.pl.get_feature_fn()
        c1 = f(data_list1[0])
        assert isinstance(c1, int)  # num_atoms should be integer

    def test_data_interface(self):
        """Test test_data_interface."""
        dm = BaseDataModule(
            Graph2dFinetuneDataPipeline,
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
            "task_type": "classification",
            "include_betti_01": True,
        }
        self.num_node_features = 11
        self.num_edge_features = 5

        dm = BaseDataModule(
            Graph2dFinetuneDataPipeline,
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

    def test_adaptive_sampling_no_collation(self):
        """Test test_adaptive_sampling_no_collation."""
        sampler = TokenBudgetSampler(self.pl, 3, 150)
        dataloader = DataLoader(
            dataset=self.pl,
            batch_sampler=sampler,
            num_workers=1,
            collate_fn=identity,
            persistent_workers=True,
            prefetch_factor=8,
            pin_memory=True,
        )
        batches = TokenBudgetSampler.get_batches(dataloader)
        batch_weights = TokenBudgetSampler.get_batch_weights(
            batches, self.pl.get_feature_fn()
        )
        logging.info(f"batch weights {batch_weights}")

    # TODO: Need to support adaptive sampling in new BaseDataModule; Below is currently doing nothing
    def test_adaptive_sampling_with_collation(self):
        """Test test_adaptive_sampling_with_collation."""
        # Update the dataset arguments to enable adaptive sampling
        self.dataset_args = {
            "task_type": "classification",
            "adaptive_sampling": True,
            "adaptive_sampling_num_bins": 3,
            "adaptive_sampling_budget": 150,
        }
        dm = BaseDataModule(
            Graph2dFinetuneDataPipeline,
            self.data_dir,
            batch_size=self.batch_size,
            pin_memory=False,
            num_workers=1,
            dataset_args=self.dataset_args,
        )

        dm.prepare_data()
        dm.setup("fit")
        train_dataloader = dm.train_dataloader()
        next(iter(train_dataloader))


if __name__ == "__main__":
    unittest.main()
