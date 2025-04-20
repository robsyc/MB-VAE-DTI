import importlib.resources
import logging
import shutil
import tempfile
import unittest

import pytorch_lightning
import torch.nn
from pytorch_lightning.callbacks import TQDMProgressBar

from bmfm_sm.core.data_modules.base_data_module import BaseDataModule
from bmfm_sm.core.data_modules.namespace import Modality, TaskType
from bmfm_sm.core.modules.base_pretrained_model import MultiTaskPredictionHead
from bmfm_sm.predictive.data_modules.graph_finetune_dataset import (
    Graph2dFinetuneDataPipeline,
    Graph2dGNNFinetuneDataPipeline,
)
from bmfm_sm.predictive.data_modules.image_finetune_dataset import (
    ImageFinetuneDataPipeline,
)
from bmfm_sm.predictive.data_modules.text_finetune_dataset import (
    TextFinetuneDataPipeline,
)
from bmfm_sm.predictive.modules.finetune_lightning_module import FineTuneLightningModule
from bmfm_sm.predictive.modules.graph_2d_models import (
    AttentiveFPModel,
    GCNModel,
    GINModel,
    Graph2dModel,
    TrimNetModel,
)
from bmfm_sm.predictive.modules.image_models import ImageModel

# TODO: cleanup this special handling to prevent travis failure in importing fast_transformers
try:
    from bmfm_sm.predictive.modules.text_models import TextModel

    module_available = True
except ImportError:
    module_available = False


def get_fqcn(cls_ref):
    return cls_ref.__module__ + "." + cls_ref.__name__


task_type = "classification"
num_tasks = 1

model_params = {
    get_fqcn(ImageModel): {"baseModel": "ResNet18"},
    get_fqcn(GCNModel): {"num_layer": 2, "emb_dim": 128, "num_tasks": num_tasks},
    get_fqcn(GINModel): {"num_layer": 2, "emb_dim": 128, "num_tasks": num_tasks},
    get_fqcn(TrimNetModel): {
        "in_channels": 9,
        "hidden_channels": 32,
        "num_layers": 2,
        "edge_dim": 5,
        "heads": 2,
        "out_channels": num_tasks,
    },
    get_fqcn(AttentiveFPModel): {
        "in_channels": 9,
        "hidden_channels": 32,
        "num_layers": 2,
        "edge_dim": 5,
        "num_timesteps": 5,
        "out_channels": 32,
    },
    get_fqcn(Graph2dModel): {
        "num_classes": num_tasks,
        "num_tasks": num_tasks,
    },
}

if module_available:
    model_params[get_fqcn(TextModel)] = {
        "n_vocab": 2362,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "num_feats": 32,
        "d_dropout": 0.2,
        "seed": 12345,
    }


class TestModelConstructor(unittest.TestCase):
    def test_GCNModel_constructor(self):
        m = GCNModel(num_layer=2, emb_dim=8, num_tasks=2)
        assert isinstance(m, torch.nn.Module)

    def test_GINModel_constructor(self):
        m = GINModel(num_layer=2, emb_dim=8, num_tasks=2)
        assert isinstance(m, torch.nn.Module)

    def test_AttentiveFPModel_constructor(self):
        m = AttentiveFPModel(
            in_channels=5,
            hidden_channels=8,
            out_channels=2,
            edge_dim=3,
            num_layers=2,
            num_timesteps=5,
        )
        assert isinstance(m, torch.nn.Module)

    def test_TrimNet_constructor(self):
        m = TrimNetModel(
            in_channels=5,
            hidden_channels=8,
            out_channels=2,
            edge_dim=3,
            num_layers=2,
            heads=3,
        )
        assert isinstance(m, torch.nn.Module)

    def test_Graph2d_constructor(self):
        m = Graph2dModel(num_classes=2)
        assert isinstance(m, torch.nn.Module)

    def test_Image_constructor(self):
        m = ImageModel(baseModel="ResNet18")
        assert isinstance(m, torch.nn.Module)

    def test_Text_constructor(self):
        if module_available:
            m = TextModel()
            assert isinstance(m, torch.nn.Module)


class TestLightningModuleConstructor(unittest.TestCase):
    def build_module(self, cls_ref: str) -> None:
        model_class = get_fqcn(cls_ref)
        params = model_params[model_class]
        m = FineTuneLightningModule(
            base_model_class=model_class,
            model_params=params,
            task_type="classification",
            num_tasks=num_tasks,
        )
        assert isinstance(m, pytorch_lightning.LightningModule)
        print(m.model)
        return m

    def test_TextModule_constructor(self):
        if module_available:
            m = self.build_module(TextModel)
            assert isinstance(m.model, TextModel)

    def test_ImageModule_constructor(self):
        m = self.build_module(ImageModel)
        assert isinstance(m.model, ImageModel)

    def test_GNNModelLightningModule_constructor(self):
        m = self.build_module(TrimNetModel)
        assert isinstance(m.model, TrimNetModel)

        m = self.build_module(GCNModel)
        assert isinstance(m.model, GCNModel)

        m = self.build_module(GINModel)
        assert isinstance(m.model, GINModel)

        m = self.build_module(AttentiveFPModel)
        assert isinstance(m.model, AttentiveFPModel)

    def test_GTModelLightningModule_constructor(self):
        m = self.build_module(Graph2dModel)
        assert isinstance(m.model, Graph2dModel)


class TestFinetuneLightningModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dirs = importlib.resources.files("bmfm_sm.predictive.tests.test-data")
        cls.data_dir = dirs.joinpath("finetune")
        cls.sample_ids = [2, 3, 4, 5, 6, 7]
        cls.sample_labels = ""
        cls.batch_size = 2
        cls.task_type = "classification"
        cls.tmp_dir = tempfile.mkdtemp()
        cls.cache_path = cls.tmp_dir
        cls.dataset_args = {
            "val_size": 0.2,
            "test_size": 0.2,
            "task_type": "classification",
        }
        cls.fuse_args = {"cache_path": cls.cache_path, "num_workers": 1}
        cls.tmp_dir = tempfile.mkdtemp()

    def run_batch(self, dataset_class, cls_ref, model_params, dataset_args):
        dm = BaseDataModule(
            dataset_class=dataset_class,
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            pin_memory=False,
            num_workers=1,
            dataset_args=dataset_args,
        )

        dm.prepare_data()
        dm.setup("fit")
        dm.setup("test")
        train_dataloader = dm.train_dataloader()
        valid_dataloader = dm.val_dataloader()
        test_dataloader = dm.test_dataloader()

        if not model_params:
            model_params[get_fqcn(GCNModel)]
        model = FineTuneLightningModule(
            get_fqcn(cls_ref),
            model_params,
            task_type="classification",
            num_tasks=num_tasks,
        )

        # Fine tune / train the model on the training and validation sets
        trainer = pytorch_lightning.Trainer(
            accelerator="cpu",
            max_epochs=2,
            callbacks=[TQDMProgressBar(refresh_rate=1)],
            log_every_n_steps=1,
            default_root_dir=self.tmp_dir,
        )
        trainer.fit(model, train_dataloader, valid_dataloader)

        # Evaluate the model on the train, vaidlation, and test sets
        trainer.test(model, train_dataloader)
        trainer.test(model, valid_dataloader)
        trainer.test(model, test_dataloader)

        # Test get_embeddings method
        x = next(iter(test_dataloader))

        # Data loader gives x&y for ImageModel and just x for Text
        if (type(x) is list or type(x) is tuple) and len(x) == 2:
            x = x[0]

        e = model.model.get_embeddings(x)
        if model.model.modality != Modality.TEXT:
            # TODO: not sure why only the text model returns None
            assert e is not None

    @classmethod
    def tearDownClass(cls):
        print("Teardown called. Deleting cache directory.")
        shutil.rmtree(cls.tmp_dir)

    def test_ImageModel(self):
        """
        Test fails when it hits the get_embeddings call in run_batch. The 'x' that is passed into get_embeddings is a list (presumably of images)
        which has len 2 BUT the SHAPE OF FIRST ITEM IN LIST IS torch.Size([2, 3, 224, 224])
        If x is a batch, I'm not sure why the first item in the batch has 4 dimensions
        Causes get_embedding to fail since the model does not expect a list with tensors of 4 dimensions (it expects images)
        Need to check if it is an issue with the test or the model.
        """
        dataset_args = dict(self.dataset_args)
        dataset_args["dynamic_stage"] = "train"
        dataset_args["ret_index"] = False
        self.run_batch(
            ImageFinetuneDataPipeline,
            ImageModel,
            model_params[get_fqcn(ImageModel)],
            dataset_args,
        )

    def test_TextModel(self):
        if module_available:
            dataset_args = dict(self.dataset_args)
            dataset_args["dynamic_stage"] = "train"
            self.run_batch(
                TextFinetuneDataPipeline,
                TextModel,
                model_params[get_fqcn(TextModel)],
                dataset_args,
            )

    def test_GNNModel_GCN(self):
        self.run_batch(
            Graph2dGNNFinetuneDataPipeline,
            GCNModel,
            model_params[get_fqcn(GCNModel)],
            self.dataset_args,
        )

    def test_GNNModel_GCN_v2(self):
        self.run_batch(
            Graph2dGNNFinetuneDataPipeline,
            GCNModel,
            model_params[get_fqcn(GCNModel)],
            self.dataset_args,
        )

    def test_GNNModel_GIN(self):
        self.run_batch(
            Graph2dGNNFinetuneDataPipeline,
            GINModel,
            model_params[get_fqcn(GINModel)],
            self.dataset_args,
        )

    def test_GNNModel_GIN_v2(self):
        self.run_batch(
            Graph2dGNNFinetuneDataPipeline,
            GINModel,
            model_params[get_fqcn(GINModel)],
            self.dataset_args,
        )

    def test_GNNModel_AttentiveFP(self):
        self.run_batch(
            Graph2dGNNFinetuneDataPipeline,
            AttentiveFPModel,
            model_params[get_fqcn(AttentiveFPModel)],
            self.dataset_args,
        )

    def test_GNNModel_AttentiveFP_v2(self):
        self.run_batch(
            Graph2dGNNFinetuneDataPipeline,
            AttentiveFPModel,
            model_params[get_fqcn(AttentiveFPModel)],
            self.dataset_args,
        )

    def test_GNNModel_AttentiveFP_v3(self):
        mp = dict(model_params[get_fqcn(AttentiveFPModel)])
        da = dict(self.dataset_args)
        mp["in_channels"] = 11
        da["include_betti_01"] = True
        self.run_batch(Graph2dGNNFinetuneDataPipeline, AttentiveFPModel, mp, da)

    def test_GNNModel_TrimNet(self):
        self.run_batch(
            Graph2dGNNFinetuneDataPipeline,
            TrimNetModel,
            model_params[get_fqcn(TrimNetModel)],
            self.dataset_args,
        )

    def test_GNNModel_TrimNet_v2(self):
        self.run_batch(
            Graph2dGNNFinetuneDataPipeline,
            TrimNetModel,
            model_params[get_fqcn(TrimNetModel)],
            self.dataset_args,
        )

    def test_GNNModel_TrimNet_v3(self):
        mp = dict(model_params[get_fqcn(TrimNetModel)])
        da = dict(self.dataset_args)
        mp["in_channels"] = 11
        da["include_betti_01"] = True
        self.run_batch(Graph2dGNNFinetuneDataPipeline, TrimNetModel, mp, da)

    def test_GNNModel_Graph2d(self):
        self.run_batch(
            Graph2dFinetuneDataPipeline,
            Graph2dModel,
            model_params[get_fqcn(Graph2dModel)],
            self.dataset_args,
        )

    def test_GNNModel_Graph2d_V2(self):
        da = dict(self.dataset_args)
        da["include_betti_01"] = True
        self.run_batch(
            Graph2dFinetuneDataPipeline,
            Graph2dModel,
            model_params[get_fqcn(Graph2dModel)],
            da,
        )

    # checkpoint file too large
    # def test_GNNModel_Graph2d_v2(self):
    #    self.run_batch(Graph2dFinetuneDataPipeline, Graph2dModel, model_params[get_fqcn(
    #        Graph2dModel)], self.dataset_args, os.path.join(self.ckpt_dir, "Graph2d_model.ckpt"))

    def test_multi_task_prediction_head_single_regression(self):
        input_dim = 512
        num_tasks = 1
        task_type = TaskType.REGRESSION
        batch_size = 4
        input_data = torch.randn(batch_size, input_dim)
        prediction_head = MultiTaskPredictionHead(
            input_dim, num_tasks, task_type, head="mlp", hidden_dims=[512, 512, 512]
        )
        predictions = prediction_head(input_data)
        assert predictions.shape == (batch_size, num_tasks, 1)

    def test_multi_task_prediction_head_multi_regression(self):
        input_dim = 512
        num_tasks = 3
        task_type = TaskType.REGRESSION
        batch_size = 4
        input_data = torch.randn(batch_size, input_dim)
        prediction_head = MultiTaskPredictionHead(
            input_dim, num_tasks, task_type, head="mlp", hidden_dims=[512, 512, 512]
        )
        predictions = prediction_head(input_data)
        assert predictions.shape == (batch_size, num_tasks, 1)

    def test_multi_task_prediction_head_single_classfication(self):
        input_dim = 512
        num_tasks = 1
        task_type = TaskType.CLASSIFICATION
        batch_size = 4
        num_classes_per_task = 2
        input_data = torch.randn(batch_size, input_dim)
        prediction_head = MultiTaskPredictionHead(
            input_dim,
            num_tasks,
            task_type,
            num_classes_per_task,
            head="mlp",
            hidden_dims=[512, 512, 512],
            softmax=True,
        )
        predictions = prediction_head(input_data)
        assert predictions.shape == (batch_size, num_tasks, num_classes_per_task)
        # verify if the softmax has been applied within a task
        logging.info(f"sum = {torch.sum(torch.sum(predictions, dim=-1))}")
        assert torch.sum(torch.sum(predictions, dim=-1)) == batch_size * num_tasks

    def test_multi_task_prediction_head_multi_classfication(self):
        input_dim = 512
        num_tasks = 12
        task_type = TaskType.CLASSIFICATION
        batch_size = 4
        num_classes_per_task = 3
        input_data = torch.randn(batch_size, input_dim)
        prediction_head = MultiTaskPredictionHead(
            input_dim,
            num_tasks,
            task_type,
            num_classes_per_task,
            head="mlp",
            hidden_dims=[512, 512, 512],
            softmax=True,
        )
        predictions = prediction_head(input_data)
        assert predictions.shape == (batch_size, num_tasks, num_classes_per_task)
        # verify if the softmax has been applied within a task
        assert torch.sum(torch.sum(predictions, dim=-1)) == batch_size * num_tasks


if __name__ == "__main__":
    unittest.main()
