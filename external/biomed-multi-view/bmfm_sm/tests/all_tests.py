import unittest

from bmfm_sm.core.tests.test_core_data_modules_fuse_ops import (
    TestOpGraphToGraphLaplacian,
    TestOpGraphToRankHomology,
    TestOpSmilesToGraph,
)
from bmfm_sm.predictive.tests.test_predictive_data_modules_graph_finetune_dataset import (
    TestGraph2dFinetuneDataset,
    TestGraphFinetuneDataset,
)
from bmfm_sm.predictive.tests.test_predictive_data_modules_image_finetune import (
    TestImageFinetuneDataset,
)
from bmfm_sm.predictive.tests.test_predictive_data_modules_multimodal_finetune import (
    TestMultiModalFinetuneDataset,
)
from bmfm_sm.predictive.tests.test_predictive_modules_finetune import (
    TestFinetuneLightningModule,
    TestLightningModuleConstructor,
    TestModelConstructor,
)

loader = unittest.TestLoader()
suite = unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule(TestOpSmilesToGraph))
suite.addTests(loader.loadTestsFromModule(TestOpGraphToGraphLaplacian))
suite.addTests(loader.loadTestsFromModule(TestOpGraphToRankHomology))
suite.addTests(loader.loadTestsFromModule(TestGraphFinetuneDataset))
suite.addTests(loader.loadTestsFromModule(TestGraph2dFinetuneDataset))
suite.addTests(loader.loadTestsFromModule(TestImageFinetuneDataset))

suite.addTests(loader.loadTestsFromModule(TestModelConstructor))
suite.addTests(loader.loadTestsFromModule(TestLightningModuleConstructor))
suite.addTests(loader.loadTestsFromModule(TestFinetuneLightningModule))

suite.addTests(loader.loadTestsFromModule(TestMultiModalFinetuneDataset))

runner = unittest.TextTestRunner()
runner.run(suite)
