import copy
import logging
import os
import unittest

import torch

from bmfm_sm.api.dataset_registry import DatasetRegistry
from bmfm_sm.api.model_registry import ModelRegistry
from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel
from bmfm_sm.core.data_modules.namespace import LateFusionStrategy, Modality
from bmfm_sm.predictive.modules.smmv_model import SmallMoleculeMultiView

if "BMFM_HOME" in os.environ and os.path.exists(os.environ["BMFM_HOME"]):
    local_run = True
else:
    local_run = False
    logging.warning(
        "Not running test_smmv_api because BMFM_HOME env variable is not defined."
    )

example_dataset = DatasetRegistry.get_instance().get_dataset_info("BACE")
example_smiles = example_dataset.get_example_smiles()


# This test checks that when a checkpoint is loaded, whether the model's params are actually updated (For both finetuned and pretrained models)
class TestSMMVCheckpointLoading(unittest.TestCase):
    def test_SmallMoleculeMultiViewModel_PretrainedCheckpointLoading(self):
        if local_run:
            m = SmallMoleculeMultiView(agg_arch="coeff_mlp", inference_mode=True)
            before_sd = copy.deepcopy(m.state_dict())
            ckpt = ModelRegistry.get_checkpoint(Modality.MULTIVIEW)
            m.load_ckpt(ckpt)
            after_sd = m.state_dict()
            assert all_diff(before_sd, after_sd, ["bn", "inv_freq", "final_layer_norm"])

    def test_SmallMoleculeMultiViewModel_FinetunedCheckpointLoading(self):
        if local_run:
            m = SmallMoleculeMultiView(agg_arch="coeff_mlp", inference_mode=True)
            before_sd = copy.deepcopy(m.state_dict())
            ckpt = DatasetRegistry.get_checkpoint(Modality.MULTIVIEW, example_dataset)
            m.load_ckpt(ckpt)
            after_sd = m.state_dict()
            assert all_diff(before_sd, after_sd, ["bn", "inv_freq", "final_layer_norm"])


# This test checks that when a Pretrained/Finetuned checkpoint is loaded twice, the two resultant models match
class TestSMMVModelConsistentLoading(unittest.TestCase):
    def test_SmallMoleculeMultiViewModel_PretrainedConsistentLoading(self):
        if local_run:
            m1 = SmallMoleculeMultiViewModel.from_pretrained(
                LateFusionStrategy.ATTENTIONAL
            )
            m2 = SmallMoleculeMultiViewModel.from_pretrained(
                LateFusionStrategy.ATTENTIONAL
            )
            assert all_match(m1, m2)

    def test_SmallMoleculeMultiViewModel_FinetunedConsistentLoading(self):
        if local_run:
            m1 = SmallMoleculeMultiViewModel.from_finetuned(example_dataset)
            m2 = SmallMoleculeMultiViewModel.from_finetuned(example_dataset)
            assert all_match(m1, m2)


# This test checks that if a Pretrained model is used to generate embeddings, the embeddings are the same each time
class TestConsistentSMMVPretrainedEmbedding(unittest.TestCase):
    def test_SmallMoleculeMultiViewModel_ConsistentPretrainedEmbedding(self):
        if local_run:
            emb1 = SmallMoleculeMultiViewModel.get_embeddings(example_smiles)
            emb2 = SmallMoleculeMultiViewModel.get_embeddings(example_smiles)
            assert torch.equal(emb1, emb2)


# This test checks that when a Finetuned model is used to make a prediction, the prediction is the same each time
class TestConsistentSMMVPredictions(unittest.TestCase):
    def test_SmallMoleculeMultiViewModel_ConsistentPredictions(self):
        if local_run:
            pred1 = SmallMoleculeMultiViewModel.get_predictions(
                example_smiles, example_dataset
            )
            pred2 = SmallMoleculeMultiViewModel.get_predictions(
                example_smiles, example_dataset
            )
            assert torch.isclose(pred1, pred2, atol=0)


# HELPER FUNCTIONS
def all_match(model1, model2):
    all_match = True
    for sd1, sd2 in zip(model1.state_dict().items(), model2.state_dict().items()):
        if not torch.equal(sd1[1], sd2[1]):
            all_match = False
    return all_match


def all_diff(before_sd, after_sd, exceptions=[]):
    all_diff = True
    for sd1, sd2 in zip(before_sd.items(), after_sd.items()):
        if torch.equal(sd1[1], sd2[1]):
            if all(exception not in sd1[0] for exception in exceptions):
                all_diff = False
                print(sd1[0], "is not different for the model")
    return all_diff
