import logging

import torch
import torch.nn as nn

import bmfm_sm.core.data_modules.namespace as ns
from bmfm_sm.core.data_modules.namespace import Modality
from bmfm_sm.core.modules.base_pretrained_model import AttentionalViewAggregator
from bmfm_sm.predictive.modules.graph_2d_models import Graph2dModel
from bmfm_sm.predictive.modules.image_models import ImageModel
from bmfm_sm.predictive.modules.text_models import TextModel


class SmallMoleculeMultiView(nn.Module):
    def __init__(
        self,
        agg_arch="concat",
        agg_gate_input="unprojected",
        agg_weight_freeze="unfrozen",
        inference_mode=False,
    ):
        super().__init__()

        self.model_graph = Graph2dModel(encoder_embed_dim=512)
        self.model_image = ImageModel("ResNet18")
        self.model_text = TextModel()
        self.models = [self.model_graph, self.model_image, self.model_text]
        self.aggregator = AttentionalViewAggregator(
            dim_list=[model.get_embed_dim() for model in self.models],
            arch_type=agg_arch,
            gate_input=agg_gate_input,
        )
        self.agg_weight_freeze = agg_weight_freeze
        self.agg_frozen = None
        self.inference_mode = inference_mode

    def get_separate_outputs(self, batch):
        combined_outputs = {}
        for model in self.models:
            if isinstance(model, TextModel):
                current_output = model.forward0(
                    (batch[ns.FIELD_TOKENIZED_SMILES], batch["attention_mask"])
                )
            elif isinstance(model, Graph2dModel):
                current_output = model.forward0(batch)
            elif isinstance(model, ImageModel):
                current_output = model.forward0(
                    (batch[ns.FIELD_IMAGE], batch.get(ns.FIELD_LABEL, None))
                )

            if self.inference_mode is False:
                assert len(batch[ns.FIELD_LABEL]) == current_output.size(dim=0)
            combined_outputs[model.__class__.__name__] = current_output

        # aggregator_output, model_coeffs = self.aggregator(combined_outputs.values())
        # combined_outputs["aggregator"] = aggregator_output
        # combined_outputs["model_coeffs"] = model_coeffs

        return combined_outputs

    def forward(self, batch):
        combined_outputs = []
        for model in self.models:
            if isinstance(model, TextModel):
                current_output = model.forward0(
                    (batch[ns.FIELD_TOKENIZED_SMILES], batch["attention_mask"])
                )
            elif isinstance(model, Graph2dModel):
                current_output = model.forward0(batch)
            elif isinstance(model, ImageModel):
                current_output = model.forward0(
                    (batch[ns.FIELD_IMAGE], batch.get(ns.FIELD_LABEL, None))
                )

            if self.inference_mode is False:
                assert len(batch[ns.FIELD_LABEL]) == current_output.size(dim=0)

            combined_outputs.append(current_output)

        output, model_coeffs = self.aggregator(combined_outputs)

        if self.inference_mode:
            return output
        else:
            return output, model_coeffs

    def forward0(self, batch):
        return self.forward(batch)

    def get_embeddings(self, x, **kwargs):
        with torch.no_grad():
            if kwargs.get("get_separate_embeddings", False):
                return self.get_separate_outputs(x)
            else:
                return self.forward(x).squeeze()

    def get_embed_dim(self):
        return self.aggregator.get_output_dim()

    @property
    def modality(self):
        return Modality.MULTIVIEW

    # Controls  freezing behavior specifically for the Aggregator (may differ from the BaseModels)
    def change_agg_freeze(self, epoch=0):
        # Beginning of training, start the aggregator as frozen/unfrozen
        if self.agg_frozen is None:
            if (
                self.agg_weight_freeze == "gradual"
                or self.agg_weight_freeze == "frozen"
            ):
                for _, param in self.aggregator.named_parameters():
                    param.requires_grad = False
                self.agg_frozen = True
            else:
                for _, param in self.aggregator.named_parameters():
                    param.requires_grad = True
                self.agg_frozen = False
            logging.info(
                f"Aggregator weights frozen at start of training: {self.agg_frozen}"
            )
            return

        # If aggregator should be kept frozen/unfrozen, change nothing (and verify initial state)
        if self.agg_weight_freeze == "frozen":
            assert self.agg_frozen is True
            return
        if self.agg_weight_freeze == "unfrozen":
            assert self.agg_frozen is False
            return

        # If the weight freeze is supposed to change over time, make the necessary adjustments
        if self.agg_weight_freeze == "gradual":
            if epoch == 0:
                assert self.agg_frozen is True
                for _, param in self.aggregator.named_parameters():
                    assert param.requires_grad == False
                return
            if epoch > 0:
                if self.agg_frozen is False:
                    return  # Already unfrozen, don't need to change
                for _, param in self.aggregator.named_parameters():
                    param.requires_grad = True
                logging.info(f"Aggregator weights have been unfrozen at epoch {epoch}")
                self.agg_frozen = False

    def load_ckpt(self, ckpt):
        checkpoint = (
            torch.load(ckpt)
            if torch.cuda.is_available()
            else torch.load(ckpt, map_location="cpu")
        )

        # Checking if state_dict is nested inside the checkpoint
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]

        is_finetune = any("pred_head" in key for key in checkpoint.keys())

        if is_finetune is False:
            load_status = self.load_state_dict(checkpoint, strict=False)
            logging.info(
                f"Loading pretrain checkpoint for SmallMoleculeMultiView Model - {load_status}"
            )
        else:
            checkpoint = {
                key.replace("model.", ""): value
                for key, value in checkpoint.items()
                if "model." in key
            }
            load_status = self.load_state_dict(checkpoint, strict=False)
            logging.info(
                f"Loading finetune checkpoint for SmallMoleculeMultiView Model - {load_status}"
            )
