# -----------------------------------------------------------------------------
# Parts of this file incorporate code from the following open-source project(s):
#   Source: https://github.com/HongxinXiang/ImageMol/blob/master/model/cnn_model_utils.py
#   License: MIT License
# -----------------------------------------------------------------------------

import logging
import math

import torch
import torch.nn as nn
import torchvision

from bmfm_sm.core.data_modules.namespace import Modality
from bmfm_sm.core.modules.base_pretrained_model import BaseModel


class ImageModel(BaseModel):
    def __init__(self, baseModel):
        super().__init__(Modality.IMAGE)

        assert baseModel in [
            "ResNet18",
            "ResNet34",
            "ResNet50",
            "ResNet101",
            "ResNet152",
        ]
        self.embedding_layer = nn.Sequential(
            *list(load_model(baseModel).children())[:-1]
        )
        self.bn = nn.BatchNorm1d(512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # Xavier Initialization for Convolutional Layers
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                # Initializing Batch Norm layers with weight params = 1, bias params = 0
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.embedding_layer(x)
        x = x.view(x.size(0), -1)
        return x

    def forward0(self, batch):
        x = batch[0]
        return self.forward(x)

    def get_embed_dim(self):
        return 512

    # TODO: Check why ImageMol initialized the BatchNorm but wasn't using it in the forward, was it just a mistake?
    def get_embeddings(self, images):
        with torch.no_grad():
            self.embedding_layer.eval()
            embeddings = self.embedding_layer(images)
            embeddings = embeddings.view(embeddings.size(0), -1)
            # embeddings = self.bn(embeddings)
            return embeddings

    def load_ckpt(self, path_to_ckpt, strict=False):
        if torch.cuda.is_available():
            ckpt = torch.load(path_to_ckpt)
        else:
            ckpt = torch.load(path_to_ckpt, map_location=torch.device("cpu"))

        # Check if you're loading from a Lora checkpoint
        if "lora_base_model" in ckpt["state_dict"].keys():
            print("Restoring the Lora Model")
            from bmfm_sm.api.model_registry import ModelRegistry
            from bmfm_sm.core.data_modules.namespace import Modality
            from bmfm_sm.predictive.modules.finetune_lightning_module import (
                FineTuneLightningModule,
            )

            # Have to add the lora layers to self
            FineTuneLightningModule.replace_with_lora(self)

            # Restore the base pretrained checkpoint
            base_checkpoint = ModelRegistry.get_checkpoint(Modality.IMAGE)
            self.load_ckpt(base_checkpoint, strict=False)

            # Restore the lora checkpoint
            self.load_state_dict(ckpt["state_dict"]["lora_base_model"], strict=False)

        else:
            # Checkpoint has prefix model_image in pretraining and model.model in finetuning, and external checkpoints may have neither
            model_key_sample = next(
                (key for key in ckpt["state_dict"].keys() if "embedding_layer" in key),
                None,
            )
            checkpoint_prefix = model_key_sample.split("embedding_layer")[0]
            fixed_state_dict = {}
            if checkpoint_prefix:
                for key in ckpt["state_dict"]:
                    if checkpoint_prefix in key:
                        fixed_state_dict[key.split(checkpoint_prefix)[1]] = ckpt[
                            "state_dict"
                        ][key]
            else:
                fixed_state_dict = ckpt["state_dict"]

            print(self.load_state_dict(fixed_state_dict, strict=strict))
            logging.info("loaded ckpt: %s" % path_to_ckpt)


# UTILITY FUNCTIONS
def load_model(modelname="ResNet18", num_classes=2):
    assert modelname in ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]

    if modelname == "ResNet18":
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet34":
        model = torchvision.models.resnet34(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet50":
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet101":
        model = torchvision.models.resnet101(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet152":
        model = torchvision.models.resnet152(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise Exception(f"{modelname} is undefined")
    return model


class ImageGenerator(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.num_channels = 3
        self.noise_dim = 100
        self.embed_dim = input_dim
        self.latent_dim = 128
        self.ngf = 64

        self.projection = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=self.latent_dim),
            nn.BatchNorm1d(num_features=self.latent_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.netG = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (num_channels) x 64 x 64
        )

    def forward(self, embed_vector):
        latent_vector = self.projection(embed_vector).unsqueeze(2).unsqueeze(3)
        output = self.netG(latent_vector)
        return output


class ImageDiscriminator(nn.Module):
    def __init__(self, nc, ndf):
        """
        ndf --> number of filters in the discriminator
        nc --> number of channels in the images.
        """
        super().__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )
