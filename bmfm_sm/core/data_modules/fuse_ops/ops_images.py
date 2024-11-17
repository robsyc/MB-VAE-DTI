# Adopted from the ImageMol team and the ImageMol Repository - https://github.com/HongxinXiang/ImageMol/

# Fuse Operations for Image Modality
import logging
import os
from io import BytesIO
from random import random

import IPython
import numpy as np
import torch
from fuse.data.ops.op_base import OpBase
from fuse.utils.ndict import NDict
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from torchvision import transforms

import bmfm_sm.core.data_modules.namespace as ns

### GLOBAL VARIABLES ###

# Based on ImageMol Repo, can change the values or add other transformations as desired
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

img_tra = [
    transforms.CenterCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomRotation(degrees=360),
]
tile_tra = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomRotation(degrees=360),
    transforms.ToTensor(),
]
img_transformer = transforms.Compose(img_tra)
tile_transformer = transforms.Compose(tile_tra)


### OPERATIONS ###


class OpSmilesToImage(OpBase):
    def __init__(self, size: int = 224):
        super().__init__()
        self.size = size

    def __call__(
        self,
        sample_dict: NDict,
        key_in=ns.FIELD_DATA_LIGAND_SMILES,
        key_out=ns.FIELD_DATA_LIGAND_IMAGE,
    ) -> NDict:
        smi = sample_dict[key_in]
        try:
            mol = Chem.MolFromSmiles(smi)
            img = Draw.MolsToGridImage(
                [mol], molsPerRow=1, subImgSize=(self.size, self.size)
            )

            if (
                type(img) == IPython.core.display.Image
            ):  # Usually when calling from Jupyter
                img = Image.open(BytesIO(img.data)).convert("RGB")
            else:
                img = img.convert("RGB")

        except Exception as e:
            logging.info(
                f"Can not draw the image from the input SMILES {smi} using rdkit! Error {e}, Creating a dummy image"
            )
            img = Image.new("RGB", (self.size, self.size))
        sample_dict[key_out] = img
        return sample_dict


class OpImageSampleIDDecode(OpBase):
    def __call__(
        self,
        sample_dict: NDict,
        key_in=ns.FIELD_DATA_SAMPLE_ID,
        key_out=ns.FIELD_PATHS_IMAGE,
    ) -> NDict:
        """Decodes the sample id into image file name and saves it under the path key."""
        img_id = sample_dict[key_in]
        sample_dict[key_out] = str(img_id) + ".png"

        return sample_dict


class OpLoadImageCustom(OpBase):
    """
    Loads in the images using the image_path stored in the nested dictionary. Preferred alternative to OpLoadImage (built into fuse) so it directly loads in PIL Images
    instead of having to convert the files to tensor then PIL Images.
    """

    def __init__(self, dir_path: str, **kwargs):
        super().__init__(**kwargs)
        self._dir_path = dir_path

    def __call__(
        self,
        sample_dict: NDict,
        key_in=ns.FIELD_PATHS_IMAGE,
        key_out=ns.FIELD_DATA_LIGAND_IMAGE,
    ) -> NDict:
        img_filename = os.path.join(self._dir_path, sample_dict[key_in])
        sample_dict[key_out] = (
            Image.open(img_filename).convert("RGB").resize((224, 224))
        )

        return sample_dict


class OpPreprocessImage(OpBase):
    """Runs the image transformation pipeline on the images to create image augmentations."""

    def __call__(
        self,
        sample_dict: NDict,
        key_in=ns.FIELD_DATA_LIGAND_IMAGE,
        key_out=ns.FIELD_DATA_LIGAND_AUG_IMG_TEMP,
        img_transf_func=None,
    ) -> NDict:
        input_img = sample_dict[key_in]

        img_aug = (
            img_transf_func(input_img)
            if img_transf_func
            else img_transformer(input_img)
        )

        sample_dict[key_out] = img_aug

        return sample_dict


class OpGenerateImageJigsaw(OpBase):
    """
    Will take an image, break it into a grid of 3x3, shuffle these patches according to a chosen permutation class, concatenate into 1 image,
    run tile transformations on this image, and store this new image along with the permtuation and class in the fuse structure.
    """

    def get_tile(img, n):
        w = float(img.size[0]) / 3
        y = int(n / 3)
        x = n % 3
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        return tile

    def get_permutations():
        current_script_path = os.path.realpath(__file__)
        relative_path = os.path.join(
            os.path.dirname(current_script_path), "./permutations_100.npy"
        )
        perms = np.load(relative_path)
        if perms.min() == 1:
            perms = perms - 1
        return perms

    def concatPILImage(patches):
        assert len(patches) == 9
        h, w = patches[0].size
        target = Image.new("RGB", (h * 3, w * 3))
        for i in range(len(patches)):
            a = h * (i % 3)
            b = w * (i // 3)
            c = h * (i % 3) + w
            d = w * (i // 3) + h
            target.paste(patches[i], (a, b, c, d))
        return target

    def get_tile_data(img, bias_whole_image, grid_size=3):
        img = img.resize((222, 222))
        n_grids = grid_size**2

        tiles = [None] * n_grids
        for n in range(n_grids):
            tiles[n] = OpGenerateImageJigsaw.get_tile(img, n)

        permutations = OpGenerateImageJigsaw.get_permutations()
        # Adding the un-jumbled permutation as the 0th class
        permutations = np.insert(permutations, 0, np.arange(n_grids), axis=0)

        # Added 1 for class 0: unsorted
        order = np.random.randint(len(permutations))

        # Bias towards keeping the original image
        if bias_whole_image and bias_whole_image > random():
            order = 0

        data = [
            tiles[permutations[order][t]] for t in range(n_grids)
        ]  # Permuting the order of the tiles

        data = OpGenerateImageJigsaw.concatPILImage(data).resize((224, 224))
        data = tile_transformer(data)

        return data, order

    def __call__(
        self,
        sample_dict: NDict,
        key_in=ns.FIELD_DATA_LIGAND_AUG_IMG_TEMP,
        img_key_out=ns.FIELD_DATA_LIGAND_JIGSAW_IMG,
        class_key_out=ns.FIELD_DATA_JIGSAW_ORDER,
        original_image_rate=0.8,
    ) -> NDict:
        """Splits the image into a 3x3 grid of tiles."""
        img = sample_dict[key_in]
        shuffled_image, shuffled_class = OpGenerateImageJigsaw.get_tile_data(
            img, bias_whole_image=original_image_rate
        )

        sample_dict[img_key_out] = shuffled_image
        sample_dict[class_key_out] = shuffled_class

        return sample_dict


class OpGenerateImageMasks(OpBase):
    """
    Generates masks for the original molecular image, along with a 64x64 smaller image. Will also run image transformations on these image to create augmentations.
    Will store all the required information for the Image model (the images, the jigsaw, the order, the masks, etc.) in the fuse data structure.
    """

    def get_mask_data(
        data_non_mask,
        data64_non_mask,
        mask_type,
        mask_shape_h,
        mask_shape_w,
        mask_ratio,
    ):
        c, h, w = data_non_mask.shape
        if mask_type == "random_mask":
            mask_matrix = create_random_mask(shape=(1, h, w), mask_ratio=mask_ratio)[0]
            mask64_matrix = create_random_mask(
                shape=(1, data64_non_mask.shape[1], data64_non_mask.shape[2]),
                mask_ratio=mask_ratio,
            )[0]
        elif mask_type == "rectangle_mask":
            mask_matrix = create_rectangle_mask(
                shape=(1, h, w), mask_shape=(mask_shape_h, mask_shape_w)
            )[0]
            mask64_matrix = create_rectangle_mask(
                shape=(1, data64_non_mask.shape[1], data64_non_mask.shape[2]),
                mask_shape=(mask_shape_h, mask_shape_w),
            )[0]
        elif mask_type == "mix_mask":
            if random() > 0.5:
                mask_matrix = create_random_mask(
                    shape=(1, data64_non_mask.shape[1], data64_non_mask.shape[2]),
                    mask_ratio=mask_ratio,
                )[0]
            else:
                mask_matrix = create_rectangle_mask(
                    shape=(1, data64_non_mask.shape[1], data64_non_mask.shape[2]),
                    mask_shape=(mask_shape_h, mask_shape_w),
                )[0]
        # starting mask
        data_mask = data_non_mask.clone()
        data64_mask = data64_non_mask.clone()
        for i in range(3):  # 3 channels
            data_mask[i][torch.from_numpy(mask_matrix) == 1] = torch.mean(data_mask[i])
            data64_mask[i][torch.from_numpy(mask64_matrix) == 1] = torch.mean(
                data64_mask[i]
            )

        return data_mask, data64_mask

    def __call__(
        self,
        sample_dict: NDict,
        key_in=ns.FIELD_DATA_LIGAND_AUG_IMG_TEMP,
        cl_mask_type="rectangle_mask",
        cl_mask_shape_h=16,
        cl_mask_shape_w=16,
        cl_mask_ratio=0.001,
    ) -> NDict:
        img = sample_dict[key_in]
        img64 = img.resize((64, 64))
        jigsaw_img = sample_dict[ns.FIELD_DATA_LIGAND_JIGSAW_IMG]

        # Converts them to tensors
        img_nonmask = tile_transformer(img)
        img64_nonmask = tile_transformer(img64)

        cl_data_mask, _ = OpGenerateImageMasks.get_mask_data(
            img_nonmask,
            img64_nonmask,
            mask_type=cl_mask_type,
            mask_shape_h=cl_mask_shape_h,
            mask_shape_w=cl_mask_shape_w,
            mask_ratio=cl_mask_ratio,
        )

        if normalize != None:
            data = normalize(jigsaw_img)
            data64_non_mask, data_non_mask = normalize(img64_nonmask), normalize(
                img_nonmask
            )
            cl_data_mask = normalize(cl_data_mask)

        sample_dict[ns.FIELD_DATA_LIGAND_JIGSAW_IMG] = data
        sample_dict[ns.FIELD_DATA_LIGAND_AUG_MASK_IMG] = cl_data_mask
        sample_dict[ns.FIELD_DATA_LIGAND_AUG_NONMASK_IMG] = data_non_mask
        sample_dict[ns.FIELD_DATA_LIGAND_AUG_NONMASK_IMG_SMALL] = data64_non_mask

        return sample_dict


class OpGenerateClusterLabels(OpBase):
    """Concatenates all the cluster labels into one label list and stores that under the labels key of the sample dict."""

    def __call__(
        self,
        sample_dict: NDict,
        cluster_labels,
        key_in=ns.FIELD_DATA_CLUSTER,
        key_out=ns.FIELD_DATA_CLUSTER_ALL,
    ) -> NDict:
        # cluster_labels should be a list of the desired cluster types. E.g. ['k_100', 'k_1000', 'k_3000']

        cluster_vals = [
            sample_dict[f"{key_in}.{cluster_label}"] for cluster_label in cluster_labels
        ]

        sample_dict[key_out] = cluster_vals
        return sample_dict


class OpNormalizeImage(OpBase):
    """Runs the normalize transformations (either custom passed in normalizations or the global ones at the top of this file)."""

    def __call__(
        self, sample_dict: NDict, key_in, key_out, normalize_func=None
    ) -> NDict:
        input_tensor = sample_dict[key_in]

        normalized_tensor = (
            normalize_func(input_tensor) if normalize_func else normalize(input_tensor)
        )

        sample_dict[key_out] = normalized_tensor

        return sample_dict


### HELPER FUNCTIONS ###


def create_random_mask(shape=(512, 224, 224), mask_ratio=0.1):
    """Get a masked image with the specified shape."""
    # :param shape: image shape (batchsize, h, w), which are 1 and 0, 1 means to mask, 0 means not to mask
    # :param mask_ratio: the ratio of masked area
    # :return:

    bsz, h, w = shape[0], shape[1], shape[2]
    mask_format = np.zeros(h * w)
    mask_format[: int(h * w * mask_ratio)] = 1

    mask_matrix = []
    for _ in range(bsz):
        np.random.shuffle(mask_format)
        mask_matrix.append(mask_format.reshape(h, w))
    mask_matrix = np.array(mask_matrix, dtype=int)

    return mask_matrix


def create_rectangle_mask(shape=(512, 224, 224), mask_shape=(16, 16)):
    """Get a masked image with the specified shape."""
    # :param shape: image shape (batchsize, h, w), which are 1 and 0, 1 means to mask, 0 means not to mask
    # :param mask_shape: The shape size of the masked area
    # :return:

    bsz, h, w = shape[0], shape[1], shape[2]
    assert h == w
    xs = np.random.randint(w, size=bsz)
    ys = np.random.randint(h, size=bsz)

    mask_matrix = []
    for i in range(bsz):
        x = xs[i]
        y = ys[i]
        mask_format = np.zeros((h, w))
        mask_format[x : x + mask_shape[0], y : y + mask_shape[1]] = 1
        mask_matrix.append(mask_format)

    mask_matrix = np.array(mask_matrix, dtype=int)

    return mask_matrix
