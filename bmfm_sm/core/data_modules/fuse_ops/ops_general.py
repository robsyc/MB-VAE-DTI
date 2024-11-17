import logging
from collections.abc import Hashable
from multiprocessing import current_process

import numpy as np
import pandas as pd
from fuse.data.ops.op_base import OpBase
from fuse.data.utils.sample import get_sample_id
from fuse.utils.file_io.file_io import read_dataframe
from fuse.utils.ndict import NDict


class OpMakeRandomTrainingExample(OpBase):
    """
    Adds a random example for a given sample id. In order to use this in a pipeline, we need
    to set audit_first_sample=False and audit_rate=None when we instantiate the SampleCatcher.
    This is meant as simple test data.
    """

    def __init__(self, shape: tuple[int, ...], classes: tuple[int, ...]):
        super().__init__()
        self.shape = shape
        self.classes = classes

    def __call__(self, sample_dict: NDict, key_out_data, key_out_label) -> NDict:
        sample_dict[key_out_data] = np.random.rand(*self.shape).astype(np.float32)
        sample_dict[key_out_label] = np.random.choice(self.classes)
        return sample_dict


# Use for CSV files for molecular property prediction tasks (bace, bbbp, tox21, toxcast, muv, hiv, sider and clintox)


class OpFormatMPPLabel(OpBase):
    def __init__(self, include_label_index: list):
        super().__init__()
        self.include_label_index = include_label_index

    def __call__(
        self, sample_dict: NDict, key_in, key_out, task_type="classification"
    ) -> NDict:
        """Coverts the label from a string of consective numbers to a list of those numbers."""
        number_string = sample_dict[key_in]

        label_list = str(number_string).split(" ")

        if self.include_label_index != None:
            l_list = []
            for ind, label in enumerate(label_list):
                if ind in self.include_label_index:
                    l_list.append(label)
            label_list = l_list

        label_list = (
            [int(x) for x in label_list]
            if task_type == "classification"
            else [float(x) for x in label_list]
        )
        label_list = np.asarray(label_list)
        sample_dict[key_out] = label_list

        return sample_dict


class OpPrintSampleId(OpBase):
    def __call__(self, sample_dict: NDict):
        worker_id = current_process().name
        logging.info(
            f"Task {get_sample_id(sample_dict)} processed by worker {worker_id}"
        )

        return sample_dict


class OpReadDataframeBmfm(OpBase):
    """
    Op reading data from pickle file / dataframe object.
    Each row will be added as a value to sample dict.
    """

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        data_filename: str | None = None,
        columns_to_extract: list[str] | None = None,
        rename_columns: dict[str, str] | None = None,
        key_name: str = "data.sample_id",
        key_column: str = "sample_id",
    ):
        """
        :param data:  input DataFrame
        :param data_filename: path to a pickled DataFrame (possible zipped)
        :param columns_to_extract: list of columns to extract from dataframe. When None (default) all columns are extracted
        :param rename_columns: rename columns from dataframe, when None (default) column names are kept
        :param key_name: name of value in sample_dict which will be used as the key/index
        :param key_column: name of the column which use as key/index. In case of None, the original dataframe index will be used to extract the values for a single sample.
        """
        super().__init__()

        # store input
        self._data_filename = data_filename
        self._columns_to_extract = columns_to_extract
        self._rename_columns = rename_columns
        self._key_name = key_name
        self._key_column = key_column
        df = data

        # verify input
        if data is None and data_filename is None:
            msg = "Error: need to provide either in-memory DataFrame or a path to file."
            raise Exception(msg)
        elif data is not None and data_filename is not None:
            msg = "Error: need to provide either 'data' or 'data_filename' args, bot not both."
            raise Exception(msg)

        # read dataframe
        if self._data_filename is not None:
            df = read_dataframe(self._data_filename)

        # extract only specified columns (in case not specified, extract all)
        if self._columns_to_extract is not None:
            df = df[self._columns_to_extract]

        # rename columns
        if self._rename_columns is not None:
            df = df.rename(self._rename_columns, axis=1)

        # convert to dictionary: {index -> {column -> value}}
        if self._key_column is not None:
            df = df.set_index(self._key_column)
        self._data = df.to_dict(orient="index")

    def __call__(
        self, sample_dict: NDict, prefix: str | None = None, prefix_map: dict = None
    ) -> None | dict | list[dict]:
        """
        See base class.

        :param prefix: specify a prefix for the sample dict keys.
                    For example, with prefix 'data.features' and a df with the columns ['height', 'weight', 'sex'],
                    the matching keys will be: 'data.features.height', 'data.features.weight', 'data.features.sex'.
                    Use this OR the prefix_map if prefixes are desired, cannot use both

        :param prefix_map: Specify a dictionary with prefix mapping; the keys should be the names of the columns in the dataframe and the values should be the desired prefix
            For example, with a df with the columns ['smiles', 'k_100', 'k_1000', 'k_10000']
            a prefix map could be {'smiles':'data.input.ligand', 'k_100':'data.input.labels', 'k_1000':'data.input.labels', 'k_10000':'data.input.labels'}
            If a column is not included as a key and was still the columns_to_extract, it will be put at the highest level of the ndict
        """
        if prefix_map is not None and prefix is not None:
            msg = "Error: need to provide either a prefix_map or prefix arg, but not both."
            raise Exception(msg)

        key = sample_dict[self._key_name]

        # locate the required item
        sample_data = self._data[key].copy()

        # add values tp sample_dict
        if prefix_map is not None:
            for name, value in sample_data.items():
                if name not in prefix_map.keys():
                    sample_dict[name] = value
                else:
                    sample_dict[f"{prefix_map[name]}.{name}"] = value
        if prefix is not None:
            for name, value in sample_data.items():
                sample_dict[f"{prefix}.{name}"] = value

        if prefix is None and prefix_map is None:
            for name, value in sample_data.items():
                sample_dict[name] = value

        return sample_dict

    def get_all_keys(self) -> list[Hashable]:
        """:return: list of  dataframe index values"""
        return list(self.data.keys())
