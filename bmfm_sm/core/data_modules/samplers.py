import logging
import math
import random
from collections.abc import Iterator

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler, T_co


class ShuffleLaterDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

    def __iter__(self) -> Iterator[T_co]:
        # Initializing the indices
        indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample according to machine rank
        indices = indices[self.rank : self.total_size : self.num_replicas]

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # type: ignore[arg-type]
            shuffle_order = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in shuffle_order]

        assert len(indices) == self.num_samples
        return iter(indices)


class TokenBudgetSampler(BatchSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_bins: int,
        budget: int,
        is_train: bool = False,
        sampler: Sampler = None,
    ) -> None:
        self.bin_idx, self.bin_len_avg, self.bin_ct = TokenBudgetSampler.generate_bins(
            dataset, dataset.get_feature_fn(), num_bins
        )
        self.budget = budget
        self.is_train = is_train
        self.batch_size = 0
        assert all(
            element != 0 for element in self.bin_len_avg
        ), f"No bin can be empty {self.bin_len_avg}"
        logging.info(f"Using parameters {self.bin_len_avg, self.bin_ct, self.budget}")

    def __iter__(self):
        bins = list(zip(self.bin_idx, self.bin_len_avg))
        if self.is_train:
            random.shuffle(bins)

        for bin, bin_len_avg in bins:
            if self.is_train:
                random.shuffle(bin)
            batch_size = int(self.budget // bin_len_avg)
            if not self.is_train:
                batch_size = int(2 * batch_size)
            assert (
                batch_size > 0
            ), f"batch_size {batch_size} became zero for {self.budget} // {bin_len_avg}"
            for i in range(0, len(bin), batch_size):
                yield bin[i : i + batch_size]

    def __len__(self):
        num_batches = sum(
            len(bucket) // int(self.budget // lengths)
            for bucket, lengths in zip(self.bin_idx, self.bin_len_avg)
        )
        return int(num_batches)

    @staticmethod
    def generate_bins(examples, feature_fn, num_bins):
        X = np.array([feature_fn(example) for example in examples]).reshape(-1, 1)
        kmeans = KMeans(n_clusters=num_bins, random_state=0).fit(X)

        bin_idx = [[] for _ in range(num_bins)]
        bin_len_avg = [0] * num_bins
        bin_ct = [0] * num_bins

        for idx, label in enumerate(kmeans.labels_):
            bin_idx[label].append(idx)
            bin_len_avg[label] += feature_fn(examples[idx])
            bin_ct[label] += 1

        for i in range(num_bins):
            if bin_ct[i] > 0:
                bin_len_avg[i] /= bin_ct[i]

        return bin_idx, bin_len_avg, bin_ct

    @staticmethod
    def get_batches(dataloader):
        l = list(dataloader)
        return l

    @staticmethod
    def get_batch_weights(batches, feature_fn):
        batch_weights = []
        for batch in batches:
            size = 0
            for example in batch:
                size += feature_fn(example)
            batch_weights.append(size)
        return batch_weights
