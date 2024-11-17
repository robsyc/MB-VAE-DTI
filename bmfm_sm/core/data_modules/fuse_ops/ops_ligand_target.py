import numpy as np
import pandas as pd
from fuse.data.ops.op_base import OpBase
from fuse.utils.ndict import NDict

import bmfm_sm.core.data_modules.namespace as ns


class OpLookupTargetEmbedding(OpBase):
    def __init__(self, target_embedding_file):
        super().__init__()
        self.target_embedding_file = target_embedding_file
        self.target_embeddings = pd.read_csv(
            target_embedding_file,
            index_col=ns.FIELD_INDEX,  # Thi might be slow, find out a better way
            converters={"embedding": lambda x: np.asarray(list(map(float, x.split())))},
        )

    def __call__(
        self, sample_dict: NDict, key_in, key_out, prefix: str | None = None
    ) -> NDict:
        the_id = sample_dict[key_in]
        sample_dict[key_out] = np.vstack(
            self.target_embeddings.loc[the_id].values, dtype="float32"
        )
        return sample_dict


class OpRandomTargetEmbedding(OpBase):
    def __init__(self, target_embedding_file):
        super().__init__()
        self.target_embedding_file = target_embedding_file
        target_embeddings_df = pd.read_csv(
            target_embedding_file,
            index_col=ns.FIELD_INDEX,  # Thi might be slow, find out a better way
            converters={
                "embedding": lambda x: np.array(list(map(float, x.split()))).astype(
                    np.float32
                )
            },
        )
        target_embeddings = np.vstack(target_embeddings_df["embedding"].to_numpy())
        self.mean = np.mean(target_embeddings, axis=0)
        self.cov = np.cov(target_embeddings.T)

        self.random_target_embeddings = pd.DataFrame(
            data=np.random.multivariate_normal(
                self.mean, self.cov, size=target_embeddings.shape[0]
            ),
            index=target_embeddings_df.index,
        )

    def __call__(
        self, sample_dict: NDict, key_in, key_out, prefix: str | None = None
    ) -> NDict:
        the_id = sample_dict[key_in]
        sample_dict[key_out] = np.expand_dims(
            self.random_target_embeddings.loc[the_id].values, axis=0
        ).astype(np.float32)
        return sample_dict


class OpPermutationTargetEmbedding(OpBase):
    def __init__(self, target_embedding_file):
        super().__init__()
        self.target_embedding_file = target_embedding_file
        target_embeddings_df = pd.read_csv(
            target_embedding_file,
            index_col=ns.FIELD_INDEX,  # Thi might be slow, find out a better way
            converters={
                "embedding": lambda x: np.array(list(map(float, x.split()))).astype(
                    np.float32
                )
            },
        )
        target_embeddings = np.vstack(target_embeddings_df["embedding"].to_numpy())
        derangement = self.random_derangement(target_embeddings.shape[0])
        idx = np.empty_like(derangement)
        idx[derangement] = np.arange(len(derangement))
        target_embeddings = target_embeddings[idx, :]

        self.random_target_embeddings = pd.DataFrame(
            data=target_embeddings, index=target_embeddings_df.index
        )

    def random_derangement(self, n):
        while True:
            v = list(range(n))
            for j in range(n - 1, 0, -1):
                p = np.random.randint(0, j)
                if v[p] == j:
                    break
                else:
                    v[j], v[p] = v[p], v[j]
            else:
                if v[0] != 0:
                    return np.array(v)

    def __call__(
        self, sample_dict: NDict, key_in, key_out, prefix: str | None = None
    ) -> NDict:
        the_id = sample_dict[key_in]
        sample_dict[key_out] = np.expand_dims(
            self.random_target_embeddings.loc[the_id].values, axis=0
        ).astype(np.float32)
        return sample_dict
