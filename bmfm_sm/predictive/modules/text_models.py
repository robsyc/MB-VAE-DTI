# -----------------------------------------------------------------------------
# Parts of this file incorporate code from the following open-source project(s):
#   Source: https://github.com/IBM/molformer
#   License: Apache License 2.0
# -----------------------------------------------------------------------------

import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_transformers.attention import AttentionLayer
from fast_transformers.builders.attention_builders import AttentionBuilder
from fast_transformers.builders.transformer_builders import (
    BaseTransformerEncoderBuilder,
)
from fast_transformers.events import QKVEvent
from fast_transformers.feature_maps import GeneralizedRandomFeatures

# imports specific to text model for fine-tuning
from fast_transformers.masking import LengthMask as LM
from fast_transformers.transformers import TransformerEncoder, TransformerEncoderLayer

from bmfm_sm.core.data_modules.namespace import Modality
from bmfm_sm.core.modules.base_pretrained_model import BaseModel


class LMLayer(nn.Module):
    def __init__(self, n_embd, n_vocab):
        super().__init__()
        self.embed = nn.Linear(n_embd, n_embd)
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, n_vocab, bias=False)

    def forward(self, tensor):
        tensor = self.embed(tensor)
        tensor = F.gelu(tensor)
        tensor = self.ln_f(tensor)
        tensor = self.head(tensor)
        return tensor


# source https://github.com/IBM/molformer/blob/main/training/rotate_attention/rotary.py


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            # if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, :, None, :]
            self.sin_cached = emb.sin()[None, :, None, :]
            # else:
            #    cos_return = self.cos_cached[..., :seq_len]
            #    sin_return = self.sin_cached[..., :seq_len]
            #    return cos_return, sin_return

        return self.cos_cached, self.sin_cached


# rotary pos emb helpers:


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    # dim=-1 triggers a bug in earlier torch versions
    return torch.cat((-x2, x1), dim=x1.ndim - 1)


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


# source https://github.com/IBM/molformer/blob/main/training/rotate_attention/attention_layer.py
# The rotate attention layer performs all the query key value projections and
# output projections leaving the implementation of the attention to the inner
# attention module.


class RotateAttentionLayer(AttentionLayer):
    """
    Rotate attention layer inherits from fast_transformer attention layer.
    The only thing added is an Embedding encoding, for more information
    on the attention layer see the fast_transformers code.
    """

    def __init__(
        self,
        attention,
        d_model,
        n_heads,
        d_keys=None,
        d_values=None,
        event_dispatcher="",
    ):
        super().__init__(
            attention,
            d_model,
            n_heads,
            d_keys=d_keys,
            d_values=d_values,
            event_dispatcher=event_dispatcher,
        )

        self.rotaryemb = RotaryEmbedding(d_keys)

    def forward(self, queries, keys, values, attn_mask, query_lengths, key_lengths):
        """
        Using the same frame work as the fast_Transformers attention layer
        but injecting rotary information to the queries and the keys
        after the keys and queries are projected.
        In the argument description we make use of the following sizes.

            - N: the batch size
            - L: The maximum length of the queries
            - S: The maximum length of the keys (the actual length per sequence
              is given by the length mask)
            - D: The input feature dimensionality passed in the constructor as
              'd_model'

        Arguments:
        ---------
            queries: (N, L, D) The tensor containing the queries
            keys: (N, S, D) The tensor containing the keys
            values: (N, S, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of

        Returns:
        -------
            The new value for each query as a tensor of shape (N, L, D).

        """
        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        cos, sin = self.rotaryemb(queries)
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)
        values = self.value_projection(values).view(N, S, H, -1)
        # Let the world know of the qkv
        self.event_dispatcher.dispatch(QKVEvent(self, queries, keys, values))

        # Compute the attention
        new_values = self.inner_attention(
            queries, keys, values, attn_mask, query_lengths, key_lengths
        ).view(N, L, -1)

        # Project the output and return
        return self.out_projection(new_values)


# source https://github.com/IBM/molformer/blob/main/training/rotate_attention/rotate_builder.py


class RotateEncoderBuilder(BaseTransformerEncoderBuilder):
    """
    Build a batch transformer encoder with Relative Rotary embeddings
    for training or processing of sequences all elements at a time.

    Example usage:

        builder = RotateEncoderBuilder()
        builder.n_layers = 12
        builder.n_heads = 8
        builder.feed_forward_dimensions = 1024
        builder.query_dimensions = 64
        builder.value_dimensions = 64
        builder.dropout = 0.1
        builder.attention_dropout = 0.1
        builder.attention_type = "linear"
        transformer = builder.get()
    """

    def _get_attention_builder(self):
        """Return an instance of the appropriate attention builder."""
        return AttentionBuilder()

    def _get_attention_layer_class(self):
        """
        Return the class for the layer that projects queries keys and
        values.
        """
        return RotateAttentionLayer

    def _get_encoder_class(self):
        """Return the class for the transformer encoder."""
        return TransformerEncoder

    def _get_encoder_layer_class(self):
        """Return the class for the transformer encoder layer."""
        return TransformerEncoderLayer


# Text Model for Fine-Tuning


class TextModel(BaseModel):
    def __init__(
        self,
        n_vocab=2362,
        n_embd=768,
        n_layer=12,
        n_head=12,
        num_feats=32,
        d_dropout=0.2,
        seed=12345,
        deterministic_eval=False,
    ):
        super().__init__(Modality.TEXT)

        # short names for classes defined above
        rotate_builder = RotateEncoderBuilder

        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.num_feats = num_feats
        self.d_dropout = d_dropout
        self.seed = seed
        self.n_vocab = n_vocab
        # Word embeddings layer
        # input embedding stem
        builder = rotate_builder.from_kwargs(
            n_layers=self.n_layer,
            n_heads=self.n_head,
            query_dimensions=self.n_embd // self.n_head,
            value_dimensions=self.n_embd // self.n_head,
            feed_forward_dimensions=self.n_embd,
            attention_type="linear",
            feature_map=partial(
                GeneralizedRandomFeatures,
                n_dims=self.num_feats,
                deterministic_eval=deterministic_eval,
            ),
            activation="gelu",
        )
        self.pos_emb = None
        self.tok_emb = nn.Embedding(n_vocab, self.n_embd)
        self.drop = nn.Dropout(self.d_dropout)
        # transformer
        self.blocks = builder.get()

    def forward(self, idx, mask):
        # each index maps to a (learnable) vector
        token_embeddings = self.tok_emb(idx)
        x = self.drop(token_embeddings)
        x = self.blocks(x, length_mask=LM(mask.sum(-1)))

        token_embeddings = x

        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        emb_layer = sum_embeddings / sum_mask

        return emb_layer

    def forward0(self, batch):
        idx = batch[0]
        mask = batch[1]
        return self.forward(idx, mask)

    def get_embed_dim(self):
        return self.n_embd

    def load_ckpt(self, path_to_ckpt):
        if "molformer" not in path_to_ckpt:
            return super().load_ckpt(path_to_ckpt)

        # Logic for specifically loading in a Molformer checkpoint
        if torch.cuda.is_available():
            checkpoint = torch.load(path_to_ckpt)
        else:
            checkpoint = torch.load(path_to_ckpt, map_location=torch.device("cpu"))
        new_state_dict = {}
        for model_key in self.state_dict().keys():
            if model_key not in checkpoint["state_dict"].keys():
                raise KeyError(f"Key {model_key} not found in checkpoint")
            elif (
                self.state_dict()[model_key].shape
                != checkpoint["state_dict"][model_key].shape
            ):
                raise KeyError(f"Size mistmatch for {model_key} tensor")
            else:
                new_state_dict[model_key] = checkpoint["state_dict"][model_key]
        self.load_state_dict(new_state_dict)
        logging.info("loaded ckpt: %s" % path_to_ckpt)

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for layer in self.blocks.layers:
            layer.attention.inner_attention.feature_map.deterministic_eval = (
                not self.training
            )
        logging.info(
            f"in train {mode} setting deterministic_eval = {not self.training}"
        )
        for module in self.children():
            module.train(mode)
        return self


class TextModelForPretraining(BaseModel):
    def __init__(
        self,
        n_vocab=2362,
        n_embd=768,
        n_layer=12,
        n_head=12,
        num_feats=32,
        d_dropout=0.2,
        seed=12345,
        **kwargs,
    ):
        super().__init__(Modality.TEXT)
        builder = RotateEncoderBuilder.from_kwargs(
            n_layers=n_layer,
            n_heads=n_head,
            query_dimensions=n_embd // n_head,
            value_dimensions=n_embd // n_head,
            feed_forward_dimensions=n_embd,
            attention_type="linear",
            feature_map=partial(
                GeneralizedRandomFeatures,
                n_dims=num_feats,
                deterministic_eval=kwargs.get("deterministic_eval", False),
            ),
            activation="gelu",
        )
        self.tok_emb = nn.Embedding(n_vocab, n_embd)
        self.drop = nn.Dropout(d_dropout)
        # transformer
        self.blocks = builder.get()

    def forward(self, x):
        token_embeddings = self.tok_emb(x)
        x = self.drop(token_embeddings)
        x = self.blocks(x)
        return x

    def forward0(self, x):
        return self.forward(x)

    def get_embeddings_w_mask(self, idx, mask):
        token_embeddings = self.get_embeddings(idx)
        with torch.no_grad():
            input_mask_expanded = (
                mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            emb_layer = sum_embeddings / sum_mask
            return emb_layer

    def get_embeddings(self, x):
        with torch.no_grad():
            return self.forward(x)
