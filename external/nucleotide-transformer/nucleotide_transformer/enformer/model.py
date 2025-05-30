from dataclasses import dataclass
from typing import Callable, Dict, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import jmp
from typing_extensions import TypeAlias

from nucleotide_transformer.enformer.layers import (
    Attention,
    AttentionPool,
    ConvBlock,
    ResidualConvBlock,
    exponential_linspace_int,
    gelu_fn,
)

SequenceMask: TypeAlias = jnp.ndarray


@dataclass
class EnformerConfig:

    embed_dim: int = 1536
    num_transformer_layers: int = 11
    num_attention_heads: int = 8
    num_human_output_heads: int = 5313
    num_mouse_output_heads: int = 1643
    target_length: int = 896
    attention_dim_key: int = 64
    use_checkpointing: bool = False
    use_convnext: bool = False
    # genetic sequence is downsampled 2 ** 7 == 128x in default Enformer
    # - can be changed for higher resolution
    num_downsamples: int = 7
    dim_divisible_by: int = 128
    positional_encoding_type: str = "enformer"


class Enformer(hk.Module):
    def __init__(
        self,
        config: EnformerConfig,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._config = config
        self._half_dim = config.embed_dim // 2
        self._twice_dim = config.embed_dim * 2

        # A,C,G,T,N token IDs
        self._nucl_token_IDs = {
            "A": 10,
            "C": 12,
            "G": 13,
            "T": 11,
            "N": 14,
        }

    def _replace_token_ids(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """
        Converts token IDs from the NucleotidesKmersTokenizer
        (A:10 / C:12 / G:13 / T:11 / N:14) into (A:0 / C:1 / G:2 / T:3 / N:4)
        to be able to use jax.nn.one_hot to create one-hot encoded vectors.

        Args:
            tokens: e.g. [14 12 13 10]

        Returns:
            Updated token IDs (e.g. [4 1 2 0])
        """
        tokens = jnp.where(tokens == 10, 0, tokens)  # 10/A --> [1, 0, 0, 0]
        tokens = jnp.where(tokens == 11, 3, tokens)  # 11/T --> [0, 0, 0, 1]
        tokens = jnp.where(tokens == 12, 1, tokens)  # 12/C --> [0, 1, 0, 0]
        tokens = jnp.where(tokens == 13, 2, tokens)  # 13/G --> [0, 0, 1, 0]
        tokens = jnp.where(tokens == 14, 4, tokens)  # 14/N --> [0, 0, 0, 0]

        return tokens

    def _one_hot_encode(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """
        Transforms a sequence of tokens into the corresponding one-hot vector.

        Args:
            tokens: e.g. [4 1 2 0]

        Returns:
            corresponding one-hot array of shape (seq_length, 4).
                e.g.
                [[0. 0. 0. 0.]
                [0. 1. 0. 0.]
                [0. 0. 1. 0.]
                [1. 0. 0. 0.]]
        """
        new_tokens = self._replace_token_ids(tokens)
        one_hots = jax.nn.one_hot(new_tokens, 4)

        return one_hots

    def _batch_one_hot_encode(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Transforms a batch of token sequences into the corresponding
        batch of one-hot vectors.

        Args:
            x: list of token sequences.

        Returns:
            corresponding array of shape (batch_size, seq_length, 4)
        """
        batch = [self._one_hot_encode(seq) for seq in x]
        jmp_policy = hk.mixed_precision.current_policy()
        if jmp_policy is not None:
            dtype = jmp_policy.compute_dtype
        else:
            dtype = jnp.float32
        batch = jnp.asarray(batch, dtype=dtype)

        return batch

    def _stem(self, x: jnp.ndarray, is_training: bool = False) -> jnp.ndarray:

        conv = hk.Conv1D(
            output_channels=self._half_dim,
            kernel_shape=15,
            padding=(7, 7),
            data_format="NCW",
        )
        res = ResidualConvBlock(dim=self._half_dim)
        attention_pool = AttentionPool(dim=self._half_dim, pool_size=2)

        x = conv(x)
        x = res(x, is_training)
        x = attention_pool(x)

        return x

    def _conv_tower(self, x: jnp.ndarray, is_training: bool = False) -> jnp.ndarray:

        filter_list = exponential_linspace_int(
            self._half_dim,
            self._config.embed_dim,
            num=(self._config.num_downsamples - 1),
            divisible_by=self._config.dim_divisible_by,
        )
        filter_list = [self._half_dim, *filter_list]

        for i, (dim_in, dim_out) in enumerate(zip(filter_list[:-1], filter_list[1:])):

            with hk.experimental.name_scope(f"layer_{i}"):
                conv = ConvBlock(dim=dim_in, dim_out=dim_out, kernel_size=5)
                res_conv = ResidualConvBlock(
                    dim=dim_out, dim_out=dim_out, kernel_size=1
                )
                attn_pool = AttentionPool(dim=dim_out, pool_size=2)

            x = conv(x, is_training)
            x = res_conv(x, is_training)
            x = attn_pool(x)

        return x

    @hk.transparent
    def _attention_layer(self) -> Attention:
        return Attention(  # type: ignore
            dim=self._config.embed_dim,
            heads=self._config.num_attention_heads,
            dim_key=self._config.attention_dim_key,
            dim_value=self._config.embed_dim // self._config.num_attention_heads,
            num_rel_pos_features=self._config.embed_dim
            // self._config.num_attention_heads,
            positional_encoding_type=self._config.positional_encoding_type,
            name="attention",
        )

    def _transformer_attention(self, x: jnp.ndarray, layer_num: int) -> jnp.ndarray:

        with hk.experimental.name_scope(f"layer_{layer_num}"):
            layer_norm = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True, name="attn_layer_norm"
            )
            attn = self._attention_layer()

        y = layer_norm(x)
        y = attn(y)

        return x + y

    def _transformer_ffn_block(self, x: jnp.ndarray, layer_num: int) -> jnp.ndarray:

        with hk.experimental.name_scope(f"layer_{layer_num}"):
            layer_norm = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True, name="ffn_layer_norm"
            )
            linear_1 = hk.Linear(output_size=self._config.embed_dim * 2, name="ffn_1")
            linear_2 = hk.Linear(output_size=self._config.embed_dim, name="ffn_2")

            y = layer_norm(x)
            y = linear_1(y)
            y = jax.nn.relu(y)
            y = linear_2(y)

        return x + y

    def _transformer_tower(self, x: jnp.ndarray) -> jnp.ndarray:
        for i in range(self._config.num_transformer_layers):
            x = self._transformer_attention(x, layer_num=i)
            x = self._transformer_ffn_block(x, layer_num=i)
        return x

    def _target_length_crop(self, x: jnp.ndarray) -> jnp.ndarray:
        seq_len, target_len = x.shape[-2], self._config.target_length

        if target_len == -1:
            return x

        if seq_len < target_len:
            raise ValueError(
                f"sequence length {seq_len} is less than target length {target_len}"
            )

        trim = (target_len - seq_len) // 2

        if trim == 0:
            return x

        return x[:, -trim:trim]

    def _final_pointwise(
        self, x: jnp.ndarray, is_training: bool = False
    ) -> jnp.ndarray:
        filter_list = exponential_linspace_int(
            self._half_dim,
            self._config.embed_dim,
            num=(self._config.num_downsamples - 1),
            divisible_by=self._config.dim_divisible_by,
        )
        filter_list = [self._half_dim, *filter_list]

        conv = ConvBlock(dim=filter_list[-1], dim_out=self._twice_dim, kernel_size=1)

        x = jnp.transpose(x, (0, 2, 1))
        x = conv(x, is_training)
        x = jnp.transpose(x, (0, 2, 1))
        x = gelu_fn(x)

        return x

    def _heads(self, x: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        linear_human = hk.Linear(
            output_size=self._config.num_human_output_heads, name="human_head"
        )
        linear_mouse = hk.Linear(
            output_size=self._config.num_mouse_output_heads, name="mouse_head"
        )

        x_human = jax.nn.softplus(linear_human(x))
        x_mouse = jax.nn.softplus(linear_mouse(x))

        return {"human_head": x_human, "mouse_head": x_mouse}

    def __call__(
        self, tokens: jnp.ndarray, is_training: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """
        Calls the Enformer model.

        Args:
            tokens: Input tokens out of the NucleotidesKmersTokenizer
                of shape (batch_size, seq_len). Token IDs should ONLY
                contain tokens for A,T,C,G,N nucleotides.
            is_training: Pass to True during training, will change the
                behaviour of Convolutional layers.

            # NOTE: the input tokens are expected to be vectors with
                token IDs corresponding to ONLY A,T,C,G,N nucleotides.
                The token IDs of A,T,C,G,N should be the ones by default
                of NucleotidesKmersTokenizer: A:10 / C:12 / G:13 / T:11 / N:14.
                If the token IDs are different, the one-hot encoded vectors
                will not match the nucleotides and it will fail.
                Sequences cannot have the pad token - to pad the sequences
                you can add the nucleotide N.

        Returns:
            Dictionary containing the final embeddings and logits
                for the human and mouse head.

        """

        # one-hot encoding
        x = self._batch_one_hot_encode(tokens)

        # Enformer model
        x = jnp.transpose(x, (0, 2, 1))
        x = self._stem(x, is_training=is_training)

        x = self._conv_tower(x, is_training=is_training)
        x = jnp.transpose(x, (0, 2, 1))

        x = self._transformer_tower(x)

        outs = {}
        outs["embedding_transformer_tower"] = x

        x = self._target_length_crop(x)

        x = self._target_length_crop(x)

        x = self._final_pointwise(x, is_training=is_training)

        outs["embedding"] = x

        # human and mouse heads
        heads_outs = self._heads(x)
        outs.update(heads_outs)

        return outs


def build_enformer_with_head_fn(
    config: EnformerConfig,
    head_fn: Callable[
        [], Callable[[jnp.ndarray, SequenceMask], Dict[str, jnp.ndarray]]
    ],
    embedding_name: str = "embedding_transformer_tower",
    compute_dtype: jnp.dtype = jnp.float32,
    param_dtype: jnp.dtype = jnp.float32,
    output_dtype: jnp.dtype = jnp.float32,
    name: Optional[str] = None,
) -> Callable:
    """
    Create the model's forward pass for Enformer,

    Args:
        config: Configuration data class containing the hyperparameters for the Enformer
            forward function.
        head_fn: Wrapper initializing a Classification/Regression head. The head cannot
            be passed directly as haiku modules cannot be initialized outside
            hk.transform.
        embedding_name: Which embeddings to use from the enformer pre-trained model as
            input to the head. It should be the name of the key inside model
            predictions ( outs ). Default is "embedding_transformer_tower".
        compute_dtype: the type of the activations. fp16 runs faster and is lighter in
            memory. bf16 handles better large int, and is hence more stable ( it avoids
            float overflows ).
        param_dtype: if compute_dtype is fp16, the model weights will be cast to fp16
            during the forward pass anyway. So in inference mode ( not training mode ),
            it is better to use params in fp16 if compute_dtype is fp16 too
        output_dtype: the output type of the model. it determines the float precision
            of the gradient when training the model.
            NOTE: when training, the gradient is often accumulated in fp32, therefore
            output_dtype need to be in fp32.
        name: the name of the model. example: enformer.

        Example of the function being used with a classification head:
        The classification head is wrapped inside head_fn because
        haiku modules cannot be instantiated outside hk.transform.
        def head_fn():
            return SimpleClassificationHead(num_classes=num_classes)
        finetune_forward_fn = build_esm_ia3_rescaling_with_head_fn(
            model_config=config, head_fn=head_fn, model_name=model_name,
        )
        finetune_forward_fn = hk.transform(finetune_forward_fn)

        # NOTE: the input tokens for enformer of shape (batch_size, seq_len)
            are expected to be vectors with token IDs corresponding
            to ONLY A,T,C,G,N nucleotides.
            The token IDs of A,T,C,G,N should be the ones by default
            of NucleotidesKmersTokenizer: A:10 / C:12 / G:13 / T:11 / N:14.
            If the token IDs are different, the one-hot encoded vectors
            will not match the nucleotides and it will fail.
            Sequences cannot have the pad token - to pad the sequences
            you can add the nucleotide N.

    Returns:
        Enformer model forward function with IA³ rescaling and indicated head.
    """

    assert {compute_dtype, param_dtype, output_dtype}.issubset(
        {
            jnp.bfloat16,
            jnp.float32,
            jnp.float16,
        }
    ), f"provide a dtype in {jnp.bfloat16, jnp.float32, jnp.float16}"

    policy = jmp.Policy(
        compute_dtype=compute_dtype, param_dtype=param_dtype, output_dtype=output_dtype
    )
    hk.mixed_precision.set_policy(Enformer, policy)

    # Remove it in batch norm to avoid instabilities
    norm_policy = jmp.Policy(
        compute_dtype=jnp.float32, param_dtype=param_dtype, output_dtype=compute_dtype
    )
    hk.mixed_precision.set_policy(hk.LayerNorm, norm_policy)
    hk.mixed_precision.set_policy(hk.BatchNorm, norm_policy)

    def enformer_fn(
        tokens: jnp.ndarray,
        is_training: bool = False,
        sequence_mask: Optional[SequenceMask] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Forward pass."""
        # Run the encoder over the inputs.
        encoder = Enformer(config, name)
        outs = encoder(tokens, is_training)
        # NOTE: For now we don't remove the Enformer human/mouse prediction heads
        # since they will not be finetuned. But we could remove them to save space

        # Get embeddings to use as input for head
        embeddings = outs[embedding_name]

        # Define head
        head = head_fn()

        if sequence_mask is None:
            # I should not get the last (embedding) dimension,
            # because the mask should have the dimensions of the input sequence
            sequence_mask = jnp.ones_like(embeddings[:, :, 0])

        head_outs = head(  # type: ignore[call-arg]
            x=embeddings, sequence_mask=sequence_mask
        )
        outs.update(head_outs)
        return outs

    return enformer_fn
