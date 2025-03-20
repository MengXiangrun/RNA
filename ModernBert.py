import copy
import math
import warnings
from typing import Optional, Union, List
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers import BertConfig
import inspect
from typing import Tuple, cast
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

#######################################################################################################################
IS_FLASH_ATTENTION = False

# Flash Attention 2, https://github.com/Dao-AILab/flash-attention
IMPL_USE_FLASH2 = False
try:
    if IS_FLASH_ATTENTION:
        from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_qkvpacked_func  # type: ignore

        installed_version = importlib.metadata.version("flash_attn")  # type: ignore
        if installed_version < "2.5.7": raise ImportError("newer flash_attn required >= 2.5.7")
        IMPL_USE_FLASH2 = True
except ImportError:
    pass

IMPL_USE_FLASH3 = False
try:
    if IS_FLASH_ATTENTION:
        from flash_attn_interface import flash_attn_varlen_func

        IMPL_USE_FLASH3 = True
except ImportError:
    pass


#######################################################################################################################
class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, indices: torch.Tensor, first_axis_dim) -> torch.Tensor:
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype)
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        (indices,) = ctx.saved_tensors
        grad_values = grad_output[indices]
        return grad_values, None, None


class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Get just the values of `input` which are at `indices`.

        Arguments:
            ctx: the autograd context object
            input: (b, ...) 2+ dimensional tensor
            indices: (num_idx) 1D tensor
        """
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]  # type: ignore
        second_dim = other_shape.numel()  # product of sizes of all but first dimension
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        return torch.gather(
            rearrange(input, "b ... -> b (...)"),  # (b, ...) -> (b, second_dim)
            0,
            repeat(indices, "z -> z d", d=second_dim),  # (indices,) -> (indices, second_dim)
        ).reshape(-1, *other_shape)  # (num_idx, ...)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]], device=grad_output.device, dtype=grad_output.dtype
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply
index_put_first_axis = IndexPutFirstAxis.apply


def unpad_input(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Remove padding from input sequences.

    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.

    Returns:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz)
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int ()
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    hidden_states = cast(torch.Tensor, index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices))
    return hidden_states, indices, cu_seqlens, max_seqlen_in_batch


def unpad_input_only(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Like unpad_input, but only return the unpadded first tensor.

    Save a small amount of overhead.

    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.

    Returns:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
    """
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    rearranged = rearrange(hidden_states, "b s ... -> (b s) ...")
    return index_first_axis(rearranged, indices)  # type: ignore


def pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    """Add padding to sequences.

    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz)
        batch: int batch_size
        seqlen: int max sequence length

    Returns:
        hidden_states: (batch, seqlen, ...)
    """
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)  # type: ignore


#######################################################################################################################
class FlexBertConfig(BertConfig):
    def __init__(
            self,
            attention_layer: str = "base",
            attention_probs_dropout_prob: float = 0.0,
            attn_out_bias: bool = False,
            attn_out_dropout_prob: float = 0.0,
            attn_qkv_bias: bool = False,
            bert_layer: str = "prenorm",
            decoder_bias: bool = True,
            embed_dropout_prob: float = 0.0,
            embed_norm: bool = True,
            final_norm: bool = False,
            embedding_layer: str = "absolute_pos",
            encoder_layer: str = "base",
            loss_function: str = "cross_entropy",
            loss_kwargs: dict = {},
            mlp_dropout_prob: float = 0.0,
            mlp_in_bias: bool = False,
            mlp_layer: str = "mlp",
            mlp_out_bias: bool = False,
            norm_kwargs: dict = {},
            normalization: str = "rmsnorm",
            padding: str = "unpadded",
            head_class_act: str = "silu",
            head_class_bias: bool = False,
            head_class_dropout: float = 0.0,
            head_class_norm: str = False,
            head_pred_act: str = "silu",
            head_pred_bias: bool = False,
            head_pred_dropout: float = 0.0,
            head_pred_norm: bool = True,
            pooling_type: str = "cls",
            rotary_emb_dim: int | None = None,
            rotary_emb_base: float = 10000.0,
            rotary_emb_scale_base=None,
            rotary_emb_interleaved: bool = False,
            use_fa2: bool = True,
            use_sdpa_attn_mask: bool = False,
            allow_embedding_resizing: bool = False,
            init_method: str = "default",
            init_std: float = 0.02,
            init_cutoff_factor: float = 2.0,
            init_small_embedding: bool = False,
            initial_attention_layer: str | None = None,
            initial_bert_layer: str | None = None,
            initial_mlp_layer: str | None = None,
            num_initial_layers: int = 1,
            skip_first_prenorm: bool = False,
            deterministic_fa2: bool = False,
            sliding_window: int = -1,
            global_attn_every_n_layers: int = -1,
            local_attn_rotary_emb_base: float = -1,
            local_attn_rotary_emb_dim: int | None = None,
            unpad_embeddings: bool = False,
            pad_logits: bool = False,
            compile_model: bool = False,
            masked_prediction: bool = False,
            **kwargs,
    ):
        """
        Args:
            attention_layer (str): Attention layer type.
            attention_probs_dropout_prob (float): Dropout probability for attention probabilities.
            attn_out_bias (bool): use bias in attention output projection.
            attn_out_dropout_prob (float): Dropout probability for attention output.
            attn_qkv_bias (bool): use bias for query, key, value linear layer(s).
            bert_layer (str): BERT layer type.
            decoder_bias (bool): use bias in decoder linear layer.
            embed_dropout_prob (float): Dropout probability for embeddings.
            embed_norm (bool): Normalize embedding output.
            final_norm (bool): Add normalization after the final encoder layer and before head.
            embedding_layer (str): Embedding layer type.
            encoder_layer (str): Encoder layer type.
            loss_function (str): Loss function to use.
            loss_kwargs (dict): Keyword arguments for loss function.
            mlp_dropout_prob (float): Dropout probability for MLP layers.
            mlp_in_bias (bool): Use bias in MLP input linear layer.
            mlp_layer (str): MLP layer type.
            mlp_out_bias (bool): Use bias in MLP output linear layer.
            norm_kwargs (dict): Keyword arguments for normalization layers.
            normalization (str): Normalization type.
            padding (str): Unpad inputs. Best with `use_fa2=True`.
            head_class_act (str): Activation function for classification head.
            head_class_bias (bool): Use bias in classification head linear layer(s).
            head_class_dropout (float): Dropout probability for classification head.
            head_class_norm (str): Normalization type for classification head.
            head_pred_act (str): Activation function for prediction head.
            head_pred_bias (bool): Use bias in prediction head linear layer(s).
            head_pred_dropout (float): Dropout probability for prediction head.
            head_pred_norm (bool): Normalize prediction head output.
            pooling_type (str): Pooling type.
            rotary_emb_dim (int | None): Rotary embedding dimension.
            rotary_emb_base (float): Rotary embedding base.
            rotary_emb_scale_base (float): Rotary embedding scale base.
            rotary_emb_interleaved (bool): Use interleaved rotary embeddings.
            use_fa2 (bool): Use FlashAttention2. Requires flash_attn package.
            use_sdpa_attn_mask (bool): Pass a mask to SDPA. This will prevent SDPA from using the PyTorch FA2 kernel.
            allow_embedding_resizing (bool): Embeddings will be automatically resized when they are smaller than the tokenizer vocab size.
            init_method (str): Model layers initialization method.
            init_std (float): Standard deviation for initialization. Used for normal and full_megatron init.
            init_cutoff_factor (float): Cutoff factor for initialization. Used for normal and full_megatron init.
            init_small_embedding (bool): Initialize embeddings with RWKV small init.
            initial_attention_layer (str | None): Replace first `num_initial_layers` attention_layer instance with this layer.
            initial_bert_layer (str | None): Replace first `num_initial_layers` bert_layer instance with this layer.
            initial_mlp_layer (str | None): Replace first `num_initial_layers` mlp_layer instance with this layer.
            num_initial_layers (int): Number of initial layers to set via `initial_attention_layer`, `initial_bert_layer`, and `initial_mlp_layer`.
            skip_first_prenorm (bool): Skip pre-normalization for the first bert layer. Requires `embed_norm=True`.
            deterministic_fa2 (bool): Use Flash Attention 2 deterministic mode. This is slower then the default non-deterministic mode.
            sliding_window (int): Use sliding window attention with window size `n`. -1 to disable. Window size split between the left and right context. Only supports FA2.
            global_attn_every_n_layers (int): Use global attention every `n` layers and sliding window for the rest. -1 to disable.
            local_attn_rotary_emb_base (float): Rotary embedding base for local attention. -1 to disable and use `rotary_emb_base` for all layers.
            local_attn_rotary_emb_dim (int | None): Rotary embedding dimension for local attention. None to disable and use `rotary_emb_dim` for all layers.
            unpad_embeddings (bool): Unpad inputs before the embedding layer.
            pad_logits (bool): Pad logits after the calculating the loss.
            compile_model (bool): Compile the subset of the model which can be compiled.
            masked_prediction (bool): Use only pass the masked tokens throught the final MLM layers
            **kwargs: Additional keyword arguments.
        """
        super().__init__(attention_probs_dropout_prob=attention_probs_dropout_prob, **kwargs)
        self.attention_layer = attention_layer
        self.attn_out_bias = attn_out_bias
        self.attn_out_dropout_prob = attn_out_dropout_prob
        self.attn_qkv_bias = attn_qkv_bias
        self.bert_layer = bert_layer
        self.decoder_bias = decoder_bias
        self.embed_dropout_prob = embed_dropout_prob
        self.embed_norm = embed_norm
        self.final_norm = final_norm
        self.embedding_layer = embedding_layer
        self.encoder_layer = encoder_layer
        self.loss_function = loss_function
        self.loss_kwargs = loss_kwargs
        self.mlp_dropout_prob = mlp_dropout_prob
        self.mlp_in_bias = mlp_in_bias
        self.mlp_layer = mlp_layer
        self.mlp_out_bias = mlp_out_bias
        self.norm_kwargs = norm_kwargs
        self.normalization = normalization
        self.padding = padding
        self.head_class_act = head_class_act
        self.head_class_bias = head_class_bias
        self.head_class_dropout = head_class_dropout
        self.head_class_norm = head_class_norm
        self.head_pred_act = head_pred_act
        self.head_pred_bias = head_pred_bias
        self.head_pred_dropout = head_pred_dropout
        self.head_pred_norm = head_pred_norm
        self.pooling_type = pooling_type
        self.rotary_emb_dim = rotary_emb_dim
        self.rotary_emb_base = rotary_emb_base
        self.rotary_emb_scale_base = rotary_emb_scale_base
        self.rotary_emb_interleaved = rotary_emb_interleaved
        self.use_fa2 = use_fa2
        self.use_sdpa_attn_mask = use_sdpa_attn_mask
        self.allow_embedding_resizing = allow_embedding_resizing
        self.init_method = init_method
        self.init_std = init_std
        self.init_cutoff_factor = init_cutoff_factor
        self.init_small_embedding = init_small_embedding
        self.initial_attention_layer = initial_attention_layer
        self.initial_bert_layer = initial_bert_layer
        self.initial_mlp_layer = initial_mlp_layer
        self.num_initial_layers = num_initial_layers
        self.skip_first_prenorm = skip_first_prenorm
        self.deterministic_fa2 = deterministic_fa2
        self.sliding_window = sliding_window
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.local_attn_rotary_emb_base = local_attn_rotary_emb_base
        self.local_attn_rotary_emb_dim = local_attn_rotary_emb_dim
        self.unpad_embeddings = unpad_embeddings
        self.pad_logits = pad_logits
        self.compile_model = compile_model
        self.masked_prediction = masked_prediction

        if loss_kwargs.get("return_z_loss", False):
            if loss_function != "fa_cross_entropy":
                raise ValueError("loss_function must be 'fa_cross_entropy' when return_z_loss is True")
            if loss_kwargs.get("lse_square_scale", 0) <= 0:
                raise ValueError(
                    "lse_square_scale must be passed to `loss_kwargs` and must be greater than 0 for z_loss"
                )
        if loss_kwargs.get("inplace_backward", False):
            self.loss_kwargs["inplace_backward"] = False
            warnings.warn("`inplace_backward=True` will cause incorrect metrics. Automatically setting to False.")

        if global_attn_every_n_layers > 0 and (self.num_hidden_layers - 1) % global_attn_every_n_layers != 0:
            raise ValueError(
                f"{global_attn_every_n_layers=} must be a divisor of one less than {self.num_hidden_layers=}"
            )

        if self.sliding_window != -1:
            if not self.use_fa2:
                raise ValueError("Sliding window attention is only supported with FlashAttention2")
            if self.sliding_window % 2 != 0 and self.sliding_window % 64 != 0:
                raise ValueError(
                    f"Sliding window must be an even number and divisible by 64: {self.sliding_window=} {self.sliding_window % 64} {self.sliding_window % 2}"
                )
        else:
            if self.global_attn_every_n_layers != -1:
                raise ValueError("global_attn_every_n_layers must be -1 when sliding_window is disabled")
            if self.local_attn_rotary_emb_base != -1:
                raise ValueError("local_attn_rotary_emb_base must be -1 when sliding_window is disabled")
            if self.local_attn_rotary_emb_dim is not None:
                raise ValueError("local_attn_rotary_emb_dim must be None when sliding_window is disabled")

        if self.unpad_embeddings and self.padding != "unpadded":
            warnings.warn(
                "`unpad_embeddings=True` requires `padding='unpadded'`. Automatically setting `padding='unpadded'`."
            )
            self.padding = "unpadded"
        if self.pad_logits and not self.unpad_embeddings:
            raise ValueError("`pad_logits=True` requires `unpad_embeddings=True`")
        if self.unpad_embeddings and self.embedding_layer == "absolute_pos":
            raise ValueError(f"{self.unpad_embeddings=} is incompatible with {self.embedding_layer=}")


#######################################################################################################################
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        output = x.type_as(x)
        return output * self.weight

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)


NORM2CLS = {"layernorm": nn.LayerNorm,
            "rmsnorm": RMSNorm, }


# "triton_layernorm": TritonLayerNorm if TritonLayerNorm is not None else nn.LayerNorm,
# "triton_rmsnorm": TritonRMSNorm if TritonRMSNorm is not None else RMSNorm,


def get_norm_layer(config: FlexBertConfig, compiled_norm: bool = False) -> nn.Module:
    try:
        if compiled_norm:
            # Use non-Triton norms when compiling
            if config.normalization.startswith("triton_"):
                norm = config.normalization.replace("triton_", "")
            else:
                norm = config.normalization
        else:
            norm = config.normalization
        signature = inspect.signature(NORM2CLS[norm])
        if hasattr(config, "norm_kwargs"):
            norm_kwargs = {k: v for k, v in config.norm_kwargs.items() if k in signature.parameters}
        else:
            norm_kwargs = {}
        return NORM2CLS[norm](config.hidden_size, **norm_kwargs)
    except KeyError:
        raise ValueError(f"Invalid normalization layer type: {config.normalization}, must be one of {NORM2CLS.keys()}.")


#######################################################################################################################
from transformers.activations import ACT2FN


def get_act_fn(config: Union[FlexBertConfig, str]) -> nn.Module:
    try:
        if isinstance(config, str):
            return ACT2FN[config]
        return ACT2FN[config.hidden_act]
    except KeyError:
        if isinstance(config, str):
            raise ValueError(f"Invalid {config}, must be one of {ACT2FN.keys()}.")
        else:
            raise ValueError(f"Invalid {config.hidden_act}, must be one of {ACT2FN.keys()}.")


#######################################################################################################################
class BertAlibiEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # ALiBi doesn't use position embeddings
        if getattr(config, "token_type_embeddings", True):
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
            self.use_token_type_embeddings = True
        else:
            self.use_token_type_embeddings = False

        self.LayerNorm = get_norm_layer(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.use_token_type_embeddings:
            token_type_ids = torch.zeros(config.max_position_embeddings, dtype=torch.long)
            self.register_buffer("token_type_ids", token_type_ids, persistent=False)

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                past_key_values_length: int = 0,
                ) -> torch.Tensor:

        if (input_ids is not None) == (inputs_embeds is not None):
            raise ValueError("Must specify either input_ids or input_embeds!")

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            assert inputs_embeds is not None  # just for type checking
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            # great! ALiBi
            pass

        # Setting the token_type_ids to the registered buffer in constructor
        # where it is all zeros, which usually occurs when it's auto-generated;
        # registered buffer helps users when tracing the model without passing
        # token_type_ids, solves issue #5664
        if self.use_token_type_embeddings and token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.use_token_type_embeddings:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = inputs_embeds + token_type_embeddings
        else:
            embeddings = inputs_embeds

        # no position embeddings! ALiBi
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertAlibiUnpadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention"
                f"heads ({config.num_attention_heads})")

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.p_dropout = config.attention_probs_dropout_prob
        self.Wqkv = nn.Linear(self.all_head_size, 3 * config.hidden_size)
        self.deterministic_fa2 = getattr(config, "deterministic_fa2", False)

        # Warn if defaulting to pytorch because of import issues
        if not IMPL_USE_FLASH2:
            warnings.warn("Unable to import flash_attn")
            warnings.warn("defaulting MosaicBERT attention implementation to vanilla PyTorch")
            warnings.warn("this will reduce throughput when using this model")

    def forward(self,
                hidden_states: torch.Tensor,
                cu_seqlens: torch.Tensor,
                max_seqlen: int,
                indices: torch.Tensor,
                attn_mask: torch.Tensor,
                bias: torch.Tensor,
                slopes: torch.Tensor,
                ) -> torch.Tensor:
        """Perform self-attention.

        There are two attention implementations: vanilla attention with ALiBi, and Flash Attention 2 with ALiBi

        The arguments are unpadded. The vanilla implementation of attention requires padded arguments while the
        Flash Attention implementation does not. If using vanilla we first call `pad_input`. Once we compute
        attention, we re-unpad our outputs for the other layers. The pad/unpad operations add overhead, but not
        sending pad tokens through ffs saves compute.

        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            max_seqlen: int
            indices: (total_nnz,)
            attn_mask: (batch, max_seqlen)
            bias: (batch, heads, max_seqlen, max_seqlen)
            slopes: (heads) or (batch, heads)

        Returns:
            attention: (total_nnz, dim)
        """
        bs, dim = hidden_states.shape
        qkv = self.Wqkv(hidden_states)

        # Option 1: Flash Attention with ALiBi
        if IMPL_USE_FLASH2:
            qkv = qkv.view(-1, 3, self.num_attention_heads, self.attention_head_size)
            assert 1 <= len(slopes.shape) <= 2, f"{slopes=}"
            assert slopes.shape[-1] == self.num_attention_heads, f"{slopes=}"

            convert_dtype = qkv.dtype not in (torch.float16, torch.bfloat16)
            if convert_dtype:
                # FA2 implementation only supports fp16 and bf16
                # If FA2 is supported, bfloat16 must be supported
                # as of FA2 2.4.2. (Turing GPUs not supported)
                orig_dtype = qkv.dtype
                qkv = qkv.to(torch.bfloat16)

                attention = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    dropout_p=self.p_dropout,
                    deterministic=self.deterministic_fa2,
                    alibi_slopes=slopes,
                )
                attention = attention.to(orig_dtype)  # type: ignore
            else:
                attention = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    dropout_p=self.p_dropout,
                    deterministic=self.deterministic_fa2,
                    alibi_slopes=slopes,
                )

        # Option 2:
        else:
            # batch, max_seqlen, thd
            qkv = pad_input(qkv, indices, cu_seqlens.shape[0] - 1, max_seqlen)
            unpad_bs, *_ = qkv.shape
            qkv = qkv.view(unpad_bs, -1, 3, self.num_attention_heads, self.attention_head_size)
            # if we have nonzero attention dropout (e.g. during fine-tuning) or no Triton, compute attention in PyTorch
            q = qkv[:, :, 0, :, :].permute(0, 2, 1, 3)  # b h s d
            k = qkv[:, :, 1, :, :].permute(0, 2, 3, 1)  # b h d s
            v = qkv[:, :, 2, :, :].permute(0, 2, 1, 3)  # b h s d
            attention_scores = torch.matmul(q, k) / math.sqrt(self.attention_head_size)
            attention_scores = attention_scores + bias
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)
            attention = torch.matmul(attention_probs, v).permute(0, 2, 1, 3)  # b s h d

            attention = unpad_input_only(attention, torch.squeeze(attn_mask) == 1)

        return attention.view(bs, dim)


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = get_norm_layer(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAlibiUnpadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertAlibiUnpadSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
            self,
            input_tensor: torch.Tensor,
            cu_seqlens: torch.Tensor,
            max_s: int,
            subset_idx: Optional[torch.Tensor] = None,
            indices: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            bias: Optional[torch.Tensor] = None,
            slopes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for scaled self-attention without padding.

        Arguments:
            input_tensor: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            max_s: int
            subset_idx: () set of indices whose values we care about at the end of the layer
                        (e.g., the masked tokens, if this is the final layer).
            indices: None or (total_nnz,)
            attn_mask: None or (batch, max_seqlen)
            bias: None or (batch, heads, max_seqlen, max_seqlen)
            slopes: None or (batch, heads) or (heads,)
        """
        assert (bias is None) == (slopes is None), f"{bias=}, {slopes=}"
        self_output = self.self(input_tensor, cu_seqlens, max_s, indices, attn_mask, bias, slopes)
        if subset_idx is not None:
            return self.output(
                index_first_axis(self_output, subset_idx),
                index_first_axis(input_tensor, subset_idx),
            )
        else:
            return self.output(self_output, input_tensor)


class BertResidualGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gated_layers = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.act = get_act_fn(config.hidden_act)
        self.wo = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layernorm = get_norm_layer(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual_connection = hidden_states
        # compute the activation
        hidden_states = self.gated_layers(hidden_states)
        gated = hidden_states[:, : self.config.intermediate_size]
        non_gated = hidden_states[:, self.config.intermediate_size:]
        hidden_states = self.act(gated) * non_gated
        hidden_states = self.dropout(hidden_states)
        # multiply by the second matrix
        hidden_states = self.wo(hidden_states)
        # add the residual connection and post-LN
        hidden_states = self.layernorm(hidden_states + residual_connection)
        return hidden_states


class BertAlibiLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAlibiUnpadAttention(config)
        self.mlp = BertResidualGLU(config)

    def forward(self,
                hidden_states: torch.Tensor,
                cu_seqlens: torch.Tensor,
                seqlen: int,
                subset_idx: Optional[torch.Tensor] = None,
                indices: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                bias: Optional[torch.Tensor] = None,
                slopes: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """Forward pass for a BERT layer, including both attention and MLP.
        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            seqlen: int
            subset_idx: () set of indices whose values we care about at the end of the layer
                        (e.g., the masked tokens, if this is the final layer).
            indices: None or (total_nnz,)
            attn_mask: None or (batch, max_seqlen_in_batch)
            bias: None or (batch, heads, max_seqlen_in_batch, max_seqlen_in_batch)
            slopes: None or (batch, heads) or (heads,)
        """
        assert (bias is None) == (slopes is None), f"{bias=}, {slopes=}"
        attention_output = self.attention(
            hidden_states, cu_seqlens, seqlen, subset_idx, indices, attn_mask, bias, slopes
        )
        layer_output = self.mlp(attention_output)
        return layer_output


class BertAlibiEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = BertAlibiLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

        self.num_attention_heads = config.num_attention_heads

        # The alibi mask will be dynamically expanded if it is too small for
        # the input the model receives. But it generally helps to initialize it
        # to a reasonably large size to help pre-allocate CUDA memory.
        # The default `alibi_starting_size` is 512.
        self._current_alibi_size = int(config.alibi_starting_size)
        self.alibi = torch.zeros((1, self.num_attention_heads, self._current_alibi_size, self._current_alibi_size))
        self.rebuild_alibi_tensor(size=config.alibi_starting_size)

    def rebuild_alibi_tensor(self, size: int, device: Optional[Union[torch.device, str]] = None):
        # Alibi
        # Following https://github.com/ofirpress/attention_with_linear_biases/issues/5 (Implementation 1)
        # In the causal case, you can exploit the fact that softmax is invariant to a uniform translation
        # of the logits, which makes the math work out *after* applying causal masking. If no causal masking
        # will be applied, it is necessary to construct the diagonal mask.
        n_heads = self.num_attention_heads

        def _get_alibi_head_slopes(n_heads: int) -> List[float]:
            def get_slopes_power_of_2(n_heads: int) -> List[float]:
                start = 2 ** (-(2 ** -(math.log2(n_heads) - 3)))
                ratio = start
                return [start * ratio ** i for i in range(n_heads)]

            # In the paper, they only train models that have 2^a heads for some a. This function
            # has some good properties that only occur when the input is a power of 2. To
            # maintain that even when the number of heads is not a power of 2, we use a
            # workaround.
            if math.log2(n_heads).is_integer():
                return get_slopes_power_of_2(n_heads)

            closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
            slopes_a = get_slopes_power_of_2(closest_power_of_2)
            slopes_b = _get_alibi_head_slopes(2 * closest_power_of_2)
            slopes_b = slopes_b[0::2][: n_heads - closest_power_of_2]
            return slopes_a + slopes_b

        context_position = torch.arange(size, device=device)[:, None]
        memory_position = torch.arange(size, device=device)[None, :]
        relative_position = torch.abs(memory_position - context_position)
        # [n_heads, max_token_length, max_token_length]
        relative_position = relative_position.unsqueeze(0).expand(n_heads, -1, -1)
        slopes = torch.Tensor(_get_alibi_head_slopes(n_heads)).to(device)
        self.slopes = slopes
        alibi = slopes.unsqueeze(1).unsqueeze(1) * -relative_position
        # [1, n_heads, max_token_length, max_token_length]
        alibi = alibi.unsqueeze(0)
        assert alibi.shape == torch.Size([1, n_heads, size, size])

        self._current_alibi_size = size
        self.alibi = alibi

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            output_all_encoded_layers: Optional[bool] = True,
            subset_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        attention_mask_bool = attention_mask.bool()
        batch, seqlen = hidden_states.shape[:2]
        # Unpad inputs and mask. It will remove tokens that are padded.
        # Assume ntokens is total number of tokens (padded and non-padded)
        # and ntokens_unpad is total number of non-padded tokens.
        # Then unpadding performs the following compression of the inputs:
        # hidden_states[ntokens,hidden] -> hidden_states[ntokens_unpad,hidden]
        hidden_states, indices, cu_seqlens, _ = unpad_input(hidden_states, attention_mask_bool)

        # Add alibi matrix to extended_attention_mask
        if self._current_alibi_size < seqlen:
            # Rebuild the alibi tensor when needed
            warnings.warn(f"Increasing alibi size from {self._current_alibi_size} to {seqlen}")
            self.rebuild_alibi_tensor(size=seqlen, device=hidden_states.device)
        elif self.alibi.device != hidden_states.device:
            # Device catch-up
            self.alibi = self.alibi.to(hidden_states.device)
            self.slopes = self.slopes.to(hidden_states.device)  # type: ignore
        alibi_bias = self.alibi[:, :, :seqlen, :seqlen]
        attn_bias = extended_attention_mask[:, :, :seqlen, :seqlen]
        alibi_attn_mask = attn_bias + alibi_bias

        all_encoder_layers = []
        if subset_mask is None:
            for layer_module in self.layer:
                hidden_states = layer_module(
                    hidden_states,
                    cu_seqlens,
                    seqlen,
                    None,
                    indices,
                    attn_mask=attention_mask,
                    bias=alibi_attn_mask,
                    slopes=self.slopes,
                )
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
            # Pad inputs and mask. It will insert back zero-padded tokens.
            # Assume ntokens is total number of tokens (padded and non-padded)
            # and ntokens_unpad is total number of non-padded tokens.
            # Then padding performs the following de-compression:
            #     hidden_states[ntokens_unpad,hidden] -> hidden_states[ntokens,hidden]
            hidden_states = pad_input(hidden_states, indices, batch, seqlen)
        else:
            for i in range(len(self.layer) - 1):
                layer_module = self.layer[i]
                hidden_states = layer_module(
                    hidden_states,
                    cu_seqlens,
                    seqlen,
                    None,
                    indices,
                    attn_mask=attention_mask,
                    bias=alibi_attn_mask,
                    slopes=self.slopes,
                )
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
            subset_idx = torch.nonzero(subset_mask[attention_mask_bool], as_tuple=False).flatten()
            hidden_states = self.layer[-1](
                hidden_states,
                cu_seqlens,
                seqlen,
                subset_idx=subset_idx,
                indices=indices,
                attn_mask=attention_mask,
                bias=alibi_attn_mask,
                slopes=self.slopes,
            )

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor, pool: Optional[bool] = True) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0] if pool else hidden_states
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer: bool = True):
        super(BertModel, self).__init__(config)
        self.embeddings = BertAlibiEmbeddings(config)
        self.encoder = BertAlibiEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                output_all_encoded_layers: Optional[bool] = False,
                masked_tokens_mask: Optional[torch.Tensor] = None,
                **kwargs
                ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], Optional[torch.Tensor]]:

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)

        subset_mask = []
        first_col_mask = []

        if masked_tokens_mask is None:
            subset_mask = None
        else:
            first_col_mask = torch.zeros_like(masked_tokens_mask)
            first_col_mask[:, 0] = True
            subset_mask = masked_tokens_mask | first_col_mask

        encoder_outputs = self.encoder(embedding_output,
                                       attention_mask,
                                       output_all_encoded_layers=output_all_encoded_layers,
                                       subset_mask=subset_mask)

        if masked_tokens_mask is None:
            sequence_output = encoder_outputs[-1]
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        else:
            # TD [2022-03-01]: the indexing here is very tricky.
            attention_mask_bool = attention_mask.bool()
            subset_idx = subset_mask[attention_mask_bool]  # type: ignore
            sequence_output = encoder_outputs[-1][masked_tokens_mask[attention_mask_bool][subset_idx]]
            if self.pooler is not None:
                pool_input = encoder_outputs[-1][first_col_mask[attention_mask_bool][subset_idx]]
                pooled_output = self.pooler(pool_input, pool=False)
            else:
                pooled_output = None

        if not output_all_encoded_layers:
            encoder_outputs = sequence_output

        if self.pooler is not None:
            return encoder_outputs, pooled_output

        return encoder_outputs, None


BertAlibiUnpadAttention
BertAlibiUnpadSelfAttention
BertSelfOutput
BertAlibiEmbeddings
BertAlibiEncoder
BertAlibiLayer
BertResidualGLU
BertPooler
BertModel


#######################################################################################################################
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_act_fn(config.head_pred_act)
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = get_norm_layer(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1), bert_model_embedding_weights.size(0))
        self.decoder.weight = bert_model_embedding_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertForPreTraining(BertPreTrainedModel):
    # TBD: Coming in Future Commit
    pass


class BertLMHeadModel(BertPreTrainedModel):
    # TBD: Coming in Future Commit
    pass


class BertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            warnings.warn(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_composer(cls,
                      pretrained_checkpoint,
                      state_dict=None,
                      cache_dir=None,
                      from_tf=False,
                      config=None,
                      *inputs,
                      **kwargs,
                      ):
        """Load from pre-trained."""
        model = cls(config, *inputs, **kwargs)
        if from_tf:
            raise ValueError("Mosaic BERT does not support loading TensorFlow weights.")

        state_dict = torch.load(pretrained_checkpoint)
        # If the state_dict was saved after wrapping with `composer.HuggingFaceModel`, it takes on the `model` prefix
        consume_prefix_in_state_dict_if_present(state_dict, prefix="model.")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if len(missing_keys) > 0:
            logger.warning(f"Found these missing keys in the checkpoint: {', '.join(missing_keys)}")
        if len(unexpected_keys) > 0:
            logger.warning(f"Found these unexpected keys in the checkpoint: {', '.join(unexpected_keys)}")

        return model

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        # labels should be a `torch.LongTensor` of shape
        # `(batch_size, sequence_length)`. These are used for computing the
        #  masked language modeling loss.
        #
        # Indices should be in `[-100, 0, ..., config.vocab_size]` (see
        # `input_ids` docstring) Tokens with indices set to `-100` are ignored
        # (masked), the loss is only computed for the tokens with labels in `[0,
        # ..., config.vocab_size]`
        #
        # Prediction scores are only computed for masked tokens and the (bs,
        # seqlen) dimensions are flattened
        if (input_ids is not None) == (inputs_embeds is not None):
            raise ValueError("Must specify either input_ids or input_embeds!")

        if labels is None:
            masked_tokens_mask = None
        else:
            masked_tokens_mask = labels > 0

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict,
                            masked_tokens_mask=masked_tokens_mask,
                            )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        loss = None
        if labels is not None:
            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            masked_token_idx = torch.nonzero(labels.flatten() > 0, as_tuple=False).flatten()
            loss = loss_fct(prediction_scores, labels.flatten()[masked_token_idx])

            assert input_ids is not None, "Coding error; please open an issue"
            batch, seqlen = input_ids.shape[:2]
            prediction_scores = rearrange(
                index_put_first_axis(prediction_scores, masked_token_idx, batch * seqlen),
                "(b s) d -> b s d",
                b=batch,
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))],
            dim=-1,
        )
        dummy_token = torch.full(
            (effective_batch_size, 1),
            self.config.pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class BertForNextSentencePrediction(BertPreTrainedModel):
    # TBD: Push in future commit
    pass


class BertForSequenceClassification(BertPreTrainedModel):
    """Bert Model transformer with a sequence classification/regression head.

    This head is just a linear layer on top of the pooled output. Used for,
    e.g., GLUE tasks.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_composer(cls,
                      pretrained_checkpoint,
                      state_dict=None,
                      cache_dir=None,
                      from_tf=False,
                      config=None,
                      *inputs,
                      **kwargs,
                      ):
        """Load from pre-trained."""
        model = cls(config, *inputs, **kwargs)
        if from_tf:
            raise ValueError("Mosaic BERT does not support loading TensorFlow weights.")

        state_dict = torch.load(pretrained_checkpoint)
        # If the state_dict was saved after wrapping with `composer.HuggingFaceModel`, it takes on the `model` prefix
        consume_prefix_in_state_dict_if_present(state_dict, prefix="model.")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if len(missing_keys) > 0:
            logger.warning(f"Found these missing keys in the checkpoint: {', '.join(missing_keys)}")
        if len(unexpected_keys) > 0:
            logger.warning(f"Found these unexpected keys in the checkpoint: {', '.join(unexpected_keys)}")

        return model

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        # labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        # Labels for computing the sequence classification/regression loss.
        # Indices should be in `[0, ..., config.num_labels - 1]`.
        # If `config.num_labels == 1` a regression loss is computed
        # (mean-square loss). If `config.num_labels > 1` a classification loss
        # is computed (cross-entropy).

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict,
                            )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # Compute loss
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(loss=loss,
                                        logits=logits,
                                        hidden_states=None,
                                        attentions=None
                                        )


class BertForMultipleChoice(BertPreTrainedModel):
    """
    Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # In multiple choice tasks, all choices are submitted in a batch, and
        # we compute a logit for each option independently. The logits are then
        # normalized in the forward pass to get a probability distribution over
        # the choices.
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_composer(cls,
                      pretrained_checkpoint,
                      state_dict=None,
                      cache_dir=None,
                      from_tf=False,
                      config=None,
                      *inputs,
                      **kwargs,
                      ):
        """Load from pre-trained."""
        model = cls(config, *inputs, **kwargs)
        if from_tf:
            raise ValueError("Mosaic BERT does not support loading TensorFlow weights.")

        state_dict = torch.load(pretrained_checkpoint)
        # If the state_dict was saved after wrapping with `composer.HuggingFaceModel`, it takes on the `model` prefix
        consume_prefix_in_state_dict_if_present(state_dict, prefix="model.")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if len(missing_keys) > 0:
            logger.warning(f"Found these missing keys in the checkpoint: {', '.join(missing_keys)}")
        if len(unexpected_keys) > 0:
            logger.warning(f"Found these unexpected keys in the checkpoint: {', '.join(unexpected_keys)}")

        return model

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict,
                            )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(loss=loss,
                                         logits=reshaped_logits,
                                         hidden_states=None,
                                         attentions=None)


class BertForTokenClassification(BertPreTrainedModel):
    # TBD: Push in future commit
    pass


class BertForQuestionAnswering(BertPreTrainedModel):
    """Bert Model with a span classification head.

    This is used for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden states' output to compute `span start logits`
    and `span end logits`).
    """

    # TBD: Push in future commit


BertLMPredictionHead,
BertForMaskedLM,
BertForSequenceClassification,
BertForMultipleChoice,
BertOnlyMLMHead,
BertOnlyNSPHead,
BertPredictionHeadTransform
