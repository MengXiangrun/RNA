from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.utils import is_flash_attn_2_available, logging
from transformers.models.modernbert.configuration_modernbert import ModernBertConfig
import math

logger = logging.get_logger(__name__)


# class ModernBertRotaryEmbedding0(nn.Module):
#     def __init__(self, config: ModernBertConfig, dim: int, base: float, device: Optional[torch.device] = None):
#         super().__init__()
#         # BC: "rope_type" was originally "type"
#         if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
#             self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
#         else:
#             self.rope_type = "default"
#         self.max_seq_len_cached = config.max_position_embeddings
#         self.original_max_seq_len = config.max_position_embeddings
#
#         self.config = config
#         self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
#         inv_freq, self.attention_scaling = self.rope_init_fn(None, device, dim=dim, base=base)
#         1
#         self.inv_freq = inv_freq
#         2
#         self.register_buffer("inv_freq", inv_freq, persistent=False)
#         self.original_inv_freq = self.inv_freq
#
#     def _dynamic_frequency_update(self, position_ids, device):
#         """
#         dynamic RoPE layers should recompute `inv_freq` in the following situations:
#         1 - growing beyond the cached sequence length (allow scaling)
#         2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
#         """
#         seq_len = torch.max(position_ids) + 1
#         if seq_len > self.max_seq_len_cached:  # growth
#             inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
#             self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
#             self.max_seq_len_cached = seq_len
#
#         if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
#             # This .to() is needed if the model has been moved to a device after being initialized (because
#             # the buffer is automatically moved, but not the original copy)
#             self.original_inv_freq = self.original_inv_freq.to(device)
#             self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
#             self.max_seq_len_cached = self.original_max_seq_len
#
#     @torch.no_grad()
#     def forward(self, x, position_index):
#         if "dynamic" in self.rope_type:
#             self._dynamic_frequency_update(position_index, device=x.device)
#
#         # Core RoPE block
#         inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_index.shape[0], -1, 1)
#         position_ids_expanded = position_index[:, None, :].float()
#
#         # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
#         device_type = x.device.type
#         device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
#         with torch.autocast(device_type=device_type, enabled=False):
#             freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
#             emb = torch.cat((freqs, freqs), dim=-1)
#             cos = emb.cos()
#             sin = emb.sin()
#
#         # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
#         cos = cos * self.attention_scaling
#         sin = sin * self.attention_scaling
#
#         return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@torch.no_grad()
def rope(dim: int,
         base: float,
         device: torch.device,
         dtype: torch.dtype,
         position_index: torch.Tensor,
         attention_scaling=1.0):
    # Compute the inverse frequencies
    dim_index = torch.arange(start=0, end=dim, step=2, dtype=torch.int64)
    dim_index = dim_index.float().to(device)
    denominator = dim_index / dim
    inverse_frequency = 1.0 / (base ** denominator)

    # (head dim/2,)
    # (1, head_dim/2, 1)
    inverse_frequency = inverse_frequency.float()
    inverse_frequency = inverse_frequency[None, :, None]
    num_sequence = position_index.shape[0]  # position_index.shape: (1, sequence_length)

    # (1, head_dim/2, 1)
    inverse_frequency = inverse_frequency.expand(num_sequence, -1, 1)
    # (1, sequence_length, 1)
    position_index = position_index[:, None, :].float()

    # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
    device_type = device.type
    if isinstance(device_type, str) and device_type != "mps":
        device_type = device_type
    else:
        device_type = "cpu"

    with torch.autocast(device_type=device_type, enabled=False):
        inverse_frequency = inverse_frequency.float()
        position_index = position_index.float()
        # (1, sequence_length, head_dim/2)
        frequency = inverse_frequency @ position_index
        frequency = frequency.transpose(1, 2)
        # (1, sequence_length, head_dim)
        emb = torch.cat([frequency, frequency], dim=-1)
        cos = emb.cos()
        sin = emb.sin()

    # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
    # (sequence_length, head_dim)
    cos = cos * attention_scaling
    sin = sin * attention_scaling
    cos = cos.to(dtype=dtype)
    sin = sin.to(dtype=dtype)

    return cos, sin


class ModernBertRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float):
        super().__init__()
        self.base = base
        self.dim = dim
        self.inverse_frequency = None
        self.attention_scaling = None
        self.device = None

    def compute_default_rope_parameters(self):
        # Unused in this type of RoPE
        attention_scaling = 1.0

        # Compute the inverse frequencies
        dim_index = torch.arange(start=0, end=self.dim, step=2, dtype=torch.int64)
        dim_index = dim_index.float()
        dim_index = dim_index.to(self.device)
        denominator = dim_index / self.dim
        inverse_frequency = 1.0 / (self.base ** denominator)

        return inverse_frequency, attention_scaling

    @torch.no_grad()
    def forward(self, device, dtype, position_index):
        self.device = device
        self.inverse_frequency, self.attention_scaling = self.compute_default_rope_parameters()

        # (head dim/2,)
        # (1, head_dim/2, 1)
        inverse_frequency = self.inverse_frequency.float()
        inverse_frequency = inverse_frequency[None, :, None]

        num_sequence = position_index.shape[0]  # 1这个维度为啥要单独表示? position_index.shape:(1, sequence_length)

        # (1, head_dim/2, 1)
        inverse_frequency = inverse_frequency.expand(num_sequence, -1, 1)

        # (1, sequence_length, 1)
        position_index = position_index[:, None, :].float()

        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = device.type
        if isinstance(device_type, str) and device_type != "mps":
            device_type = device_type
        else:
            device_type = "cpu"

        with torch.autocast(device_type=device_type, enabled=False):
            inverse_frequency = inverse_frequency.float()
            position_index = position_index.float()

            # (1, sequence_length, head_dim/2)
            frequency = inverse_frequency @ position_index
            frequency = frequency.transpose(1, 2)

            # (1, sequence_length, head_dim)
            emb = torch.cat([frequency, frequency], dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        # 高级RoPE类型（例如yarn）应用后处理缩放因子，相当于缩放注意力

        # (sequence_length, head_dim)
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)

        return cos, sin


# class ModernBertMLP(nn.Module):
#     def __init__(self, config: ModernBertConfig):
#         super().__init__()
#         self.config = config
#         self.in_linear = nn.Linear(config.hidden_size, int(config.intermediate_size) * 2, bias=config.mlp_bias)
#         self.activation = ACT2FN[config.hidden_activation]
#         self.dropout = nn.Dropout(config.mlp_dropout)
#         self.out_linear = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
#
#     def forward(self, hidden_embed: torch.Tensor) -> torch.Tensor:
#         hidden_embed = self.in_linear(hidden_embed)
#         ungate_hidden_embed, gate_hidden_embed = hidden_embed.chunk(2, dim=-1)
#         ungate_hidden_embed = self.activation(ungate_hidden_embed)
#         hidden_embed = ungate_hidden_embed * gate_hidden_embed
#         hidden_embed = self.dropout(hidden_embed)
#         hidden_embed = self.out_linear(hidden_embed)
#
#         return hidden_embed
#
#
# class ModernBertAttention(nn.Module):
#     def __init__(self, config: ModernBertConfig, layer_id: Optional[int] = None):
#         super().__init__()
#         self.config = config
#         self.layer_id = layer_id
#
#         self.hidden_dim = config.hidden_size
#         self.num_head = config.num_attention_heads
#         assert self.hidden_dim % self.num_head == 0
#         self.head_dim = self.hidden_dim // self.num_head
#
#         self.attention_type = config._attn_implementation
#         self.is_attention_bias = config.attention_bias
#         self.attention_dropout = config.attention_dropout
#         self.deterministic_flash_attn = config.deterministic_flash_attn
#
#         if layer_id % config.global_attn_every_n_layers != 0:
#             self.local_attention = config.local_attention
#             self.local_attention = (self.local_attention // 2, self.local_attention // 2)
#             self.rope_theta = config.global_rope_theta
#             self.max_position_embeddings = config.max_position_embeddings
#         else:
#             self.local_attention = (-1, -1)
#             self.rope_theta = config.local_rope_theta
#             self.max_position_embeddings = config.local_attention
#
#         self.q_linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.is_attention_bias)
#         self.k_linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.is_attention_bias)
#         self.v_linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.is_attention_bias)
#         self.out_linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.is_attention_bias)
#         self.out_dropout = nn.Dropout(self.attention_dropout)
#         self.pruned_heads = set()
#
#     def forward(self,
#                 hidden_embed,
#                 is_output_attentions,
#                 attention_mask,
#                 sliding_window_mask,
#                 position_index,
#                 cu_seqlens,
#                 max_seqlen) -> torch.Tensor:
#         batch_size, sequence_length, hidden_dim = hidden_embed.shape
#         q = self.q_linear(hidden_embed)
#         k = self.k_linear(hidden_embed)
#         v = self.v_linear(hidden_embed)
#         batch_size, target_sequence_length, hidden_dim = q.shape
#         batch_size, source_sequence_length, hidden_dim = k.shape
#         batch_size, source_sequence_length, hidden_dim = v.shape
#
#         if self.attention_type == "sdpa":
#             q = q.view(batch_size, target_sequence_length, self.num_head, self.head_dim)
#             k = q.view(batch_size, source_sequence_length, self.num_head, self.head_dim)
#             v = q.view(batch_size, source_sequence_length, self.num_head, self.head_dim)
#             batch_size, target_sequence_length, num_head, head_dim = q.shape
#             batch_size, source_sequence_length, num_head, head_dim = k.shape
#             batch_size, source_sequence_length, num_head, head_dim = v.shape
#             hidden_embed = self.sdpa_attention_forward(q=q, k=k, v=v,
#                                                        batch_size=batch_size,
#                                                        attention_mask=attention_mask,
#                                                        sliding_window_mask=sliding_window_mask,
#                                                        position_index=position_index)
#         if self.attention_type == "eager":
#             qkv = qkv.view(batch_size, -1, 3, self.num_head, self.head_dim)
#             batch_size, sequence_length, num_tensor, num_head, head_dim = qkv.shape
#             hidden_embed = self.eager_attention_forward(qkv=qkv,
#                                                         attention_mask=attention_mask,
#                                                         sliding_window_mask=sliding_window_mask,
#                                                         position_index=position_index,
#                                                         batch_size=batch_size,
#                                                         is_output_attentions=is_output_attentions)
#         if self.attention_type == "flash_attention_2":
#             qkv = qkv.view(-1, 3, self.num_head, self.head_dim)
#             hidden_embed = self.flash_attention_forward(qkv=qkv,
#                                                         cu_seqlens=cu_seqlens,
#                                                         max_seqlen=max_seqlen,
#                                                         batch_size=batch_size,
#                                                         target_dtype=qkv.dtype)
#         hidden_embed = self.out_linear(hidden_embed)
#         hidden_embed = self.out_dropout(hidden_embed)
#
#         # return (hidden_embed,) + attn_outputs[1:]  # add attentions if outputted
#         return hidden_embed
#
#     def sdpa_attention_forward(self,
#                                q, k, v,
#                                attention_mask: torch.Tensor,
#                                sliding_window_mask: torch.Tensor,
#                                position_index: Optional[torch.LongTensor],
#                                batch_size: int,
#                                ) -> Tuple[torch.Tensor]:
#         batch_size, target_sequence_length, num_head, head_dim = q.shape
#         batch_size, source_sequence_length, num_head, head_dim = k.shape
#         batch_size, source_sequence_length, num_head, head_dim = v.shape
#         q = q.transpose(1, 2)
#         k = k.transpose(1, 2)
#         v = v.transpose(1, 2)
#         batch_size, num_head, target_sequence_length, head_dim = q.shape
#         batch_size, num_head, source_sequence_length, head_dim = k.shape
#         batch_size, num_head, source_sequence_length, head_dim = v.shape
#
#         # rotary_emb
#         cos, sin = self.rope(dim=head_dim,
#                              base=self.rope_theta,
#                              dtype=q.dtype,
#                              device=q.device,
#                              position_index=position_index)
#
#         # query, key, value: [batch_size, heads, seq_len, head_dim]
#         q, k = self.apply_rotary_pos_emb(q=q, k=k, cos=cos, sin=sin)
#
#         if self.local_attention != (-1, -1):
#             attention_mask = sliding_window_mask
#
#         if self.training:
#             dropout_p = self.attention_dropout
#         else:
#             dropout_p = 0.0
#
#         hidden_embed = F.scaled_dot_product_attention(query=q,
#                                                       key=k,
#                                                       value=v,
#                                                       is_causal=False,
#                                                       scale=None,
#                                                       dropout_p=dropout_p,
#                                                       attn_mask=attention_mask)
#         hidden_embed = hidden_embed.transpose(1, 2)
#         hidden_embed = hidden_embed.contiguous()
#         hidden_embed = hidden_embed.view(batch_size, target_sequence_length, self.hidden_dim)
#
#         return hidden_embed
#
#     def flash_attention_forward(self,
#                                 qkv: torch.Tensor,
#                                 cu_seqlens: torch.Tensor,
#                                 max_seqlen: int,
#                                 batch_size: int,
#                                 target_dtype: torch.dtype = torch.bfloat16,
#                                 ) -> Tuple[torch.Tensor]:
#         # (total_seqlen, 3, nheads, headdim)
#         qkv = self.rotary_emb(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
#
#         convert_dtype = qkv.dtype not in (torch.float16, torch.bfloat16)
#         if convert_dtype:
#             # FA2 implementation only supports fp16 and bf16. If FA2 is supported,
#             # bfloat16 must be supported as of FA2 2.5.7. (Turing GPUs not supported)
#             orig_dtype = qkv.dtype
#             qkv = qkv.to(target_dtype)
#
#             attn = flash_attn_varlen_qkvpacked_func(qkv,
#                                                     cu_seqlens=cu_seqlens,
#                                                     max_seqlen=max_seqlen,
#                                                     dropout_p=self.attention_dropout if self.training else 0.0,
#                                                     deterministic=self.deterministic_flash_attn,
#                                                     window_size=self.local_attention,
#                                                     )
#             attn = attn.to(orig_dtype)  # type: ignore
#         else:
#             attn = flash_attn_varlen_qkvpacked_func(qkv,
#                                                     cu_seqlens=cu_seqlens,
#                                                     max_seqlen=max_seqlen,
#                                                     dropout_p=self.attention_dropout if self.training else 0.0,
#                                                     deterministic=self.deterministic_flash_attn,
#                                                     window_size=self.local_attention,
#                                                     )
#         attn = attn.view(batch_size, self.hidden_dim)
#         return attn
#
#     def eager_attention_forward(self,
#                                 qkv: torch.Tensor,
#                                 attention_mask: torch.Tensor,
#                                 sliding_window_mask: torch.Tensor,
#                                 position_index: Optional[torch.LongTensor],
#                                 batch_size: int,
#                                 is_output_attentions: Optional[bool] = False,
#                                 ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
#         # qkv: [batch_size, seqlen, 3, nheads, headdim]
#         cos, sin = self.self.rotary_emb(qkv, position_ids=position_index)
#         query, key, value = qkv.transpose(3, 1).unbind(dim=2)
#         # query, key, value: [batch_size, heads, seq_len, head_dim]
#         query, key = apply_rotary_pos_emb(query, key, cos, sin)
#
#         scale = self.head_dim ** -0.5
#         attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale
#
#         if self.local_attention != (-1, -1):
#             attention_mask = sliding_window_mask
#
#         attn_weights = attn_weights + attention_mask
#
#         # upcast attention to fp32
#         attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
#         attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
#         attn_output = torch.matmul(attn_weights, value)
#         attn_output = attn_output.transpose(1, 2).contiguous()
#         attn_output = attn_output.view(batch_size, -1, self.hidden_dim)
#         if is_output_attentions:
#             return (attn_output, attn_weights)
#         return (attn_output,)
#
#     def apply_rotary_pos_emb(self, q, k, cos, sin, position_index=None, unsqueeze_dim=1):
#         sequence_length, head_dim = cos.shape
#         sequence_length, head_dim = sin.shape
#
#         batch_size, num_head, target_sequence_length, head_dim = q.shape
#         batch_size, num_head, source_sequence_length, head_dim = k.shape
#
#         cos = cos.view(1, 1, sequence_length, head_dim)
#         sin = sin.view(1, 1, sequence_length, head_dim)
#
#         cos = cos.repeat(batch_size, num_head, 1, 1)
#         sin = sin.repeat(batch_size, num_head, 1, 1)
#
#         rotate_half_q = self.rotate_half(x=q)
#         rotate_half_k = self.rotate_half(x=k)
#
#         q_embed = (q * cos) + (rotate_half_q * sin)
#         k_embed = (k * cos) + (rotate_half_k * sin)
#
#         return q_embed, k_embed
#
#     def rotate_half(self, x):
#         """Rotates half the hidden dims of the input."""
#         mid = x.shape[-1] // 2
#         x1 = x[..., :mid]
#         x2 = x[..., mid:]
#         out = torch.cat((-x2, x1), dim=-1)
#
#         return out
#
#     @torch.no_grad()
#     def rope(self,
#              dim: int,
#              base: float,
#              device: torch.device,
#              dtype: torch.dtype,
#              position_index: torch.Tensor,
#              attention_scaling=1.0):
#         one, sequence_length = position_index.shape
#
#         # Compute the inverse frequencies
#         dim_index = torch.arange(start=0, end=dim, step=2, dtype=torch.int64)
#         dim_index = dim_index.float().to(device)
#         denominator = dim_index / dim
#         inverse_frequency = 1.0 / (base ** denominator)
#
#         # (dim/2, 1)
#         inverse_frequency = inverse_frequency.view(dim // 2, 1)
#
#         # (1, sequence_length)
#         position_index = position_index.view(1, sequence_length)
#
#         # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
#         device_type = device.type
#         if isinstance(device_type, str) and device_type != "mps":
#             device_type = device_type
#         else:
#             device_type = "cpu"
#
#         with torch.autocast(device_type=device_type, enabled=False):
#             inverse_frequency = inverse_frequency.float()
#             position_index = position_index.float()
#             frequency = inverse_frequency @ position_index
#             frequency = frequency.transpose(0, 1)
#             emb = torch.cat([frequency, frequency], dim=-1)
#             cos = emb.cos()
#             sin = emb.sin()
#
#         # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
#         # (sequence_length, head_dim)
#         sequence_length, dim = cos.shape
#         sequence_length, dim = sin.shape
#         cos = cos * attention_scaling
#         sin = sin * attention_scaling
#         cos = cos.to(dtype=dtype)
#         sin = sin.to(dtype=dtype)
#
#         return cos, sin
#
#     def scaled_dot_product_attention(self,
#                                      query,
#                                      key,
#                                      value,
#                                      attn_mask=None,
#                                      dropout_p=0.0,
#                                      is_causal=False,
#                                      scale=None) -> torch.Tensor:
#         L, S = query.size(-2), key.size(-2)
#
#         if scale is None:
#             scale_factor = 1 / math.sqrt(query.size(-1))
#         else:
#             scale_factor = scale
#
#         attn_bias = torch.zeros(L, S, dtype=query.dtype)
#
#         if is_causal:
#             assert attn_mask is None
#             temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
#             attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
#             attn_bias.to(query.dtype)
#
#         if attn_mask is not None:
#             if attn_mask.dtype == torch.bool:
#                 attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
#             else:
#                 attn_bias += attn_mask
#
#         attn_weight = query @ key.transpose(-2, -1) * scale_factor
#         attn_weight += attn_bias
#         attn_weight = torch.softmax(attn_weight, dim=-1)
#         attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
#         out = attn_weight @ value
#         return out


class ModernBertMLP(nn.Module):
    def __init__(self, config, in_dim, hidden_dim, is_bias, activation_type, dropout_probability):
        super().__init__()
        self.config = config
        self.in_linear = nn.Linear(in_dim, int(hidden_dim) * 2, bias=is_bias)
        self.activation = ACT2FN[activation_type]
        self.dropout = nn.Dropout(dropout_probability)
        self.out_linear = nn.Linear(hidden_dim, in_dim, bias=is_bias)

    def forward(self, hidden_embed: torch.Tensor) -> torch.Tensor:
        hidden_embed = self.in_linear(hidden_embed)

        ungate_hidden_embed, gate_hidden_embed = hidden_embed.chunk(2, dim=-1)
        ungate_hidden_embed = self.activation(ungate_hidden_embed)

        hidden_embed = ungate_hidden_embed * gate_hidden_embed
        hidden_embed = self.dropout(hidden_embed)
        hidden_embed = self.out_linear(hidden_embed)

        return hidden_embed


class ModernBertAttention(nn.Module):
    def __init__(self,
                 config: ModernBertConfig,
                 layer_index: Optional[int] = None,
                 hidden_dim=None,
                 num_head=None,
                 attention_type=None,
                 is_qkv_linear_bias=None,
                 dropout_probability=None,
                 use_global_attention_per_num_layer=None,
                 global_rope_theta=None,
                 local_rope_theta=None):
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        assert self.hidden_dim % self.num_head == 0
        self.head_dim = self.hidden_dim // self.num_head
        self.attention_type = attention_type
        self.is_qkv_linear_bias = is_qkv_linear_bias
        self.dropout_probability = dropout_probability

        if layer_index % use_global_attention_per_num_layer != 0:
            self.is_local_attention = False
            self.rope_theta = global_rope_theta
        else:
            self.is_local_attention = True
            self.rope_theta = local_rope_theta

        self.q_linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.is_qkv_linear_bias)
        self.k_linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.is_qkv_linear_bias)
        self.v_linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.is_qkv_linear_bias)
        self.out_linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.is_qkv_linear_bias)
        self.out_dropout = nn.Dropout(self.dropout_probability)

        self.pruned_heads = set()

    def forward(self,
                hidden_embed,
                global_attention_mask,
                local_attention_mask,
                position_index) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_embed.shape

        q = self.q_linear(hidden_embed)
        k = self.k_linear(hidden_embed)
        v = self.v_linear(hidden_embed)
        batch_size, target_sequence_length, hidden_dim = q.shape
        batch_size, source_sequence_length, hidden_dim = k.shape
        batch_size, source_sequence_length, hidden_dim = v.shape

        q = q.view(batch_size, target_sequence_length, self.num_head, self.head_dim)
        k = q.view(batch_size, source_sequence_length, self.num_head, self.head_dim)
        v = q.view(batch_size, source_sequence_length, self.num_head, self.head_dim)
        batch_size, target_sequence_length, num_head, head_dim = q.shape
        batch_size, source_sequence_length, num_head, head_dim = k.shape
        batch_size, source_sequence_length, num_head, head_dim = v.shape

        if self.attention_type == "sdpa":
            hidden_embed = self.sdpa_attention_forward(q=q, k=k, v=v,
                                                       global_attention_mask=global_attention_mask,
                                                       local_attention_mask=local_attention_mask,
                                                       position_index=position_index)
        if self.attention_type == "eager":
            hidden_embed = self.eager_attention_forward(q=q, k=k, v=v,
                                                        attention_mask=global_attention_mask,
                                                        position_index=position_index)
        hidden_embed = self.out_linear(hidden_embed)
        hidden_embed = self.out_dropout(hidden_embed)

        return hidden_embed

    def sdpa_attention_forward(self,
                               q, k, v,
                               global_attention_mask: torch.Tensor,
                               local_attention_mask: torch.Tensor,
                               position_index: Optional[torch.LongTensor],
                               ) -> Tuple[torch.Tensor]:
        batch_size, target_sequence_length, num_head, head_dim = q.shape
        batch_size, source_sequence_length, num_head, head_dim = k.shape
        batch_size, source_sequence_length, num_head, head_dim = v.shape
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        batch_size, num_head, target_sequence_length, head_dim = q.shape
        batch_size, num_head, source_sequence_length, head_dim = k.shape
        batch_size, num_head, source_sequence_length, head_dim = v.shape

        # rotary_emb
        cos, sin = self.rope(dim=head_dim,
                             base=self.rope_theta,
                             dtype=q.dtype,
                             device=q.device,
                             position_index=position_index)

        q, k = self.apply_rotary_pos_emb(q=q, k=k, cos=cos, sin=sin)

        if self.is_local_attention:
            attention_mask = local_attention_mask
        else:
            attention_mask = global_attention_mask

        if self.training:
            dropout_probability = self.dropout_probability
        else:
            dropout_probability = 0.0

        hidden_embed = F.scaled_dot_product_attention(query=q,
                                                      key=k,
                                                      value=v,
                                                      is_causal=False,
                                                      scale=None,
                                                      dropout_p=dropout_probability,
                                                      attn_mask=attention_mask)

        # to (batch_size, source_sequence_length, num_head, head_dim)
        # to (batch_size, source_sequence_length, hidden_dim)
        hidden_embed = hidden_embed.transpose(1, 2)
        hidden_embed = hidden_embed.contiguous()
        hidden_embed = hidden_embed.view(batch_size, target_sequence_length, self.hidden_dim)

        return hidden_embed

    def apply_rotary_pos_emb(self, q, k, cos, sin):
        sequence_length, head_dim = cos.shape
        sequence_length, head_dim = sin.shape

        batch_size, num_head, target_sequence_length, head_dim = q.shape
        batch_size, num_head, source_sequence_length, head_dim = k.shape

        cos = cos.view(1, 1, sequence_length, head_dim)
        sin = sin.view(1, 1, sequence_length, head_dim)

        cos = cos.repeat(batch_size, num_head, 1, 1)
        sin = sin.repeat(batch_size, num_head, 1, 1)

        rotate_half_q = self.rotate_half(x=q)
        rotate_half_k = self.rotate_half(x=k)

        q_embed = (q * cos) + (rotate_half_q * sin)
        k_embed = (k * cos) + (rotate_half_k * sin)

        return q_embed, k_embed

    def rotate_half(self, x):
        mid = x.shape[-1] // 2
        x1 = x[..., :mid]
        x2 = x[..., mid:]
        out = torch.cat((-x2, x1), dim=-1)

        return out

    @torch.no_grad()
    def rope(self,
             dim: int,
             base: float,
             device: torch.device,
             dtype: torch.dtype,
             position_index: torch.Tensor,
             attention_scaling=1.0):
        one, sequence_length = position_index.shape

        # Compute the inverse frequencies
        dim_index = torch.arange(start=0, end=dim, step=2, dtype=torch.int64)
        dim_index = dim_index.float().to(device)
        denominator = dim_index / dim
        inverse_frequency = 1.0 / (base ** denominator)

        # (dim/2, 1)
        inverse_frequency = inverse_frequency.view(dim // 2, 1)

        # (1, sequence_length)
        position_index = position_index.view(1, sequence_length)

        device_type = device.type
        if isinstance(device_type, str) and device_type != "mps":
            device_type = device_type
        else:
            device_type = "cpu"

        with torch.autocast(device_type=device_type, enabled=False):
            inverse_frequency = inverse_frequency.float()
            position_index = position_index.float()
            frequency = inverse_frequency @ position_index
            frequency = frequency.transpose(0, 1)
            emb = torch.cat([frequency, frequency], dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        # (sequence_length, dim)
        sequence_length, dim = cos.shape
        sequence_length, dim = sin.shape
        cos = cos * attention_scaling
        sin = sin * attention_scaling
        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)

        return cos, sin

    def eager_attention_forward(self,
                                q, k, v,
                                attention_mask: torch.Tensor,
                                position_index: Optional[torch.LongTensor],
                                ) -> Tuple[torch.Tensor]:
        batch_size, target_sequence_length, num_head, head_dim = q.shape
        batch_size, source_sequence_length, num_head, head_dim = k.shape
        batch_size, source_sequence_length, num_head, head_dim = v.shape
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        batch_size, num_head, target_sequence_length, head_dim = q.shape
        batch_size, num_head, source_sequence_length, head_dim = k.shape
        batch_size, num_head, source_sequence_length, head_dim = v.shape

        # rotary_emb
        cos, sin = self.rope(dim=head_dim,
                             base=self.rope_theta,
                             dtype=q.dtype,
                             device=q.device,
                             position_index=position_index)

        q, k = self.apply_rotary_pos_emb(q=q, k=k, cos=cos, sin=sin)

        if self.training:
            hidden_embed = F.scaled_dot_product_attention(query=q,
                                                          key=k,
                                                          value=v,
                                                          is_causal=False,
                                                          scale=None,
                                                          dropout_p=self.dropout_probability,
                                                          attn_mask=attention_mask)
        else:
            hidden_embed = F.scaled_dot_product_attention(query=q,
                                                          key=k,
                                                          value=v,
                                                          is_causal=False,
                                                          scale=None,
                                                          dropout_p=0.0,
                                                          attn_mask=attention_mask)

        hidden_embed = hidden_embed.transpose(1, 2)
        hidden_embed = hidden_embed.contiguous()
        hidden_embed = hidden_embed.view(batch_size, target_sequence_length, self.hidden_dim)

        return hidden_embed

    def scaled_dot_product_attention(self,
                                     q,
                                     k,
                                     v,
                                     attention_mask=None,
                                     dropout_probability=0.0,
                                     is_causal=False,
                                     scale=None) -> torch.Tensor:
        batch_size, num_head, target_sequence_length, head_dim = q.shape
        batch_size, num_head, source_sequence_length, head_dim = k.shape
        batch_size, num_head, source_sequence_length, head_dim = v.shape

        if scale is None:
            scale_factor = 1 / math.sqrt(q.size(-1))
        else:
            scale_factor = scale

        shape = (target_sequence_length, source_sequence_length)
        attention_bias = torch.zeros(shape, dtype=q.dtype)

        if is_causal:
            assert attention_mask is None
            temp_mask = torch.ones(shape, dtype=torch.bool).tril(diagonal=0)
            attention_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attention_bias.to(q.dtype)

        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attention_bias.masked_fill_(attention_mask.logical_not(), float("-inf"))
            else:
                attention_bias += attention_mask

        # attention_bias is mask
        # mask token = -inf = False
        # numask token = 0.0 = True

        attention = q @ k.transpose(-2, -1) * scale_factor
        attention += attention_bias
        attention = torch.softmax(attention, dim=-1)
        attention = torch.dropout(attention, dropout_probability, train=True)
        out = attention @ v

        return out


class ModernBertEncoderLayer(nn.Module):
    def __init__(self, config: ModernBertConfig, layer_index: Optional[int] = None):
        super().__init__()
        self.config = config

        if layer_index == 0:
            self.attention_norm = nn.Identity()
        else:
            self.attention_norm = nn.LayerNorm(normalized_shape=config.hidden_size,
                                               eps=config.norm_eps,
                                               bias=config.norm_bias)
        self.attention = ModernBertAttention(config=config,
                                             layer_index=layer_index,
                                             hidden_dim=config.hidden_size,
                                             num_head=config.num_attention_heads,
                                             attention_type=config._attn_implementation,
                                             is_qkv_linear_bias=config.attention_bias,
                                             dropout_probability=config.attention_dropout,
                                             use_global_attention_per_num_layer=config.global_attn_every_n_layers,
                                             global_rope_theta=config.global_rope_theta,
                                             local_rope_theta=config.local_rope_theta)

        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.mlp = ModernBertMLP(config=config,
                                 in_dim=config.hidden_size,
                                 hidden_dim=config.intermediate_size * 2,
                                 is_bias=config.mlp_bias,
                                 activation_type=config.hidden_activation,
                                 dropout_probability=config.mlp_dropout)

    def forward(self,
                hidden_embed: torch.Tensor,
                global_attention_mask: Optional[torch.Tensor] = None,
                local_attention_mask: Optional[torch.Tensor] = None,
                position_index: Optional[torch.LongTensor] = None) -> torch.Tensor:
        # attention
        hidden_embed = self.attention_norm.forward(input=hidden_embed)
        residual = hidden_embed.clone()
        hidden_embed = self.attention.forward(hidden_embed=hidden_embed,
                                              global_attention_mask=global_attention_mask,
                                              local_attention_mask=local_attention_mask,
                                              position_index=position_index)
        hidden_embed = hidden_embed + residual

        # ffn
        residual = hidden_embed.clone()
        hidden_embed = self.mlp_norm.forward(input=hidden_embed)
        hidden_embed = self.mlp.forward(hidden_embed=hidden_embed)
        hidden_embed = hidden_embed + residual

        return hidden_embed
