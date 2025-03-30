from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from torch.nn import init
from transformers.configuration_utils import PretrainedConfig
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.utils import logging
import math

logger = logging.get_logger(__name__)

class AttentionConfig():
    def __init__(self,
                 norm_eps=1e-5,
                 is_norm_bias=False,
                 num_head=12,
                 hidden_dim=768,
                 attention_type='sdpa',
                 is_linear_bias=False,
                 use_global_attention_per_num_layer=3,
                 global_rope_theta=160000.0,
                 local_rope_theta=10000.0,
                 dropout_probability=0.0):
        self.norm_eps = norm_eps
        self.is_norm_bias = is_norm_bias

        self.num_head = num_head
        self.hidden_dim = hidden_dim
        self.attention_type = attention_type
        self.is_linear_bias = is_linear_bias
        self.use_global_attention_per_num_layer = use_global_attention_per_num_layer
        self.dropout_probability = dropout_probability
        self.global_rope_theta = global_rope_theta
        self.local_rope_theta = local_rope_theta


class ModernBertAttention(nn.Module):
    def __init__(self, config, layer_index):
        super().__init__()
        self.config = config

        self.hidden_dim = config.hidden_dim
        self.num_head = config.num_head
        assert self.hidden_dim % self.num_head == 0
        self.head_dim = self.hidden_dim // self.num_head

        self.attention_type = config.attention_type
        self.is_linear_bias = config.is_linear_bias
        self.dropout_probability = config.dropout_probability

        self.layer_index = layer_index
        self.use_global_attention_per_num_layer = config.use_global_attention_per_num_layer
        if self.layer_index % self.use_global_attention_per_num_layer != 0:
            self.is_local_attention = False
            self.rope_theta = config.global_rope_theta
        else:
            self.is_local_attention = True
            self.rope_theta = config.local_rope_theta

        self.q_linear = nn.Linear(in_features=self.hidden_dim,
                                  out_features=self.hidden_dim,
                                  bias=self.is_linear_bias)
        self.k_linear = nn.Linear(in_features=self.hidden_dim,
                                  out_features=self.hidden_dim,
                                  bias=self.is_linear_bias)
        self.v_linear = nn.Linear(in_features=self.hidden_dim,
                                  out_features=self.hidden_dim,
                                  bias=self.is_linear_bias)
        self.out_linear = nn.Linear(in_features=self.hidden_dim,
                                    out_features=self.hidden_dim,
                                    bias=self.is_linear_bias)
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