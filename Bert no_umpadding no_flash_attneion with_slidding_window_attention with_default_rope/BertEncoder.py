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
from BertEncoderFFN import ModernBertMLP
from BertEncoderAttention import ModernBertAttention

logger = logging.get_logger(__name__)


class ModernBertEncoderLayer(nn.Module):
    def __init__(self, attention_config, ffn_config, layer_index):
        super().__init__()
        self.attention_config = attention_config
        self.ffn_config = ffn_config
        self.layer_index = layer_index

        if self.layer_index == 0:
            self.attention_norm = nn.Identity()
        else:
            self.attention_norm = nn.LayerNorm(normalized_shape=self.attention_config.hidden_dim,
                                               eps=self.attention_config.norm_eps,
                                               bias=self.attention_config.is_norm_bias)
        self.attention = ModernBertAttention(config=self.attention_config,
                                             layer_index=self.layer_index)

        self.mlp_norm = nn.LayerNorm(normalized_shape=self.attention_config.hidden_dim,
                                     eps=self.ffn_config.norm_eps,
                                     bias=self.ffn_config.is_norm_bias)
        self.mlp = ModernBertMLP(config=ffn_config)

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