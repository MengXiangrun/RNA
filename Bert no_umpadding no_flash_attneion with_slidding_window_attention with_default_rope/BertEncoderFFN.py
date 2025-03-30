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

class FFNConfig():
    def __init__(self,
                 norm_eps=1e-5,
                 is_norm_bias=False,
                 in_dim=768,
                 hidden_dim=1152,
                 out_dim=768,
                 is_linear_bias=False,
                 activation_type="gelu",
                 dropout_probability=0.0):
        self.norm_eps = norm_eps
        self.is_norm_bias = is_norm_bias
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.is_linear_bias = is_linear_bias
        self.activation_type = activation_type
        self.dropout_probability = dropout_probability


class ModernBertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_dim = config.in_dim
        self.hidden_dim = config.hidden_dim
        self.out_dim = config.out_dim
        self.is_linear_bias = config.is_linear_bias
        self.activation_type = config.activation_type
        self.dropout_probability = config.dropout_probability

        self.in_linear = nn.Linear(in_features=self.in_dim,
                                   out_features=self.hidden_dim * 2,
                                   bias=self.is_linear_bias)
        self.activation = ACT2FN[self.activation_type]
        self.dropout = nn.Dropout(p=self.dropout_probability)
        self.out_linear = nn.Linear(in_features=self.hidden_dim,
                                    out_features=self.out_dim,
                                    bias=self.is_linear_bias)

    def forward(self, hidden_embed: torch.Tensor) -> torch.Tensor:
        hidden_embed = self.in_linear(hidden_embed)

        ungate_hidden_embed, gate_hidden_embed = hidden_embed.chunk(2, dim=-1)
        ungate_hidden_embed = self.activation(ungate_hidden_embed)

        hidden_embed = ungate_hidden_embed * gate_hidden_embed
        hidden_embed = self.dropout(hidden_embed)
        hidden_embed = self.out_linear(hidden_embed)

        return hidden_embed