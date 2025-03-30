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


class EmbeddingConfig():
    def __init__(self,
                 vocabulary_size=8192,
                 hidden_dim=768,
                 pad_token_id=0,
                 norm_eps=1e-5,
                 is_norm_bias=False,
                 dropout_probability=0.0):
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        self.norm_eps = norm_eps
        self.is_norm_bias = is_norm_bias
        self.dropout_probability = dropout_probability


class ModernBertEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocabulary_size = config.vocabulary_size
        self.hidden_dim = config.hidden_dim
        self.pad_token_id = config.pad_token_id
        self.norm_eps = config.norm_eps
        self.is_norm_bias = config.is_norm_bias
        self.dropout_probability = config.dropout_probability

        self.index2embed = nn.Embedding(self.vocabulary_size, self.hidden_dim, padding_idx=self.pad_token_id)
        self.norm = nn.LayerNorm(self.hidden_dim, eps=self.norm_eps, bias=self.is_norm_bias)
        self.dropout = nn.Dropout(self.dropout_probability)

    def forward(self,
                input_index: torch.LongTensor = None,
                input_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        if input_embed is not None:
            hidden_embed = self.norm(input_embed)
            hidden_embed = self.dropout(hidden_embed)
        if input_embed is None:
            hidden_embed = self.index2embed(input_index)
            hidden_embed = self.norm(hidden_embed)
            hidden_embed = self.dropout(hidden_embed)

        return hidden_embed