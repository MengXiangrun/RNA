from typing import Dict, Optional, Tuple, Union
import torch
from torch import nn
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.utils import is_flash_attn_2_available
from transformers.models.modernbert.configuration_modernbert import ModernBertConfig


class ModernBertEmbeddings(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.dropout = nn.Dropout(config.embedding_dropout)

    def forward(self,
                input_index: torch.LongTensor = None,
                input_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        if input_embed is not None:
            hidden_embed = self.norm(input_embed)
            hidden_embed = self.dropout(hidden_embed)
        if input_embed is None:
            hidden_embed = self.tok_embeddings(input_index)
            hidden_embed = self.norm(hidden_embed)
            hidden_embed = self.dropout(hidden_embed)

        return hidden_embed
