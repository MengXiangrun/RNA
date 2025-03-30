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
from BertEncoder import ModernBertEncoderLayer
from BertEmbedding import ModernBertEmbedding

logger = logging.get_logger(__name__)


class ModernBertPreTrainedModel(PreTrainedModel):
    # config_class = ModernBertConfig
    # base_model_prefix = "model"
    # supports_gradient_checkpointing = True
    # _no_split_modules = ["ModernBertEmbeddings", "ModernBertEncoderLayer"]
    # _supports_flash_attn_2 = True
    # _supports_sdpa = True
    # _supports_flex_attn = False

    def _init_weights(self, module):
        self.init_linear(module=module)
        self.init_embedding(module=module)

    def init_linear(self, module, gain=1.0):
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight, gain=gain)
            if module.bias is not None: init.zeros_(module.bias)
        else:
            for name, submodule in module.__dict__.items():
                if isinstance(submodule, nn.Linear):
                    self.init_linear(module=submodule, gain=gain)

    def init_embedding(self, module, gain=1.0):
        if isinstance(module, nn.Embedding):
            # init.xavier_normal_(module.weight, gain=gain)
            init.xavier_uniform_(module.weight, gain=gain)
        else:
            for name, submodule in module.__dict__.items():
                if isinstance(submodule, nn.Embedding):
                    self.init_embedding(module=submodule, gain=gain)

    def cutoff_init(self, module: nn.Module, std: float, cutoff_factor=3):
        a = - cutoff_factor * std
        b = cutoff_factor * std
        nn.init.trunc_normal_(tensor=module.weight, mean=0.0, std=std, a=a, b=b)
        if isinstance(module, nn.Linear):
            if module.bias is not None: nn.init.zeros_(module.bias)


class ModernBertConfig(PretrainedConfig):
    model_type = "modernbert"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self,
                 embed_config,
                 attention_config,
                 ffn_config,
                 num_encoder_layer=12,
                 hidden_dim=768,
                 norm_eps=1e-5,
                 is_norm_bias=False,
                 local_attention_size=128
                 ):
        super().__init__()
        self.num_encoder_layer = num_encoder_layer
        self.embed_config = embed_config
        self.attention_config = attention_config
        self.ffn_config = ffn_config
        self.hidden_dim = hidden_dim
        self.norm_eps = norm_eps
        self.is_norm_bias = is_norm_bias
        self.local_attention_size = local_attention_size


class ModernBertModel(ModernBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config=config)
        self.config = config
        self.embed_config = config.embed_config
        self.attention_config = config.attention_config
        self.ffn_config = config.ffn_config
        self.num_encoder_layer = config.num_encoder_layer
        self.local_attention_size = config.local_attention_size

        self.embed_layer = ModernBertEmbedding(config=self.embed_config)

        self.encoder_layers = nn.ModuleList()
        for layer_index in range(self.num_encoder_layer):
            encoder_layer = ModernBertEncoderLayer(attention_config=self.attention_config,
                                                   ffn_config=self.ffn_config,
                                                   layer_index=layer_index)
            self.encoder_layers.append(encoder_layer)

        self.final_norm_layer = nn.LayerNorm(normalized_shape=self.config.hidden_dim,
                                             eps=self.config.norm_eps,
                                             bias=self.config.is_norm_bias)
        self.post_init()

    def forward(self,
                input_index: Optional[torch.LongTensor] = None,
                input_embed: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                sliding_window_mask: Optional[torch.Tensor] = None,
                position_index: Optional[torch.LongTensor] = None,
                batch_size: Optional[int] = None,
                sequence_length: Optional[int] = None,
                ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutput]:
        one = 1
        assert (input_index is None) ^ (input_embed is None)

        if input_index is not None:
            self.warn_if_padding_and_no_attention_mask(input_index, attention_mask)

        if input_embed is not None:
            batch_size, sequence_length, hidden_dim = input_embed.shape
            device = input_embed.device

        if input_index is not None:
            batch_size, sequence_length = input_index.shape
            device = input_index.device

        if attention_mask is None:
            shape = (batch_size, sequence_length)
            attention_mask = torch.ones(size=shape, device=device, dtype=torch.bool)

        if attention_mask is not None:
            assert attention_mask.shape[0] == batch_size
            assert attention_mask.shape[1] == sequence_length

        if position_index is None:
            position_index = torch.arange(sequence_length, device=device)
            position_index = position_index.unsqueeze(0)
            assert position_index.shape[0] == one
            assert position_index.shape[1] == sequence_length

        global_attention_mask, local_attention_mask = self.update_attention_mask(attention_mask=attention_mask)
        assert global_attention_mask.shape[0] == batch_size
        assert global_attention_mask.shape[1] == one
        assert global_attention_mask.shape[2] == sequence_length
        assert global_attention_mask.shape[3] == sequence_length
        assert local_attention_mask.shape[0] == batch_size
        assert local_attention_mask.shape[1] == one
        assert local_attention_mask.shape[2] == sequence_length
        assert local_attention_mask.shape[3] == sequence_length

        hidden_embed = self.embed_layer.forward(input_index=input_index, input_embed=input_embed)

        for encoder_layer in self.encoder_layers:
            hidden_embed = encoder_layer.forward(hidden_embed=hidden_embed,
                                                 global_attention_mask=global_attention_mask,
                                                 local_attention_mask=local_attention_mask,
                                                 position_index=position_index)

        hidden_embed = self.final_norm_layer(hidden_embed)

        return hidden_embed

    def get_input_embeddings(self):
        return self.embed_layer.tok_embeddings

    def set_input_embeddings(self, value):
        self.embed_layer.tok_embeddings = value

    def update_attention_mask(self, attention_mask: torch.Tensor, target_sequence_length=None) -> torch.Tensor:
        device = attention_mask.device
        global_attention_mask = attention_mask.clone()

        # actual token = 0.0, mask token = -inf
        # 只对source_sequence_length做mask target_sequence_length不做mask
        batch_size, source_sequence_length = global_attention_mask.shape
        if target_sequence_length is None:
            target_sequence_length = source_sequence_length

        global_attention_mask = global_attention_mask.view(batch_size, 1, source_sequence_length)
        global_attention_mask = global_attention_mask.repeat(1, target_sequence_length, 1)

        batch_size, target_sequence_length, source_sequence_length = global_attention_mask.shape

        source_position_index = torch.arange(start=0, end=source_sequence_length, step=1, device=device)
        target_position_index = torch.arange(start=0, end=target_sequence_length, step=1, device=device)

        source_position_index = source_position_index.view(source_sequence_length, 1)
        target_position_index = target_position_index.view(target_sequence_length, 1)

        distance = target_position_index - source_position_index.T
        distance = torch.abs(distance)

        # (batch_size, target_sequence_length, source_sequence_length)
        window_size = self.local_attention_size
        window_mask = distance <= window_size // 2
        window_mask = window_mask.view(1, target_sequence_length, source_sequence_length)
        window_mask = window_mask.repeat(batch_size, 1, 1)

        # (batch_size, target_sequence_length, source_sequence_length)
        local_attention_mask = global_attention_mask * window_mask

        if self.dtype:
            min_value = torch.finfo(self.dtype).min
        else:
            min_value = torch.finfo(torch.float32).min

        global_attention_mask = 1.0 - global_attention_mask
        global_attention_mask = global_attention_mask * min_value

        local_attention_mask = 1.0 - local_attention_mask
        local_attention_mask = local_attention_mask * min_value

        global_attention_mask = global_attention_mask.view(batch_size,
                                                           1,
                                                           target_sequence_length,
                                                           source_sequence_length)
        local_attention_mask = local_attention_mask.view(batch_size,
                                                         1,
                                                         target_sequence_length,
                                                         source_sequence_length)

        return global_attention_mask, local_attention_mask


class ModernBertPredictionHead(nn.Module):
    def __init__(self,
                 in_dim=768,
                 hidden_dim=768,
                 out_dim=768,
                 is_linear_bias=False,
                 dropout_probability=0.0,
                 norm_eps=1e-5,
                 is_norm_bias=False,
                 activation='gelu',
                 pool_type="cls"):
        super().__init__()
        self.dense = nn.Linear(in_dim, hidden_dim, is_linear_bias)
        self.act = ACT2FN[activation]
        self.norm = nn.LayerNorm(hidden_dim, eps=norm_eps, bias=is_norm_bias)
        self.dropout = torch.nn.Dropout(dropout_probability)
        self.predictor = nn.Linear(hidden_dim, out_dim)
        self.pool_type = pool_type

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, hidden_embed: torch.Tensor, attention_mask) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_embed.shape
        assert attention_mask.shape[0] == batch_size
        assert attention_mask.shape[1] == sequence_length

        if self.pool_type == "cls":
            hidden_embed = hidden_embed[:, 0]
        elif self.pool_type == "mean":
            a = (hidden_embed * attention_mask.unsqueeze(-1)).sum(dim=1)
            b = attention_mask.sum(dim=1, keepdim=True)
            hidden_embed = a / b

        hidden_embed = self.norm(self.act(self.dense(hidden_embed)))
        hidden_embed = self.dropout(hidden_embed)
        out = self.predictor(hidden_embed)
        return out
