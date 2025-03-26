from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.utils import is_flash_attn_2_available, logging
from transformers.models.modernbert.configuration_modernbert import ModernBertConfig
from BertEmbedding import apply_rotary_pos_emb, ModernBertUnpaddedRotaryEmbedding, ModernBertRotaryEmbedding

if is_flash_attn_2_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    from flash_attn.layers.rotary import RotaryEmbedding
    from flash_attn.ops.triton.rotary import apply_rotary
else:
    RotaryEmbedding = object

logger = logging.get_logger(__name__)


class ModernBertMLP(nn.Module):
    """
    在每个 ModernBERT 层的末尾应用 GLU（门控线性单元）。
    与默认的 BERT 架构相比，此模块用具有相似功能的单个模块替换了 :
    class:`~transformers.model.bert.modeling_bert.BertIntermediate` 和
    :class:`~transformers.model.bert.modeling_bert.SelfOutput`。
    """

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.Wi = nn.Linear(config.hidden_size, int(config.intermediate_size) * 2, bias=config.mlp_bias)
        self.act = ACT2FN[config.hidden_activation]
        self.drop = nn.Dropout(config.mlp_dropout)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(input) * gate))


def eager_attention_forward(module: "ModernBertAttention",
                            qkv: torch.Tensor,
                            attention_mask: torch.Tensor,
                            sliding_window_mask: torch.Tensor,
                            position_ids: Optional[torch.LongTensor],
                            local_attention: Tuple[int, int],
                            bs: int,
                            dim: int,
                            output_attentions: Optional[bool] = False,
                            **_kwargs,
                            ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    # query, key, value: [batch_size, heads, seq_len, head_dim]
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    scale = module.head_dim ** -0.5
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale

    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask

    attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=module.attention_dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bs, -1, dim)
    if output_attentions:
        return (attn_output, attn_weights)
    return (attn_output,)


def flash_attention_forward(module: "ModernBertAttention",
                            qkv: torch.Tensor,
                            rotary_emb: ModernBertUnpaddedRotaryEmbedding,
                            cu_seqlens: torch.Tensor,
                            max_seqlen: int,
                            local_attention: Tuple[int, int],
                            bs: int,
                            dim: int,
                            target_dtype: torch.dtype = torch.bfloat16,
                            **_kwargs,
                            ) -> Tuple[torch.Tensor]:
    # (total_seqlen, 3, nheads, headdim)
    qkv = rotary_emb(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

    convert_dtype = qkv.dtype not in (torch.float16, torch.bfloat16)
    if convert_dtype:
        # FA2 implementation only supports fp16 and bf16. If FA2 is supported,
        # bfloat16 must be supported as of FA2 2.5.7. (Turing GPUs not supported)
        orig_dtype = qkv.dtype
        qkv = qkv.to(target_dtype)

        attn = flash_attn_varlen_qkvpacked_func(qkv,
                                                cu_seqlens=cu_seqlens,
                                                max_seqlen=max_seqlen,
                                                dropout_p=module.attention_dropout if module.training else 0.0,
                                                deterministic=module.deterministic_flash_attn,
                                                window_size=local_attention,
                                                )
        attn = attn.to(orig_dtype)  # type: ignore
    else:
        attn = flash_attn_varlen_qkvpacked_func(qkv,
                                                cu_seqlens=cu_seqlens,
                                                max_seqlen=max_seqlen,
                                                dropout_p=module.attention_dropout if module.training else 0.0,
                                                deterministic=module.deterministic_flash_attn,
                                                window_size=local_attention,
                                                )
    return (attn.view(bs, dim),)


def sdpa_attention_forward(module: "ModernBertAttention",
                           qkv: torch.Tensor,
                           attention_mask: torch.Tensor,
                           sliding_window_mask: torch.Tensor,
                           position_ids: Optional[torch.LongTensor],
                           local_attention: Tuple[int, int],
                           bs: int,
                           dim: int,
                           **_kwargs,
                           ) -> Tuple[torch.Tensor]:
    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    # query, key, value: [batch_size, heads, seq_len, head_dim]
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask

    attn_output = (F.scaled_dot_product_attention(query,
                                                  key,
                                                  value,
                                                  dropout_p=module.attention_dropout if module.training else 0.0,
                                                  attn_mask=attention_mask,
                                                  )
                   .transpose(1, 2)
                   .contiguous()
                   )
    attn_output = attn_output.view(bs, -1, dim)
    return (attn_output,)


MODERNBERT_ATTENTION_FUNCTION = {
    "flash_attention_2": flash_attention_forward,
    "eager": eager_attention_forward,
    "sdpa": sdpa_attention_forward,
}


class ModernBertAttention(nn.Module):
    """
    对一批未填充的序列执行多头自注意力。
    如果安装了 Flash Attention 2，则此模块使用 Flash Attention 来提高吞吐量。
    如果没有安装 Flash Attention 2，则实现将使用 PyTorch 的 SDPA 内核，这需要填充和取消填充输入，从而增加一些开销。
    请参阅 `forward` 方法了解更多详细信息。
    """
    def __init__(self, config: ModernBertConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})"
            )

        self.attention_dropout = config.attention_dropout
        self.deterministic_flash_attn = config.deterministic_flash_attn
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.head_dim * self.num_heads
        self.Wqkv = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=config.attention_bias)

        if layer_id % config.global_attn_every_n_layers != 0:
            self.local_attention = (config.local_attention // 2, config.local_attention // 2)
        else:
            self.local_attention = (-1, -1)

        rope_theta = config.global_rope_theta
        max_position_embeddings = config.max_position_embeddings
        if self.local_attention != (-1, -1):
            if config.local_rope_theta is not None:
                rope_theta = config.local_rope_theta
            max_position_embeddings = config.local_attention

        if config._attn_implementation == "flash_attention_2":
            self.rotary_emb = ModernBertUnpaddedRotaryEmbedding(dim=self.head_dim, max_seqlen=max_position_embeddings,
                                                                base=rope_theta
                                                                )
        else:
            self.rotary_emb = ModernBertRotaryEmbedding(config=config, dim=self.head_dim, base=rope_theta)

        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.out_drop = nn.Dropout(config.attention_dropout) if config.attention_dropout > 0.0 else nn.Identity()
        self.pruned_heads = set()

    def forward(self,
                hidden_states: torch.Tensor,
                output_attentions: Optional[bool] = False,
                **kwargs,
                ) -> torch.Tensor:
        qkv = self.Wqkv(hidden_states)

        bs = hidden_states.shape[0]
        if self.config._attn_implementation == "flash_attention_2":
            qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
        else:
            qkv = qkv.view(bs, -1, 3, self.num_heads, self.head_dim)

        attn_outputs = MODERNBERT_ATTENTION_FUNCTION[self.config._attn_implementation](self,
                                                                                       qkv=qkv,
                                                                                       rotary_emb=self.rotary_emb,
                                                                                       local_attention=self.local_attention,
                                                                                       bs=bs,
                                                                                       dim=self.all_head_size,
                                                                                       output_attentions=output_attentions,
                                                                                       **kwargs,
                                                                                       )
        hidden_states = attn_outputs[0]
        hidden_states = self.out_drop(self.Wo(hidden_states))

        return (hidden_states,) + attn_outputs[1:]  # add attentions if outputted


class ModernBertEncoderLayer(nn.Module):
    def __init__(self, config: ModernBertConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        if layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.attn = ModernBertAttention(config=config, layer_id=layer_id)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.mlp = ModernBertMLP(config)

    @torch.compile(dynamic=True)
    def compiled_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.mlp_norm(hidden_states))

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                sliding_window_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                cu_seqlens: Optional[torch.Tensor] = None,
                max_seqlen: Optional[int] = None,
                output_attentions: Optional[bool] = False,
                ) -> torch.Tensor:
        attn_outputs = self.attn(self.attn_norm(hidden_states),
                                 attention_mask=attention_mask,
                                 sliding_window_mask=sliding_window_mask,
                                 position_ids=position_ids,
                                 cu_seqlens=cu_seqlens,
                                 max_seqlen=max_seqlen,
                                 output_attentions=output_attentions,
                                 )
        hidden_states = hidden_states + attn_outputs[0]
        mlp_output = (self.compiled_mlp(hidden_states)
                      if self.config.reference_compile
                      else self.mlp(self.mlp_norm(hidden_states))
                      )
        hidden_states = hidden_states + mlp_output

        return (hidden_states,) + attn_outputs[1:]  # add attentions if outputted
