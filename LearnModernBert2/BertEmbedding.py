from typing import Dict, Optional, Tuple, Union
import torch
from torch import nn
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.utils import is_flash_attn_2_available
from transformers.models.modernbert.configuration_modernbert import ModernBertConfig

if is_flash_attn_2_available():
    from flash_attn.layers.rotary import RotaryEmbedding
    from flash_attn.ops.triton.rotary import apply_rotary
else:
    RotaryEmbedding = object


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    对查询和键张量应用旋转位置嵌入。
    参数:
        q (torch.Tensor): 查询张量，形状为 [batch_size, heads, seq_len, head_dim] 或 [batch_size, seq_len, heads, head_dim]。
        k (torch.Tensor): 键张量，形状与 q 相同。
        cos (torch.Tensor): 旋转嵌入的余弦部分，形状为 [batch_size, seq_len, head_dim]。
        sin (torch.Tensor): 旋转嵌入的正弦部分，形状与 cos 相同。
        position_ids (torch.Tensor, 可选): 已弃用，未使用。
        unsqueeze_dim (int, 可选, 默认为 1): 指定沿哪个维度展开 cos 和 sin，以便广播到 q 和 k 的维度。
    返回:
        tuple(torch.Tensor)：包含使用旋转位置嵌入旋转后的查询和键张量。
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_unpadded(qkv,
                          cos,
                          sin,
                          cu_seqlens: Optional[torch.Tensor] = None,
                          max_seqlen: Optional[int] = None,
                          ):
    """
    应用旋转位置嵌入。

    参数：
        qkv (torch.Tensor): 打包的 QKV 输入张量，形状为 (total_nnz, 3, nheads, headdim)。
        cos (torch.Tensor): 余弦旋转嵌入，形状为 (seqlen_rotary, rotary_dim / 2)。
        sin (torch.Tensor): 正弦旋转嵌入，形状与 cos 相同。
        interleaved (bool): 是否交错旋转维度 (True: GPT-J 风格; False: GPT-NeoX 风格)。
        inplace (bool): 是否就地应用旋转嵌入。
        seqlen_offsets (torch.Tensor 或 int): 每个序列的偏移量，形状为 (batch_size,) 或 int。
        cu_seqlens (torch.Tensor, 可选): 累积序列长度，形状为 (batch + 1,)。
        max_seqlen (int): 最大序列长度。

    返回：
        out (torch.Tensor): 输出张量，形状为 (total_nnz, dim)。

    rotary_dim 必须小于等于 headdim。仅对 x 的前 rotary_dim 维度应用旋转嵌入。
    """

    return ApplyRotaryEmbUnpad.apply(qkv, cos, sin, cu_seqlens, max_seqlen)


class ApplyRotaryEmbUnpad(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                qkv,
                cos,
                sin,
                cu_seqlens: Optional[torch.Tensor] = None,
                max_seqlen: Optional[int] = None,
                ):
        # qkv 的形状为 (total_nnz, 3, nheads, headdim)
        qkv = qkv.contiguous()
        total_nnz, _three, _nheads, headdim = qkv.shape
        # 需要 qkv 是连续的，以便在重塑时组合 (3, nheads) 维度时得到相同的张量
        # qk = rearrange(qkv[:, :2], "b_s t h d -> b_s (t h) d")
        qk = qkv[:, :2].view(total_nnz, -1, headdim)
        apply_rotary(qk,
                     cos,
                     sin,
                     seqlen_offsets=0,
                     cu_seqlens=cu_seqlens,
                     max_seqlen=max_seqlen,
                     interleaved=False,
                     inplace=True,
                     )
        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.max_seqlen = max_seqlen
        return qkv

    @staticmethod
    def backward(ctx, do):
        cos, sin, cu_seqlens = ctx.saved_tensors
        do = do.contiguous()
        total_nnz, _three, _nheads, headdim = do.shape
        # 需要 dqkv 是连续的，以便在重塑时组合 (3, nheads) 维度时得到相同的张量
        dqk = do[:, :2].view(total_nnz, -1, headdim)
        apply_rotary(dqk,
                     cos,
                     sin,
                     seqlen_offsets=0,
                     cu_seqlens=cu_seqlens,
                     max_seqlen=ctx.max_seqlen,
                     interleaved=False,
                     inplace=True,
                     conjugate=True,
                     )

        return do, None, None, None, None, None, None


class ModernBertUnpaddedRotaryEmbedding(RotaryEmbedding):
    """
    The rotary position embeddings applied directly to unpadded sequences.
    """

    def __init__(self,
                 dim: int,
                 base: float = 10000.0,
                 max_seqlen: Optional[int] = None,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 ):
        """
        max_seqlen: 如果提供了 max_seqlen、device 和 dtype，则预先计算 cos_sin_cache，直到 max_seqlen。
        如果训练/推理期间的 max_seqlen、device 或 dtype 不同，则在前向传播期间重新计算 cos_sin_cache。
        """
        super().__init__(dim=dim, base=base, pos_idx_in_fp32=True, device=device, interleaved=False)
        self.max_seqlen = max_seqlen

        if max_seqlen is not None and device is not None and dtype is not None:
            self._update_cos_sin_cache(max_seqlen, device=device, dtype=dtype)

    def forward(self,
                qkv: torch.Tensor,
                cu_seqlens: torch.Tensor,
                max_seqlen: Optional[int] = None,
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        """
        对 qkv 应用旋转嵌入（就地操作）。
        qkv: (total_nnz, 3, nheads, headdim)
        cu_seqlens: (batch + 1,) 累积序列长度
        max_seqlen: int 批次中的最大序列长度
        """
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)

        qkv = apply_rotary_unpadded(qkv,
                                    self._cos_cached,
                                    self._sin_cached,
                                    cu_seqlens=cu_seqlens,
                                    max_seqlen=max_seqlen,
                                    )

        return qkv

    def extra_repr(self) -> str:
        return f"dim={self.dim}, base={self.base}, scale_base={self.scale_base}"


class ModernBertEmbeddings(nn.Module):
    """
    与 BertEmbeddings 相同，只是位置嵌入索引略有调整。
    """

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.drop = nn.Dropout(config.embedding_dropout)

    @torch.compile(dynamic=True)
    def compiled_embeddings(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.drop(self.norm(self.tok_embeddings(input_ids)))

    def forward(self, input_ids: torch.LongTensor = None, inputs_embeds: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = self.drop(self.norm(inputs_embeds))
        else:
            hidden_states = (self.compiled_embeddings(input_ids)
                             if self.config.reference_compile
                             else self.drop(self.norm(self.tok_embeddings(input_ids)))
                             )
        return hidden_states


class ModernBertRotaryEmbedding(nn.Module):
    def __init__(self, config: ModernBertConfig, dim: int, base: float, device: Optional[torch.device] = None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(None, device, dim=dim, base=base)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
