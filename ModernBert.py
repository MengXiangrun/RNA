import copy
import math
import warnings
from typing import Optional, Union, List, Tuple, cast, Dict, Any
import torch
from torch.nn import Module, ModuleList, Linear, LayerNorm, Parameter, Embedding, Dropout
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from einops import rearrange, repeat
from transformers.activations import ACT2FN
import torch.nn.functional as F
from transformers import BertTokenizer
import re


class FlexBertConfig(PretrainedConfig):
    def __init__(self,
                 num_different_token=30522,
                 hidden_dim=768,
                 num_hidden_layer=12,
                 num_attention_head=12,
                 ffn_hidden_dim=3072,
                 hidden_activation="gelu",
                 hidden_dropout=0.1,
                 attention_dropout=0.1,
                 max_position_emb=512,
                 num_sequence_segment=2,
                 init_weight_standard_deviation=0.02,
                 layer_norm_eps=1e-12,
                 pad_token_index=0,
                 position_emb_type="absolute",
                 use_cache=True,
                 classifier_dropout=None,
                 attention_layer_type: str = "base",
                 is_attention_out_bias: bool = False,
                 attention_out_dropout: float = 0.0,
                 is_attention_qkv_bias: bool = False,
                 bert_layer_type: str = "prenorm",
                 is_decoder_bias: bool = True,
                 emb_dropout: float = 0.0,
                 is_emb_norm: bool = True,
                 is_encoder_decoder_norm: bool = False,
                 emb_layer_type: str = "absolute_pos",
                 encoder_layer_type: str = "base",
                 loss_type: str = "cross_entropy",
                 loss_kwargs_dict: dict = {},
                 mlp_dropout: float = 0.0,
                 is_mlp_in_bias: bool = False,
                 mlp_layer_type: str = "mlp",
                 is_mlp_out_bias: bool = False,
                 norm_kwargs_dict: dict = {},
                 norm_type: str = "rmsnorm",
                 pad_type: str = "unpadded",
                 classifier_head_activation: str = "silu",
                 is_classifier_head_bias: bool = False,
                 classifier_head_dropout: float = 0.0,
                 classifier_head_norm_type: bool = False,
                 predict_head_activation: str = "silu",
                 is_predict_head_bias: bool = False,
                 predict_head_dropout: float = 0.0,
                 is_predict_head_norm: bool = True,
                 pool_type: str = "cls",
                 rotary_emb_dim: int | None = None,
                 rotary_emb_base: float = 10000.0,
                 rotary_emb_scale_base=None,
                 is_rotary_emb_interleaved: bool = False,
                 is_flash_attention_2: bool = False,
                 is_SDPA_attention_mask: bool = False,
                 is_emb_resize: bool = False,
                 init_method: str = "default",
                 init_standard_deviation: float = 0.02,
                 init_cutoff_factor: float = 2.0,
                 is_init_RWKV_small_emb: bool = False,
                 init_replace_attention_layer_type: str | None = None,
                 init_replace_bert_layer_type: str | None = None,
                 init_replace_mlp_layer_type: str | None = None,
                 num_initial_layer: int = 1,
                 is_skip_first_bert_layer_pre_norm: bool = False,
                 is_deterministic_flash_attention_2: bool = False,
                 is_slide_window_attention: int = -1,
                 use_global_attention_per_n_layers: int = -1,
                 local_attention_rotary_emb_base: float = -1,
                 local_attention_rotary_emb_dim: int | None = None,
                 is_unpad_before_emb_layer: bool = False,
                 is_pad_logits_after_get_loss: bool = False,
                 is_compile_model: bool = False,
                 is_masked_prediction: bool = False,
                 alibi_dim=512):
        super().__init__()
        self.num_different_token = num_different_token
        self.hidden_dim = hidden_dim
        self.num_hidden_layer = num_hidden_layer
        self.num_attention_head = num_attention_head
        self.ffn_hidden_dim = ffn_hidden_dim
        self.hidden_activation = hidden_activation
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.max_position_emb = max_position_emb
        self.num_sequence_segment = num_sequence_segment
        self.init_weight_standard_deviation = init_weight_standard_deviation
        self.initializer_range = self.init_weight_standard_deviation
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_index = pad_token_index
        self.position_emb_type = position_emb_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.attention_layer_type = attention_layer_type
        self.is_attention_out_bias = is_attention_out_bias
        self.attention_out_dropout = attention_out_dropout
        self.is_attention_qkv_bias = is_attention_qkv_bias
        self.bert_layer_type = bert_layer_type
        self.is_decoder_bias = is_decoder_bias
        self.emb_dropout = emb_dropout
        self.is_emb_norm = is_emb_norm
        self.is_encoder_decoder_norm = is_encoder_decoder_norm
        self.emb_layer_type = emb_layer_type
        self.encoder_layer_type = encoder_layer_type
        self.loss_type = loss_type
        self.loss_kwargs_dict = loss_kwargs_dict
        self.mlp_dropout = mlp_dropout
        self.is_mlp_in_bias = is_mlp_in_bias
        self.mlp_layer_type = mlp_layer_type
        self.is_mlp_out_bias = is_mlp_out_bias
        self.norm_kwargs_dict = norm_kwargs_dict
        self.norm_type = norm_type
        self.pad_type = pad_type
        self.classifier_head_activation = classifier_head_activation
        self.is_classifier_head_bias = is_classifier_head_bias
        self.classifier_head_dropout = classifier_head_dropout
        self.classifier_head_norm_type = classifier_head_norm_type
        self.predict_head_activation = predict_head_activation
        self.is_predict_head_bias = is_predict_head_bias
        self.predict_head_dropout = predict_head_dropout
        self.is_predict_head_norm = is_predict_head_norm
        self.pool_type = pool_type
        self.rotary_emb_dim = rotary_emb_dim
        self.rotary_emb_base = rotary_emb_base
        self.rotary_emb_scale_base = rotary_emb_scale_base
        self.is_rotary_emb_interleaved = is_rotary_emb_interleaved
        self.is_flash_attention_2 = is_flash_attention_2
        self.is_SDPA_attention_mask = is_SDPA_attention_mask
        self.is_emb_resize = is_emb_resize
        self.init_method = init_method
        self.init_standard_deviation = init_standard_deviation
        self.init_cutoff_factor = init_cutoff_factor
        self.is_init_RWKV_small_emb = is_init_RWKV_small_emb
        self.init_replace_attention_layer_type = init_replace_attention_layer_type
        self.init_replace_bert_layer_type = init_replace_bert_layer_type
        self.init_replace_mlp_layer_type = init_replace_mlp_layer_type
        self.num_initial_layer = num_initial_layer
        self.is_skip_first_bert_layer_pre_norm = is_skip_first_bert_layer_pre_norm
        self.is_deterministic_flash_attention_2 = is_deterministic_flash_attention_2
        self.is_slide_window_attention = is_slide_window_attention
        self.use_global_attention_per_n_layers = use_global_attention_per_n_layers
        self.local_attention_rotary_emb_base = local_attention_rotary_emb_base
        self.local_attention_rotary_emb_dim = local_attention_rotary_emb_dim
        self.is_unpad_before_emb_layer = is_unpad_before_emb_layer
        self.is_pad_logits_after_get_loss = is_pad_logits_after_get_loss
        self.is_compile_model = is_compile_model
        self.is_masked_prediction = is_masked_prediction
        self.alibi_dim = alibi_dim

        if loss_kwargs_dict.get("return_z_loss", False):
            if loss_type != "fa_cross_entropy":
                text = "当 return_z_loss 为 True 时，loss_type 必须为 'fa_cross_entropy'"
                raise ValueError(text)
            if loss_kwargs_dict.get("lse_square_scale", 0) <= 0:
                text = "lse_square_scale 必须传递给 loss_kwargs_dict 并且必须大于 0 以用于 z_loss"
                raise ValueError(text)
        if loss_kwargs_dict.get("inplace_backward", False):
            self.loss_kwargs_dict["inplace_backward"] = False
            text = "inplace_backward=True 会导致不正确的度量。自动设置为 False。"
            warnings.warn(text)
        condition1 = use_global_attention_per_n_layers > 0
        condition2 = (self.num_hidden_layer - 1) % use_global_attention_per_n_layers != 0
        if condition1 and condition2:
            text = f"{use_global_attention_per_n_layers=} 必须是比 {self.num_hidden_layer=} 少 1 的因子"
            raise ValueError(text)

        if self.is_slide_window_attention != -1:
            if not self.is_flash_attention_2:
                raise ValueError("滑动窗口注意力仅支持 FlashAttention2")
            if self.is_slide_window_attention % 2 != 0 and self.is_slide_window_attention % 64 != 0:
                text1 = f"滑动窗口必须是偶数并且可以被 64 整除: {self.is_slide_window_attention=}"
                text2 = f"{self.is_slide_window_attention % 64} {self.is_slide_window_attention % 2}"
                text = text1 + text2
                raise ValueError(text)
        else:
            if self.use_global_attention_per_n_layers != -1:
                text = "当 is_slide_window_attention 被禁用时，use_global_attention_per_n_layers 必须为 -1"
                raise ValueError(text)
            if self.local_attention_rotary_emb_base != -1:
                text = "当 is_slide_window_attention 被禁用时，local_attention_rotary_emb_base 必须为 -1"
                raise ValueError(text)
            if self.local_attention_rotary_emb_dim is not None:
                text = "当 is_slide_window_attention 被禁用时，local_attention_rotary_emb_dim 必须为 None"
                raise ValueError(text)
        if self.is_unpad_before_emb_layer and self.pad_type != "unpadded":
            text = "is_unpad_before_emb_layer=True 需要 pad_type='unpadded'。自动设置 pad_type='unpadded'"
            warnings.warn(text)
            self.pad_type = "unpadded"
        if self.is_pad_logits_after_get_loss and not self.is_unpad_before_emb_layer:
            raise ValueError("is_pad_logits_after_get_loss=True 需要 is_unpad_before_emb_layer=True")
        if self.is_unpad_before_emb_layer and self.emb_layer_type == "absolute_pos":
            raise ValueError(f"{self.is_unpad_before_emb_layer=} 与 {self.emb_layer_type=} 不兼容")


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, indices: torch.Tensor, first_axis_dim) -> torch.Tensor:
        ctx.save_for_backward(indices)
        assert indices.ndim == 1 and values.ndim >= 2
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
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        input1 = rearrange(input, "b ... -> b (...)")
        input2 = 0
        input3 = repeat(indices, "z -> z d", d=second_dim)
        out = torch.gather(input1, input2, input3)
        out = out.reshape(-1, *other_shape)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        shape = [ctx.first_axis_dim, grad_output.shape[1]]
        grad_input = torch.zeros(shape, device=grad_output.device, dtype=grad_output.dtype)
        input1 = 0
        input2 = repeat(indices, "z -> z d", d=grad_output.shape[1])
        input3 = grad_output
        grad_input.scatter_(input1, input2, input3)
        out = grad_input.reshape(ctx.first_axis_dim, *other_shape)
        return out, None


class Unpadding(Module):
    def __init__(self, config):
        super().__init__()
        self.index_first_axis = IndexFirstAxis.apply
        self.index_put_first_axis = IndexPutFirstAxis.apply

    def unpad_input(self, token_emb: torch.Tensor, attention_mask: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = int(seqlens_in_batch.max().item())
        input1 = torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32)
        input2 = (1, 0)
        cu_seqlens = F.pad(input1, input2)
        input = rearrange(token_emb, "b s ... -> (b s) ...")
        input = self.index_first_axis(input, indices)
        token_emb = cast(torch.Tensor, input)
        return token_emb, indices, cu_seqlens, max_seqlen_in_batch

    def unpad_input_only(self, token_emb: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        rearranged = rearrange(token_emb, "b s ... -> (b s) ...")
        out = self.index_first_axis(rearranged, indices)
        return out

    def pad_input(self, token_emb: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
        output = self.index_put_first_axis(token_emb, indices, batch * seqlen)
        out = rearrange(output, "(b s) ... -> b s ...", b=batch)
        return out


class Alibi(Module):
    def __init__(self, config):
        super().__init__()
        # 注意力头的数量
        self.num_attention_head = config.num_attention_head
        # ALiBi 偏差张量的初始大小
        self.alibi_dim = config.alibi_dim
        # 当前 ALiBi 偏差张量的大小
        self.alibi_dim = int(self.alibi_dim)

        self.slopes = None

        # 初始化 ALiBi 偏差张量，初始全为0
        # (one_batch, num_head, target_sequence_length, source_sequence_length)
        self.alibi = torch.zeros((1, self.num_attention_head, self.alibi_dim, self.alibi_dim))

        # 预先计算初始大小的ALiBi偏差
        self.rebuild_alibi_tensor(alibi_dim=self.alibi_dim)

        self.check_variable()

    def rebuild_alibi_tensor(self, alibi_dim: int, device: Optional[Union[torch.device, str]] = None):
        # 创建上下文位置和记忆位置索引
        target_position = torch.arange(alibi_dim, device=device)[:, None]  # shape: (alibi_dim, 1)
        source_position = torch.arange(alibi_dim, device=device)[None, :]  # shape: (1, alibi_dim)

        # 计算相对位置距离
        # shape: (alibi_dim, alibi_dim)
        relative_distance = torch.abs(source_position - target_position)

        # 扩展到注意力头的维度
        # shape: (num_head, alibi_dim, alibi_dim)
        relative_distance = relative_distance.unsqueeze(0).expand(self.num_attention_head, -1, -1)

        # 获取每个注意力头的斜率
        head_slopes = self.get_alibi_head_slopes(num_attention_head=self.num_attention_head)
        slopes = torch.Tensor(head_slopes).to(device)
        self.slopes = slopes  # 保存斜率用于后续使用

        # 计算ALiBi偏差
        # (num_head)
        # -> (num_head, 1, 1)
        slopes = slopes.unsqueeze(1).unsqueeze(1)

        # (num_head, alibi_dim, alibi_dim)
        # -> (1, num_head, alibi_dim, alibi_dim)
        alibi = slopes * -relative_distance
        alibi = alibi.unsqueeze(0)
        assert alibi.shape == torch.Size([1, self.num_attention_head, alibi_dim, alibi_dim])

        # 更新当前大小
        self.alibi_dim = alibi_dim
        # 更新ALiBi偏差张量
        self.alibi = alibi

    def get_alibi_head_slopes(self, num_attention_head: int) -> List[float]:
        # 如果注意力头数量是2的幂，则直接计算
        if math.log2(num_attention_head).is_integer():
            out = self.get_slopes_power_of_2(num_attention_head=num_attention_head)
            return out

        # 否则，找到最接近的2的幂，并插值计算
        closest_power_of_2 = 2 ** math.floor(math.log2(num_attention_head))
        slopes_a = self.get_slopes_power_of_2(num_attention_head=closest_power_of_2)
        slopes_b = self.get_alibi_head_slopes(2 * closest_power_of_2)
        slopes_b = slopes_b[0::2][: num_attention_head - closest_power_of_2]
        out = slopes_a + slopes_b
        return out

    def get_slopes_power_of_2(self, num_attention_head: int) -> List[float]:
        start = 2 ** (-(2 ** -(math.log2(num_attention_head) - 3)))
        ratio = start
        out = [start * ratio ** i for i in range(num_attention_head)]
        return out

    def forward(self, sequence_length: int, device: Optional[Union[torch.device, str]] = None) -> torch.Tensor:
        # 如果当前ALiBi张量大小小于输入序列长度，则重新构建
        if self.alibi_dim < sequence_length:
            warnings.warn(f"Increasing alibi size from {self.alibi_dim} to {sequence_length}")
            self.rebuild_alibi_tensor(alibi_dim=sequence_length, device=device)
        # 如果设备不一致，则将ALiBi张量移动到指定设备
        elif self.alibi.device != device:
            self.alibi = self.alibi.to(device)
            self.slopes = self.slopes.to(device)

        # 返回根据序列长度裁剪后的ALiBi偏差张量
        out = self.alibi[:, :, :sequence_length, :sequence_length]

        return out

    def check_variable(self):
        print()
        print(self.__class__.__name__)
        variable_dict = vars(self)
        for variable_name, variable_value in variable_dict.items():
            if isinstance(variable_value, (int, float, str, bool, dict, list, tuple)):
                print(variable_name, variable_value)
        print()


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(torch.ones(dim))

    def forward(self, x):
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        output = x.type_as(x)
        return output * self.weight

    def reset_parameters(self):
        torch.init.ones_(self.weight)


class BertAlibiEmbeddings(Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_different_token = config.num_different_token
        self.pad_token_index = config.pad_token_index
        self.num_sequence_segment = config.num_sequence_segment
        self.dropout = config.hidden_dropout
        self.max_position_emb = config.max_position_emb

        self.token2emb = Embedding(num_embeddings=self.num_different_token,
                                   embedding_dim=self.hidden_dim,
                                   padding_idx=self.pad_token_index)
        if self.num_sequence_segment:
            self.segment2emb = Embedding(num_embeddings=self.num_sequence_segment,
                                         embedding_dim=self.hidden_dim)
            self.use_segment_emb = True
        else:
            self.use_segment_emb = False

        if config.norm_type == "layernorm":
            self.layer_norm = LayerNorm(self.hidden_dim)
        if config.norm_type == "rmsnorm":
            self.layer_norm = RMSNorm(self.hidden_dim)

        self.dropout = Dropout(p=self.dropout)
        if self.use_segment_emb:
            segment_index = torch.zeros(self.max_position_emb, dtype=torch.long)
            self.register_buffer("segment_index", segment_index, persistent=False)

        self.check_variable()

    def forward(self,
                token_index: Optional[torch.LongTensor] = None,
                segment_index: Optional[torch.LongTensor] = None,
                position_index: Optional[torch.LongTensor] = None,
                token_emb: Optional[torch.FloatTensor] = None,
                past_key_values_length: int = 0,
                ) -> torch.Tensor:
        assert (token_index is not None) ^ (token_emb is not None), "Must specify either token_index or token_emb!"
        if token_index is not None:
            token_index_shape = token_index.size()
        else:
            assert token_emb is not None
            token_index_shape = token_emb.size()[:-1]

        seq_length = token_index_shape[1]
        if position_index is None:
            pass

        if self.use_segment_emb and segment_index is None:
            if hasattr(self, "segment_index"):
                buffered_segment_index = self.segment_index[:, :seq_length]
                buffered_segment_index_expanded = buffered_segment_index.expand(token_index_shape[0], seq_length)
                segment_index = buffered_segment_index_expanded
            else:
                segment_index = torch.zeros(token_index_shape, dtype=torch.long, device=token_index.device)
        if token_emb is None:
            token_emb = self.token2emb(token_index)

        if self.use_segment_emb:
            segment_emb = self.segment2emb(segment_index)
            token_emb = token_emb + segment_emb
        else:
            token_emb = token_emb

        token_emb = self.layer_norm(token_emb)
        token_emb = self.dropout(token_emb)
        return token_emb

    def check_variable(self):
        print()
        print(self.__class__.__name__)
        variable_dict = vars(self)
        for variable_name, variable_value in variable_dict.items():
            if isinstance(variable_value, (int, float, str, bool, dict, list, tuple)):
                print(variable_name, variable_value)
        print()


class BertAlibiUnpadAttention(Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_head = config.num_attention_head
        self.head_dim = self.hidden_dim // self.num_head
        assert self.hidden_dim % self.num_head == 0

        self.attention_dropout = Dropout(p=config.attention_dropout)
        self.q_linear = Linear(self.hidden_dim, self.hidden_dim)
        self.k_linear = Linear(self.hidden_dim, self.hidden_dim)
        self.v_linear = Linear(self.hidden_dim, self.hidden_dim)
        self.qkv_linear = Linear(self.hidden_dim, 3 * self.hidden_dim)
        self.dense_linear = Linear(self.hidden_dim, self.hidden_dim)
        if config.norm_type == "layernorm": self.layer_norm = LayerNorm(self.hidden_dim)
        if config.norm_type == "rmsnorm": self.layer_norm = RMSNorm(self.hidden_dim)
        self.mlp_dropout = Dropout(p=config.hidden_dropout)
        self.check_variable()

    def forward(self,
                token_emb: torch.Tensor,
                unpadding_block: Unpadding,
                alibi_block: Alibi,
                cu_seqlens: torch.Tensor,
                max_s: int,
                subset_index: Optional[torch.Tensor] = None,
                indices: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        batch_size, sequence_length, hidden_dim = token_emb.shape
        token_emb_1 = token_emb.clone()
        token_emb_2 = token_emb.clone()
        qkv = self.qkv_linear(token_emb_1)

        qkv = unpadding_block.pad_input(token_emb=qkv, indices=indices, batch=cu_seqlens.shape[0] - 1, seqlen=max_s)
        unpad_batch_size, *_ = qkv.shape
        qkv = qkv.view(unpad_batch_size, -1, 3, self.num_head, self.head_dim)

        q = qkv[:, :, 0, :, :].permute(0, 2, 1, 3)
        k = qkv[:, :, 1, :, :].permute(0, 2, 3, 1)
        v = qkv[:, :, 2, :, :].permute(0, 2, 1, 3)
        attention = torch.matmul(q, k) / math.sqrt(self.head_dim)
        attention = attention + alibi_block(max_s, token_emb.device)
        attention = F.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)

        token_emb_1 = torch.matmul(attention, v).permute(0, 2, 1, 3)
        token_emb_1 = unpadding_block.unpad_input_only(token_emb_1, torch.squeeze(attention_mask) == 1)
        token_emb_1 = token_emb_1.view(batch_size, hidden_dim)
        if subset_index is not None:
            token_emb_1 = IndexFirstAxis.apply(token_emb_1, subset_index)
            token_emb_2 = IndexFirstAxis.apply(token_emb_2, subset_index)
        token_emb_1 = self.dense_linear(token_emb_1)
        token_emb_1 = self.mlp_dropout(token_emb_1)
        out = self.layer_norm(token_emb_1 + token_emb_2)
        return out

    def check_variable(self):
        print()
        print(self.__class__.__name__)
        variable_dict = vars(self)
        for variable_name, variable_value in variable_dict.items():
            if isinstance(variable_value, (int, float, str, bool, dict, list, tuple)):
                print(variable_name, variable_value)
        print()


class BertResidualGLU(Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.ffn_hidden_dim = config.ffn_hidden_dim
        self.hidden_activation = config.hidden_activation
        self.hidden_dropout = config.hidden_dropout

        self.linear1 = Linear(self.hidden_dim, self.ffn_hidden_dim * 2, bias=False)
        self.activation = ACT2FN[self.hidden_activation]
        self.linear2 = Linear(self.ffn_hidden_dim, self.hidden_dim)
        self.dropout = Dropout(self.hidden_dropout)

        if config.norm_type == "layernorm": self.layer_norm = LayerNorm(self.hidden_dim)
        if config.norm_type == "rmsnorm": self.layer_norm = RMSNorm(self.hidden_dim)
        self.check_variable()

    def forward(self, token_emb: torch.Tensor) -> torch.Tensor:
        residual_emb = token_emb.clone()
        token_emb = self.linear1(token_emb)

        gated = token_emb[:, : self.ffn_hidden_dim]
        non_gated = token_emb[:, self.ffn_hidden_dim:]
        token_emb = self.activation(gated) * non_gated
        token_emb = self.dropout(token_emb)
        token_emb = self.linear2(token_emb)
        token_emb = token_emb + residual_emb
        token_emb = self.layer_norm(token_emb)
        return token_emb

    def check_variable(self):
        print()
        print(self.__class__.__name__)
        variable_dict = vars(self)
        for variable_name, variable_value in variable_dict.items():
            if isinstance(variable_value, (int, float, str, bool, dict, list, tuple)):
                print(variable_name, variable_value)
        print()


class BertAlibiEncoderLayer(Module):
    def __init__(self, config):
        super().__init__()
        self.attention_block = BertAlibiUnpadAttention(config)
        self.mlp_block = BertResidualGLU(config)
        self.unpadding_block = Unpadding(config)
        self.alibi_block = Alibi(config)
        self.check_variable()

    def forward(self,
                token_emb: torch.Tensor,
                cu_seqlens: torch.Tensor,
                seqlen: int,
                subset_index: Optional[torch.Tensor] = None,
                indices: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        token_emb = self.attention_block.forward(token_emb=token_emb,
                                                 unpadding_block=self.unpadding_block,
                                                 alibi_block=self.alibi_block,
                                                 cu_seqlens=cu_seqlens,
                                                 max_s=seqlen,
                                                 subset_index=subset_index,
                                                 indices=indices,
                                                 attention_mask=attention_mask)
        token_emb = self.mlp_block.forward(token_emb=token_emb)
        return token_emb

    def check_variable(self):
        print()
        print(self.__class__.__name__)
        variable_dict = vars(self)
        for variable_name, variable_value in variable_dict.items():
            if isinstance(variable_value, (int, float, str, bool, dict, list, tuple)):
                print(variable_name, variable_value)
        print()


class BertAlibiEncoder(Module):
    def __init__(self, config):
        super().__init__()
        self.num_hidden_layer = config.num_hidden_layer
        self.hidden_dim = config.hidden_dim
        self.num_attention_head = config.num_attention_head
        self.alibi_dim = config.alibi_dim

        self.encoder_layers = ModuleList()
        layer = BertAlibiEncoderLayer(config)
        for layer_index in range(self.num_hidden_layer):
            self.encoder_layers.append(copy.deepcopy(layer))

        self.check_variable()

    def forward(self,
                token_emb: torch.Tensor,
                attention_mask: torch.Tensor,
                subset_mask: Optional[torch.Tensor] = None):
        unpadding_block = Unpadding(self.config)
        token_emb, indices, cu_seqlens, _ = unpadding_block.unpad_input(token_emb, attention_mask.bool())

        alibi_block = Alibi(self.config)
        alibi_bias = alibi_block.forward(sequence_length=token_emb.shape[1], device=token_emb.device)

        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        attention_mask = (1.0 - attention_mask) * -10000.0

        # attention_bias = attention_mask[:, :, :token_emb.shape[1], :token_emb.shape[1]]
        # alibi_attention_mask = attention_bias + alibi_bias

        attention_mask = attention_mask[:, :, :token_emb.shape[1], :token_emb.shape[1]]
        attention_mask = attention_mask + alibi_bias

        token_emb_list = []
        if subset_mask is None:
            for encoder_layer in self.encoder_layers:
                token_emb = encoder_layer.forward(token_emb=token_emb,
                                                  cu_seqlens=cu_seqlens,
                                                  seqlen=token_emb.shape[1],
                                                  subset_index=None,
                                                  indices=indices,
                                                  attention_mask=attention_mask)
                token_emb_list.append(token_emb)
            token_emb = unpadding_block.pad_input(token_emb=token_emb,
                                                  indices=indices,
                                                  batch=token_emb.shape[0],
                                                  seqlen=attention_mask.shape[1])
        if subset_mask is not None:
            for i in range(len(self.encoder_layers) - 1):
                encoder_layer = self.encoder_layers[i]
                token_emb = encoder_layer(token_emb=token_emb,
                                          cu_seqlens=cu_seqlens,
                                          seqlen=token_emb.shape[1],
                                          subset_index=None,
                                          indices=indices,
                                          attention_mask=attention_mask)
                token_emb_list.append(token_emb)
            subset_index = torch.nonzero(subset_mask[attention_mask.bool()], as_tuple=False).flatten()
            token_emb = self.encoder_layers[-1].forward(token_emb=token_emb,
                                                        cu_seqlens=cu_seqlens,
                                                        seqlen=token_emb.shape[1],
                                                        subset_index=subset_index,
                                                        indices=indices,
                                                        attention_mask=attention_mask)
            token_emb_list.append(token_emb)

        return token_emb_list

    def check_variable(self):
        print()
        print(self.__class__.__name__)
        variable_dict = vars(self)
        for variable_name, variable_value in variable_dict.items():
            if isinstance(variable_value, (int, float, str, bool, dict, list, tuple)):
                print(variable_name, variable_value)
        print()


class BertPooler(Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.linear = Linear(config.hidden_dim, config.hidden_dim)
        self.activation = F.tanh
        self.check_variable()

    def forward(self, token_emb: torch.Tensor, pool: Optional[bool] = True) -> torch.Tensor:
        first_token_tensor = token_emb[:, 0] if pool else token_emb
        pooled_output = self.linear(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

    def check_variable(self):
        print()
        print(self.__class__.__name__)
        variable_dict = vars(self)
        for variable_name, variable_value in variable_dict.items():
            if isinstance(variable_value, (int, float, str, bool, dict, list, tuple)):
                print(variable_name, variable_value)
        print()


class BertModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer: bool = True):
        super(BertModel, self).__init__(config)
        self.embeddings = BertAlibiEmbeddings(config)
        self.encoder = BertAlibiEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.post_init()

    def forward(self,
                token_index: torch.Tensor,
                segment_index: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_index: Optional[torch.Tensor] = None,
                masked_tokens_mask: Optional[torch.Tensor] = None,
                **kwargs
                ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], Optional[torch.Tensor]]:
        if attention_mask is None: attention_mask = torch.ones_like(token_index)
        if segment_index is None: segment_index = torch.zeros_like(token_index)

        token_emb = self.embeddings.forward(token_index=token_index,
                                            segment_index=segment_index,
                                            position_index=position_index)

        subset_mask = []
        first_col_mask = []
        if masked_tokens_mask is None:
            subset_mask = None
        else:
            first_col_mask = torch.zeros_like(masked_tokens_mask)
            first_col_mask[:, 0] = True
            subset_mask = masked_tokens_mask | first_col_mask

        token_emb_list = self.encoder.forward(token_emb=token_emb,
                                              attention_mask=attention_mask,
                                              subset_mask=subset_mask)
        pooled_token_emb = None
        if masked_tokens_mask is None:
            token_emb = token_emb_list[-1]
            pooled_token_emb = self.pooler(token_emb) if self.pooler is not None else None

        if masked_tokens_mask is not None:
            attention_mask_bool = attention_mask.bool()
            subset_index = subset_mask[attention_mask_bool]
            token_emb = token_emb_list[-1][masked_tokens_mask[attention_mask_bool][subset_index]]

            if self.pooler is not None:
                token_emb = token_emb_list[-1][first_col_mask[attention_mask_bool][subset_index]]
                pooled_token_emb = self.pooler(token_emb, pool=False)

            if self.pooler is None:
                pooled_token_emb = None

        return token_emb_list, pooled_token_emb

    def get_input_embeddings(self):
        return self.embeddings.token2emb

    def set_input_embeddings(self, value):
        self.embeddings.token2emb = value


class CustomTokenizer:
    def __init__(self, sentences):
        self.vocab = {}
        self.vocab_size = 0
        self.add_tokens(tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]'])
        self.add_tokens_from_sentences(sentences=sentences)  # 初始化时添加所有单词和符号

    def add_tokens(self, tokens):
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = self.vocab_size
                self.vocab_size += 1

    def add_tokens_from_sentences(self, sentences):
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            for token in tokens:
                self.add_tokens(tokens=[token])

    def tokenize(self, text):
        # 使用正则表达式分割单词和符号，处理更复杂的情况
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)  # 匹配单词或非单词字符
        return tokens

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            try:
                ids.append(self.vocab[token])
            except KeyError:
                # 处理不在词汇表中的token，这里用一个特殊token代替
                ids.append(self.vocab['UNK'])  # 使用UNK代替#
        return ids

    def encode(self, sentence_list, return_tensors='pt'):
        tokenized_sentences = []
        for sentence in sentence_list:
            tokens = self.tokenize(sentence)
            ids = self.convert_tokens_to_ids(tokens)
            tokenized_sentences.append([self.vocab['[CLS]']] + ids + [self.vocab['[SEP]']])

        max_len = max(len(ids) for ids in tokenized_sentences)
        padded_sentences = [ids + [self.vocab['[PAD]']] * (max_len - len(ids)) for ids in tokenized_sentences]
        attention_mask = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in tokenized_sentences]
        input_ids = torch.tensor(padded_sentences)
        attention_mask = torch.tensor(attention_mask)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}


def create_batches(input_ids, attention_mask, batch_size):
    num_batches = math.ceil(input_ids.size(0) / batch_size)
    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_input_ids = input_ids[start_idx:end_idx]
        batch_attention_mask = attention_mask[start_idx:end_idx]
        batches.append((batch_input_ids, batch_attention_mask))
    return batches


sentence_list = [
    "Quantum mechanics is a branch of physics that studies the behavior of microscopic particles, revealing wave-particle duality and the uncertainty principle.",
    "Machine learning is a subfield of artificial intelligence that enables computers to learn and improve performance from data through algorithms.",
    "Climate change refers to significant long-term changes in the Earth's climate system, primarily caused by greenhouse gas emissions from human activities.",
    "Gene editing is a technology that alters the genetic characteristics of an organism by modifying DNA sequences, with CRISPR-Cas9 being one of the most well-known tools.",
    "Blockchain is a distributed ledger technology that ensures data security and immutability through cryptographic algorithms.",
    "Psychology studies human behavior and mental processes, encompassing cognition, emotion, and social behavior, among other aspects.",
    "Renewable energy refers to energy that can be naturally replenished in a short period of time, such as solar, wind, and hydropower.",
    "Economics analyzes the allocation of resources and the production, distribution, and consumption of goods and services, aiming to maximize social welfare.",
    "Virtual reality is a technology that simulates environments through computer technology, allowing users to interact in a three-dimensional space.",
    "An ecosystem is a complex network composed of biological communities and their non-living environment, maintaining biodiversity and ecological balance."
]
config = FlexBertConfig(num_different_token=30522)  # 需要调整num_different_token以适应自定义词表大小
model = BertModel(config)
tokenizer = CustomTokenizer(sentence_list)
inputs = tokenizer.encode(sentence_list, return_tensors='pt')
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

batch_size = 2
batches = create_batches(input_ids, attention_mask, batch_size)

for batch_input_ids, batch_attention_mask in batches:
    with torch.no_grad():
        output1, output2 = model.forward(token_index=batch_input_ids, attention_mask=batch_attention_mask)
    print("Encoder outputs:", output1)
    print("Pooled output:", output2)
