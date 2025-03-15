import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
import uuid


class RowSelfAttention(nn.Module):
    def __init__(self,
                 emb_dim,
                 num_heads,
                 dropout=0.0,
                 max_tokens_per_msa: int = 2 ** 16):
        super().__init__()
        """计算2D输入的行自注意力。"""
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = emb_dim // num_heads
        self.max_tokens_per_msa = max_tokens_per_msa
        self.attention_shape = "hnij"

        self.k_linear = nn.Linear(emb_dim, emb_dim)
        self.v_linear = nn.Linear(emb_dim, emb_dim)
        self.q_linear = nn.Linear(emb_dim, emb_dim)
        self.out_linear = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def align_scaling(self, q):
        num_rows = q.size(0)
        scaling = self.head_dim ** -0.5
        out = scaling / math.sqrt(num_rows)

        return out

    def batched_forward(self, x, self_attention_mask=None, self_attention_padding_mask=None):
        num_rows, num_columns, batch_size, emb_dim = x.size()
        max_rows = max(1, self.max_tokens_per_msa // num_columns)

        attention = 0

        for start in range(0, num_rows, max_rows):
            batch_x = x[start: start + max_rows]

            if self_attention_padding_mask is not None:
                batch_self_attention_padding_mask = self_attention_padding_mask[:, start: start + max_rows]
            else:
                batch_self_attention_padding_mask = None

            batch_attention = self.compute_attention(
                x=batch_x,
                self_attention_mask=self_attention_mask,
                self_attention_padding_mask=batch_self_attention_padding_mask)

            attention = attention + batch_attention

        attention = attention.softmax(-1)
        attention = self.dropout(attention)

        out = []
        for start in range(0, num_rows, max_rows):
            batch_x = x[start: start + max_rows]
            batch_out = self.compute_output(x=batch_x, attnetion=attention)
            out.append(batch_out)
        out = torch.cat(out, 0)

        return out, attention

    def compute_attention(self, x, self_attention_mask=None, self_attention_padding_mask=None):
        num_rows, num_columns, batch_size, embed_dim = x.size()

        q = self.q_linear(x).view(num_rows, num_columns, batch_size, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(num_rows, num_columns, batch_size, self.num_heads, self.head_dim)

        scaling = self.align_scaling(x)
        q = q * scaling

        if self_attention_padding_mask is not None:
            # 将任何填充的对齐位置置零 - 这很重要，因为我们在对齐轴上进行求和。
            self_attention_padding_mask = self_attention_padding_mask.permute(1, 2, 0)
            self_attention_padding_mask = self_attention_padding_mask.unsqueeze(3)
            self_attention_padding_mask = self_attention_padding_mask.unsqueeze(4)
            self_attention_padding_mask = self_attention_padding_mask.to(q)
            self_attention_padding_mask = 1 - self_attention_padding_mask
            q = q * self_attention_padding_mask

        attention = torch.einsum(f"rinhd,rjnhd->{self.attention_shape}", q, k)

        # 掩码大小: [B x R x C]
        # 权重大小: [H x B x C x C]
        if self_attention_mask is not None: raise NotImplementedError

        if self_attention_padding_mask is not None:
            self_attention_padding_mask = self_attention_padding_mask[:, 0].unsqueeze(0)
            self_attention_padding_mask = self_attention_padding_mask.unsqueeze(2)

            attention = attention.masked_fill(self_attention_padding_mask, -10000)

        return attention

    def compute_output(self, x, attnetion):
        num_rows, num_columns, batch_size, emb_dim = x.size()

        v = self.v_linear(x)
        v = v.view(num_rows, num_columns, batch_size, self.num_heads, self.head_dim)

        out = torch.einsum(f"{self.attention_shape},rjnhd->rinhd", attnetion, v)
        out = out.contiguous().view(num_rows, num_columns, batch_size, emb_dim)
        out = self.out_linear(out)

        return out

    def forward(self, x, self_attention_mask=None, self_attention_padding_mask=None):
        num_rows, num_columns, batch_size, emb_dim = x.size()

        if (num_rows * num_columns > self.max_tokens_per_msa) and not torch.is_grad_enabled():
            out = self.batched_forward(x=x,
                                       self_attention_mask=self_attention_mask,
                                       self_attention_padding_mask=self_attention_padding_mask)
            return out

        else:
            attention = self.compute_attention(x=x,
                                               self_attention_mask=self_attention_mask,
                                               self_attention_padding_mask=self_attention_padding_mask)
            attention = attention.softmax(dim=-1)
            attention = self.dropout(attention)
            out = self.compute_output(x, attention)
            return out, attention


class ColumnSelfAttention(nn.Module):
    def __init__(self,
                 emb_dim,
                 num_heads,
                 dropout=0.0,
                 max_tokens_per_msa: int = 2 ** 16):
        super().__init__()
        """计算2D输入的列自注意力。"""
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = emb_dim // num_heads
        self.max_tokens_per_msa = max_tokens_per_msa
        self.k_linear = nn.Linear(emb_dim, emb_dim)
        self.v_linear = nn.Linear(emb_dim, emb_dim)
        self.q_linear = nn.Linear(emb_dim, emb_dim)
        self.out_linear = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def batched_forward(self, x, self_attention_mask=None, self_attention_padding_mask=None):
        num_rows, num_columns, batch_size, emb_dim = x.size()
        max_columns = max(1, self.max_tokens_per_msa // num_rows)

        out = []
        attention = []
        for start in range(0, num_columns, max_columns):
            batch_x = x[:, start: start + max_columns]

            if self_attention_padding_mask is not None:
                batch_self_attention_padding_mask = self_attention_padding_mask[:, :, start: start + max_columns]
            else:
                batch_self_attention_padding_mask = None

            batch_out, batch_attention = self.forward(batch_x,
                                                      self_attention_mask=self_attention_mask,
                                                      self_attention_padding_mask=batch_self_attention_padding_mask)
            out.append(batch_out)
            attention.append(batch_attention)

        batch_out = torch.cat(out, 1)
        attention = torch.cat(attention, 1)

        return batch_out, attention

    def compute_output(self, x, self_attention_mask=None, self_attention_padding_mask=None):
        num_rows, num_columns, batch_size, emb_dim = x.size()

        # 如果只有一个位置，这是等价的，并且不会因填充而中断
        if num_rows == 1:
            attention = torch.ones(self.num_heads, num_columns, batch_size, num_rows, num_rows,
                                   device=x.device, dtype=x.dtype)
            v = self.v_linear(x)
            out = self.out_linear(v)
        else:
            q = self.q_linear(x).view(num_rows, num_columns, batch_size, self.num_heads, self.head_dim)
            k = self.k_linear(x).view(num_rows, num_columns, batch_size, self.num_heads, self.head_dim)
            v = self.v_linear(x).view(num_rows, num_columns, batch_size, self.num_heads, self.head_dim)

            scaling = self.head_dim ** -0.5
            q = q * scaling

            attention = torch.einsum("icnhd,jcnhd->hcnij", q, k)

            if self_attention_mask is not None:  raise NotImplementedError

            if self_attention_padding_mask is not None:
                self_attention_padding_mask = self_attention_padding_mask.permute(2, 0, 1)
                self_attention_padding_mask = self_attention_padding_mask.unsqueeze(0)
                self_attention_padding_mask = self_attention_padding_mask.unsqueeze(3)
                attention = attention.masked_fill(self_attention_padding_mask, -10000)

            attention = attention.softmax(-1)
            attention = self.dropout(attention)

            out = torch.einsum("hcnij,jcnhd->icnhd", attention, v)
            out = out.contiguous().view(num_rows, num_columns, batch_size, emb_dim)
            out = self.out_linear(out)

        return out, attention

    def forward(self, x, self_attention_mask=None, self_attention_padding_mask=None):
        num_rows, num_columns, batch_size, emb_dim = x.size()

        # if False and num_rows * num_cols > 2 ** 14 and not torch.is_grad_enabled():
        if (num_rows * num_columns) > self.max_tokens_per_msa and not torch.is_grad_enabled():
            out = self.batched_forward(x=x,
                                       self_attention_mask=self_attention_mask,
                                       self_attention_padding_mask=self_attention_padding_mask)
            return out
        else:
            out = self.compute_output(x=x,
                                      self_attention_mask=self_attention_mask,
                                      self_attention_padding_mask=self_attention_padding_mask)
            return out


class NormalizedResidualBlock(nn.Module):
    def __init__(self,
                 layer: nn.Module,
                 emb_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.layer = layer
        self.dropout = nn.Dropout(p=dropout)
        # self.layer_norm = ESM1bLayerNorm(normalized_shape=self.emb_dim)
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=self.emb_dim)

    def forward(self, x, *args, **kwargs):
        residual = x
        x = self.layer_norm(x)
        out = self.layer(x, *args, **kwargs)

        if isinstance(out, tuple):
            x, *out = out
        else:
            x = out
            out = None

        x = self.dropout(x)
        x = residual + x

        if out is not None:
            return (x,) + tuple(out)
        else:
            return x


class FeedForwardNetwork(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 ffn_dim: int,
                 dropout: float = 0.1,
                 max_tokens_per_msa: int = 2 ** 14):
        super().__init__()
        self.emb_dim = emb_dim
        self.ffn_dim = ffn_dim
        self.max_tokens_per_msa = max_tokens_per_msa

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(emb_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, emb_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class AxialTransformerLayer(nn.Module):
    def __init__(self,
                 emb_dim: int = 768,
                 ffn_dim: int = 3072,
                 num_attention_heads: int = 8,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 activation_dropout: float = 0.1,
                 max_tokens_per_msa: int = 2 ** 14) -> None:
        super().__init__()
        """ 实现一个轴向MSA Transformer块。"""

        self.emb_dim = emb_dim
        self.dropout_probability = dropout

        row_self_attention = RowSelfAttention(emb_dim=emb_dim,
                                              num_heads=num_attention_heads,
                                              dropout=dropout,
                                              max_tokens_per_msa=max_tokens_per_msa)
        column_self_attention = ColumnSelfAttention(emb_dim=emb_dim,
                                                    num_heads=num_attention_heads,
                                                    dropout=dropout,
                                                    max_tokens_per_msa=max_tokens_per_msa)
        feed_forward_network = FeedForwardNetwork(emb_dim=emb_dim,
                                                  ffn_dim=ffn_dim,
                                                  dropout=activation_dropout,
                                                  max_tokens_per_msa=max_tokens_per_msa)
        self.row_self_attention = self.build_residual(layer=row_self_attention)
        self.column_self_attention = self.build_residual(layer=column_self_attention)
        self.feed_forward_network = self.build_residual(layer=feed_forward_network)

    def build_residual(self, layer: nn.Module):
        out = NormalizedResidualBlock(layer, self.emb_dim, self.dropout_probability)
        return out

    def forward(self,
                x: torch.Tensor,
                self_attn_mask: Optional[torch.Tensor] = None,
                self_attn_padding_mask: Optional[torch.Tensor] = None,
                need_head_weights: bool = False):
        """层归一化在自注意力/前馈网络模块之前或之后应用，类似于原始Transformer实现。"""
        x, row_attention = self.row_self_attention(x=x,
                                                   self_attn_mask=self_attn_mask,
                                                   self_attn_padding_mask=self_attn_padding_mask)
        x, column_attention = self.column_self_attention(x=x,
                                                         self_attn_mask=self_attn_mask,
                                                         self_attn_padding_mask=self_attn_padding_mask)
        x = self.feed_forward_network(x)

        if need_head_weights:
            return x, column_attention, row_attention
        else:
            return x


class ContactPredictionHead(nn.Module):
    def __init__(self,
                 in_dim: int,
                 prepend_bos: bool,
                 append_eos: bool,
                 bias=True,
                 eos_index: Optional[int] = None):
        super().__init__()
        """执行对称化、apc，并在输出特征上计算逻辑回归"""

        self.in_dim = in_dim
        self.prepend_bos = prepend_bos  # prepend beginning of sequence
        self.append_eos = append_eos  # append end of sequence
        if append_eos and eos_index is None: raise ValueError("使用带有eos标记的字母表，但未传入eos标记。")
        self.eos_index = eos_index
        self.regression = nn.Linear(in_dim, 1, bias)
        self.activation = nn.Sigmoid()

    def forward(self, tokens, attention):
        # 移除eos标记的注意力
        if self.append_eos:
            eos_mask = tokens.ne(self.eos_index).to(attention)
            eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
            attention = attention * eos_mask[:, None, None, :, :]
            attention = attention[..., :-1, :-1]

        # 移除cls标记的注意力
        if self.prepend_bos:
            attention = attention[..., 1:, 1:]

        batch_size, num_layers, num_heads, num_token, _ = attention.size()

        attention = attention.view(batch_size, num_layers * num_heads, num_token, num_token)

        # 特征: B x C x T x T
        attention = attention.to(next(self.parameters()))  # 注意力始终为float32，可能需要转换为float16
        attention = self.symmetrize(x=attention)
        attention = self.apc(x=attention)
        attention = attention.permute(0, 2, 3, 1)

        out = self.regression(attention)
        out = out.squeeze(3)
        out = self.activation(out)

        return out

    def symmetrize(self, x):
        "Make layer symmetric in final two dimensions, used for contact prediction."
        out = x + x.transpose(-1, -2)
        return out

    def apc(self, x):
        "Perform average product correct, used for contact prediction."
        a1 = x.sum(-1, keepdims=True)
        a2 = x.sum(-2, keepdims=True)
        a12 = x.sum((-1, -2), keepdims=True)

        avg = a1 * a2
        avg.div_(a12)  # in-place to reduce memory
        normalized = x - avg
        return normalized


class RobertaLMHead(nn.Module):
    def __init__(self, emb_dim, out_dim, weight):
        super().__init__()
        """用于掩码语言建模的头部"""
        self.dense = nn.Linear(emb_dim, emb_dim)  # 全连接层
        # self.layer_norm = ESM1bLayerNorm(emb_dim)  # 层归一化
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=emb_dim)
        self.weight = weight  # 词汇表嵌入权重
        self.bias = nn.Parameter(torch.zeros(out_dim))  # 偏置参数

    def forward(self, x, masked_tokens=None):
        # 仅在训练时投影掩码标记，节省内存和计算 # 仅处理掩码标记
        if masked_tokens is not None:  x = x[masked_tokens, :]

        x = self.dense(x)  # 全连接层
        x = self.gelu(x)  # GELU激活函数
        x = self.layer_norm(x)  # 层归一化

        # 投影回词汇表大小，并加上偏置
        x = F.linear(x, self.weight) + self.bias  # 线性层

        return x

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_emb: int, emb_dim: int, padding_index: int):
        self.max_positions = num_emb  # 记录最大位置数量

        # 如果有填充索引，则调整嵌入数量 否则，嵌入数量不变
        if padding_index is not None:
            num_emb = num_emb + padding_index + 1
        else:
            num_emb = num_emb
        super().__init__(num_emb, emb_dim, padding_index)  # 调用父类初始化函数
        """
        此模块学习固定最大大小的位置嵌入。
        填充 ID 通过基于 padding_idx 的偏移或通过将 padding_idx 设置为 None 并确保将适当的位置 ID 传递给 forward 函数来忽略。
        """

    def forward(self, input: torch.Tensor):
        """输入应为 [bsz x seqlen] 大小。"""
        # 创建掩码，用于忽略填充标记
        mask = input.ne(self.padding_idx)
        mask = mask.int()

        # 计算位置索引
        positions = torch.cumsum(mask, dim=1).type_as(mask)
        positions = positions * mask
        positions = positions.long()
        positions = positions + self.padding_idx

        # 使用F.embedding函数生成位置嵌入
        out = F.embedding(input=positions,
                          weight=self.weight,
                          padding_idx=self.padding_idx,
                          max_norm=self.max_norm,
                          norm_type=self.norm_type,
                          scale_grad_by_freq=self.scale_grad_by_freq,
                          sparse=self.sparse)

        return out


class MSATransformer(nn.Module):
    def __init__(self,
                 alphabet,
                 padding_index,
                 mask_index,
                 cls_index,
                 eos_index,
                 prepend_bos,
                 append_eos,

                 num_layers=12,
                 emb_dim=768,
                 logit_bias=False,
                 ffn_embed_dim=3072,
                 attention_heads=12,
                 dropout=0.1,
                 attention_dropout=0.1,
                 activation_dropout=0.1,
                 max_tokens_per_msa=2 ** 14):
        super().__init__()
        self.alphabet_size = len(alphabet)
        self.padding_index = padding_index
        self.mask_index = mask_index
        self.cls_index = cls_index
        self.eos_index = eos_index
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos

        # 嵌入层
        self.emb_map = nn.Embedding(num_embeddings=self.alphabet_size,
                                    embedding_dim=emb_dim,
                                    padding_idx=self.padding_index)

        # 位置嵌入
        if logit_bias:
            self.msa_position_emb = nn.Parameter(0.01 * torch.randn(1, 1024, 1, 1), requires_grad=True)
        else:
            self.register_parameter("msa_position_emb", None)

        self.dropout = nn.Dropout(dropout)

        # transformer层
        self.axial_transformer_layers = nn.ModuleList()
        for _ in range(num_layers):
            axial_transformer_layer = AxialTransformerLayer(emb_dim,
                                                            ffn_embed_dim,
                                                            attention_heads,
                                                            dropout,
                                                            attention_dropout,
                                                            activation_dropout,
                                                            max_tokens_per_msa)
            self.axial_transformer_layers.append(axial_transformer_layer)

        # 接触预测头
        self.contact_prediction_head = ContactPredictionHead(in_dim=num_layers * attention_heads,
                                                             prepend_bos=self.prepend_bos,
                                                             append_eos=self.append_eos,
                                                             eos_index=self.eos_index)

        # 位置嵌入层
        self.learned_position_emb = LearnedPositionalEmbedding(num_emb=max_tokens_per_msa,
                                                               emb_dim=emb_dim,
                                                               padding_index=self.padding_index)

        # 层归一化
        # self.layer_norm_before = ESM1bLayerNorm(emb_dim)
        # self.layer_norm_after = ESM1bLayerNorm(emb_dim)
        self.layer_norm_before = torch.nn.LayerNorm(normalized_shape=emb_dim)
        self.layer_norm_after = torch.nn.LayerNorm(normalized_shape=emb_dim)

        # 语言模型头
        self.roberta_lm_head = RobertaLMHead(emb_dim=emb_dim,
                                             out_dim=self.alphabet_size,
                                             weight=self.emb_map.weight)

    def forward(self,
                tokens,
                representation_layers=[],
                need_head_weights=False,
                return_contacts=False):
        if return_contacts: need_head_weights = True
        assert tokens.ndim == 3

        batch_size, num_alignments, num_token = tokens.size()

        # B: batch_size
        # R: num_alignments
        # C: num_token
        # D: emb_dim

        # (batch_size, num_alignments, num_token)
        padding_mask = tokens.eq(self.padding_index)

        if not padding_mask.any():
            padding_mask = None

        # (batch_size, num_alignments, num_token, emb_dim)
        x1 = self.emb_map(tokens)

        reshaped_tokens = tokens.view(batch_size * num_alignments, num_token)
        x2 = self.learned_position_emb(reshaped_tokens)
        x2 = x2.view(x1.size())

        # (batch_size, num_alignments, num_token, emb_dim)
        x = x1 + x2

        if self.msa_position_emb is not None:
            if x.size(1) > 1024:
                raise RuntimeError("使用的模型与MSA位置嵌入在最大MSA深度为1024时训练，但接收到的对齐数为 {x.size(1)}。")
            x3 = self.msa_position_emb[:, :num_alignments]
            x = x1 + x2 + x3

        x = self.layer_norm_before(x)
        x = self.dropout(x)

        if padding_mask is not None:
            unsqueezed_padding_mask = padding_mask.unsqueeze(-1)
            unsqueezed_padding_mask = unsqueezed_padding_mask.type_as(x)
            unsqueezed_padding_mask = 1 - unsqueezed_padding_mask
            x = x * unsqueezed_padding_mask

        representation_layers = set(representation_layers)

        hidden_representations = {}

        if 0 in representation_layers:
            hidden_representations[0] = x

        if need_head_weights:
            row_attention_weights = []
            col_attention_weights = []

        # (batch_size, num_alignments, num_token, emb_dim)
        # -> (num_alignments, num_token, batch_size, emb_dim)
        x = x.permute(1, 2, 0, 3)

        for layer_index, axial_transformer_layer in enumerate(self.axial_transformer_layers):
            x = axial_transformer_layer(x=x,
                                        self_attn_padding_mask=padding_mask,
                                        need_head_weights=need_head_weights)
            if need_head_weights:
                x, col_attention, row_attention = x

                # (H, num_token, batch_size, num_alignments, num_alignments)
                # -> (batch_size, H, num_token, num_alignments, num_alignments)
                col_attention = col_attention.permute(2, 0, 1, 3, 4)
                col_attention_weights.append(col_attention)

                # (H x batch_size x num_token x num_token)
                # -> (batch_size x H x num_token x num_token)
                row_attention = row_attention.permute(1, 0, 2, 3)

                row_attention_weights.append(row_attention)

            if (layer_index + 1) in representation_layers:
                hidden_representations[layer_index + 1] = x.permute(2, 0, 1, 3)

        x = self.layer_norm_after(x)

        # (num_alignments, num_token, batch_size, emb_dim)
        # -> (batch_size, num_alignments, num_token, emb_dim)
        x = x.permute(2, 0, 1, 3)

        # 最后的隐藏表示应该应用层归一化
        if (layer_index + 1) in representation_layers:
            hidden_representations[layer_index + 1] = x

        x = self.roberta_lm_head(x)

        result = {"logits": x, "representations": hidden_representations}

        if need_head_weights:
            # (batch_size, L, H, num_token, num_alignments, num_alignments)
            col_attentions = torch.stack(col_attention_weights, 1)

            # (batch_size x L x H x num_token x num_token)
            row_attentions = torch.stack(row_attention_weights, 1)

            result["col_attentions"] = col_attentions
            result["row_attentions"] = row_attentions
            if return_contacts:
                contacts = self.contact_prediction_head(tokens, row_attentions)
                result["contacts"] = contacts

        return result

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]

    @property
    def num_layers(self):
        return len(self.axial_transformer_layers)

    def max_tokens_per_msa_(self, value: int) -> None:
        # MSA Transformer在梯度禁用时自动批处理注意力计算
        # 以允许在测试时传入比可以放入GPU内存中的更大的MSA。
        # 默认情况下
        # 当输入MSA中传入的令牌数多于2^14时会发生这种情况。
        # 可以将此值设置为无穷大以禁用此行为。
        for module in self.modules():
            if isinstance(module, (RowSelfAttention, ColumnSelfAttention)):
                module.max_tokens_per_msa = value