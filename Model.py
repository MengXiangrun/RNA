import pandas as pd
import torch
import numpy as np
import random
import os
import torch_geometric
import torch
import math
from Dataset import RNADataset


class Linear(torch.nn.Module):
    def __init__(self, in_dim=-1, out_dim=-1, bias=True,
                 weight_initializer='kaiming_uniform', bias_initializer='zeros'):
        super().__init__()
        if out_dim <= 0:  assert False, 'out_dim <= 0'
        self.out_dim = out_dim
        self.linear = torch_geometric.nn.Linear(in_channels=in_dim,
                                                out_channels=out_dim,
                                                weight_initializer=weight_initializer,
                                                bias=bias,
                                                bias_initializer=bias_initializer)
        self.linear.reset_parameters()

    def forward(self, x):
        return self.linear(x)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, emb_dim, num_head, dropout=0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_head = num_head
        self.dropout = torch.nn.Dropout(p=dropout)
        self.head_dim = self.emb_dim // self.num_head
        assert self.head_dim * self.num_head == self.emb_dim

        self.q_linear = Linear(self.emb_dim, self.emb_dim, bias=False)
        self.k_linear = Linear(self.emb_dim, self.emb_dim, bias=False)
        self.v_linear = Linear(self.emb_dim, self.emb_dim, bias=False)
        self.out_linear = Linear(self.emb_dim, self.emb_dim, bias=True)

    def forward(self, source_emb, target_emb,
                source_pad_mask=None, target_pad_mask=None, attention_mask=None, is_casual=False):
        # batch first:
        # source_emb (batch_size, num_source_token, token_dim)
        # target_emb (batch_size, num_target_token, token_dim)
        # sequence_length = num_token = num_source_token = num_target_token

        # key_padding_mask
        # (source sequence length)
        # (batch size, source sequence length)

        num_head = self.num_head
        head_dim = self.head_dim

        # (batch_size, num_target_token, token_dim)
        # -> (batch_size, num_target_token, num_head, head_dim)
        # -> (batch_size, num_head, num_target_token, head_dim)
        # -> (batch_size * num_head, num_target_token, head_dim)
        q_emb = target_emb.clone()
        q_emb = self.q_linear(q_emb)
        batch_size, num_target_token, emb_dim = q_emb.shape
        q_emb = q_emb.view(batch_size, num_target_token, num_head, head_dim)
        q_emb = q_emb.transpose(1, 2)
        q_emb = q_emb.contiguous().view(batch_size * num_head, num_target_token, head_dim)

        # (batch_size, num_source_token, token_dim)
        # -> (batch_size, num_source_token, num_head, head_dim)
        # -> (batch_size, num_head, num_source_token, head_dim)
        # -> (batch_size * num_head, num_source_token, head_dim)
        # -> (batch_size * num_head, head_dim, num_source_token)
        k_emb = source_emb.clone()
        k_emb = self.k_linear(k_emb)
        batch_size, num_source_token, emb_dim = k_emb.shape
        k_emb = k_emb.view(batch_size, num_source_token, num_head, head_dim)
        k_emb = k_emb.transpose(1, 2)
        k_emb = k_emb.contiguous().view(batch_size * num_head, num_source_token, head_dim)
        k_emb_transpose = k_emb.transpose(-2, -1)

        # (batch_size, num_source_token, token_dim)
        # -> (batch_size, num_source_token, num_head, head_dim)
        # -> (batch_size, num_head, num_source_token, head_dim)
        # -> (batch_size * num_head, num_source_token, head_dim)
        v_emb = source_emb.clone()
        v_emb = self.v_linear(v_emb)
        batch_size, num_source_token, emb_dim = v_emb.shape
        v_emb = v_emb.view(batch_size, num_source_token, num_head, head_dim)
        v_emb = v_emb.transpose(1, 2)
        v_emb = v_emb.contiguous().view(batch_size * num_head, num_source_token, head_dim)

        # attention_mask
        # (batch_size * num_head, num_target_token, num_source_token)
        attention_mask = self.merge_mask(num_source_token=num_source_token,
                                         source_pad_mask=source_pad_mask,
                                         num_target_token=num_target_token,
                                         target_pad_mask=target_pad_mask,
                                         attention_mask=attention_mask,
                                         batch_size=batch_size,
                                         is_casual=is_casual,
                                         num_head=num_head)

        if attention_mask is not None:
            q_emb = q_emb / math.sqrt(float(head_dim))  # scaling
            attention = torch.baddbmm(attention_mask, q_emb, k_emb_transpose)
        else:
            q_emb = q_emb / math.sqrt(float(head_dim))  # scaling
            attention = torch.bmm(q_emb, k_emb.transpose(-2, -1))

        attention = torch.nn.functional.softmax(attention, dim=-1)
        attention = self.dropout(input=attention)

        # (batch_size * num_head, num_target_token, head_dim)
        # -> (num_target_token, batch_size * num_head, head_dim)
        # -> (num_target_token, batch_size, num_head, head_dim)
        # -> (num_target_token, batch_size, emb_dim)
        # -> (batch_size, num_target_token, emb_dim)
        out_emb = torch.bmm(attention, v_emb)
        out_emb = out_emb.transpose(dim0=0, dim1=1).contiguous()
        out_emb = out_emb.view(num_target_token, batch_size, num_head, head_dim).contiguous()
        out_emb = out_emb.view(num_target_token, batch_size, emb_dim).contiguous()
        out_emb = out_emb.contiguous().view(batch_size, num_target_token, emb_dim)
        out_emb = self.out_linear(out_emb)

        # (batch_size * num_head, num_target_token, num_source_token)
        # -> (batch_size, num_head, num_target_token, num_source_token)
        attention = attention.view(batch_size, num_head, num_target_token, num_source_token)

        return out_emb, attention

    def merge_mask(self,
                   num_source_token, source_pad_mask,
                   num_target_token, target_pad_mask,
                   attention_mask, batch_size, num_head, is_casual):
        if source_pad_mask is None:
            shape = (batch_size, num_source_token)
            source_pad_mask = torch.ones(shape, device=self.out_linear.linear.weight.device, dtype=torch.float32)

        if target_pad_mask is None:
            shape = (batch_size, num_target_token)
            target_pad_mask = torch.ones(shape, device=self.out_linear.linear.weight.device, dtype=torch.float32)

        if attention_mask is None:
            shape = (batch_size, num_target_token, num_source_token)
            attention_mask = torch.ones(shape, device=self.out_linear.linear.weight.device, dtype=torch.float32)

        # pad mask
        batch_size_source_pad_mask, num_source_token = source_pad_mask.shape
        batch_size_target_pad_mask, num_target_token = target_pad_mask.shape

        # attention_mask
        batch_size_attention_mask, num_target_token, num_source_token = attention_mask.shape

        assert batch_size == batch_size_source_pad_mask
        assert batch_size == batch_size_target_pad_mask
        assert batch_size == batch_size_attention_mask

        # target_pad_mask: (batch_size, num_target_token) -> (batch_size, num_target_token, 1)
        # source_pad_mask: (batch_size, num_source_token) -> (batch_size, 1, num_source_token)
        target_pad_mask = target_pad_mask.view(batch_size, num_target_token, 1)
        source_pad_mask = source_pad_mask.view(batch_size, 1, num_source_token)

        # pad_mask
        # (batch_size, num_target_token, num_source_token)
        # -> (batch_size, num_head, num_target_token, num_source_token)
        # -> (batch_size * num_head, num_target_token, num_source_token)
        pad_mask = torch.bmm(target_pad_mask, source_pad_mask)
        pad_mask = pad_mask.view(batch_size, 1, num_target_token, num_source_token)
        pad_mask = pad_mask.expand(batch_size, num_head, num_target_token, num_source_token)
        pad_mask = pad_mask.reshape(batch_size * num_head, num_target_token, num_source_token)
        pad_mask = pad_mask.float()

        # attention_mask
        # (batch_size, num_target_token, num_source_token)
        # -> (batch_size, num_head, num_target_token, num_source_token)
        # -> (batch_size * num_head, num_target_token, num_source_token)
        attention_mask = attention_mask.view(batch_size, 1, num_target_token, num_source_token)
        attention_mask = attention_mask.expand(batch_size, num_head, num_target_token, num_source_token)
        attention_mask = attention_mask.reshape(batch_size * num_head, num_target_token, num_source_token)

        # both are the same shape
        attention_mask = pad_mask * attention_mask

        # is_casual
        if is_casual:
            causal_mask = torch.ones((num_target_token, num_source_token))
            causal_mask = torch.tril(causal_mask)
            causal_mask = causal_mask.view(1, 1, num_target_token, num_source_token)
            causal_mask = causal_mask.expand(batch_size, num_head, num_target_token, num_source_token)
            causal_mask = causal_mask.reshape(batch_size * num_head, num_target_token, num_source_token)

            # both are the same shape
            attention_mask = causal_mask * attention_mask

        return attention_mask


class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, in_dim=-1, hidden_dim=2048, out_dim=-1, bias=True, dropout=0.1):
        super().__init__()
        self.linear1 = Linear(in_dim=in_dim, out_dim=hidden_dim, bias=bias)
        self.activation = torch.nn.GELU()
        self.dropout1 = torch.nn.Dropout(p=dropout)

        self.linear2 = Linear(in_dim=hidden_dim, out_dim=out_dim, bias=bias)

    def forward(self, emb):
        out_emb = self.linear1(emb)
        out_emb = self.activation(out_emb)
        out_emb = self.dropout1(out_emb)

        out_emb = self.linear2(out_emb)

        return out_emb


class EncoderLayer(torch.nn.Module):
    def __init__(self,
                 emb_dim: int,
                 num_head: int,
                 ffn_dim: int = 2048,
                 dropout: float = 0.1,
                 eps: float = 1e-5):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(emb_dim=emb_dim, num_head=num_head, dropout=dropout)
        self.mha_dropout = torch.nn.Dropout(p=dropout)
        self.mha_layer_norm = torch.nn.LayerNorm(normalized_shape=emb_dim, eps=eps, bias=True)

        self.feed_forward_network = FeedForwardNetwork(in_dim=emb_dim,
                                                       hidden_dim=ffn_dim,
                                                       out_dim=emb_dim,
                                                       dropout=dropout)
        self.ffn_dropout = torch.nn.Dropout(p=dropout)
        self.ffn_layer_norm = torch.nn.LayerNorm(normalized_shape=emb_dim, eps=eps, bias=True)

    def forward(self, source_emb, source_pad_mask, attention_mask, is_casual):
        # batch first:
        # source_emb (num_batch, num_source_token, token_dim)
        # sequence_length = num_token = num_source_token = num_target_token

        # self-attention block
        residual = source_emb.clone()
        mha_emb, attention = self.multi_head_attention.forward(source_emb=source_emb,
                                                               source_pad_mask=source_pad_mask,
                                                               target_emb=source_emb,
                                                               target_pad_mask=source_pad_mask,
                                                               attention_mask=attention_mask,
                                                               is_casual=is_casual)
        mha_emb = self.mha_dropout(mha_emb)
        mha_emb = mha_emb + residual
        mha_emb = self.mha_layer_norm(mha_emb)

        # feed forward block
        residual = mha_emb.clone()
        ffn_emb = self.feed_forward_network.forward(emb=mha_emb)
        ffn_emb = self.ffn_dropout(ffn_emb)
        ffn_emb = ffn_emb + residual
        ffn_emb = self.ffn_layer_norm(ffn_emb)

        return ffn_emb


class SinusoidalPositionalEmbedding(torch.nn.Module):
    def __init__(self, emb_dim, padding_index=None, learned=False):
        super().__init__()
        """正弦位置嵌入"""
        self.emb_dim = emb_dim
        self.padding_index = padding_index
        self.register_buffer("pe_float_tensor", torch.FloatTensor(1))  # 注册一个浮点张量缓冲区
        self.position_weight = None  # 位置嵌入权重，初始化为None

    def forward(self, token_sequence, max_num_token, pad_mask):
        # bert
        # token_sequence:索引化之后的序列, 已经包括cls, eos, pad, cls/eos/pad 分别算一个token

        self.padding_index = pad_mask

        # 获取批大小和序列长度
        batch_size, num_token = token_sequence.shape

        # 计算最大位置索引
        # max_position = self.padding_index + 1 + num_token
        max_position = max_num_token - 1

        # 如果权重不存在或最大位置超过权重大小
        # 生成位置嵌入权重
        if self.position_weight is None or max_position > self.position_weight.size(0):
            self.position_weight = self.get_emb(num_emb=max_position)

        # 将权重转换为与输入数据相同的类型
        self.position_weight = self.position_weight.type_as(self.pe_float_tensor)

        positions = self.make_positions(token_sequence)  # 生成位置索引

        # 获取位置嵌入并返回
        out = self.position_weight.index_select(0, positions.view(-1)).view(batch_size, num_token, -1).detach()

        return out

    def make_positions(self, token_sequence):
        # 创建一个掩码，用于区分填充标记
        mask = token_sequence.ne(self.padding_index)

        # 生成位置索引
        range_buf = torch.arange(token_sequence.size(1), device=token_sequence.device).expand_as(token_sequence) + self.padding_index + 1

        positions = range_buf.expand_as(token_sequence)  # 将位置索引扩展到与输入相同形状

        out = positions * mask.long() + self.padding_index * (1 - mask.long())  # 根据掩码生成最终的位置索引

        return out

    def get_emb(self, num_emb):
        half_dim = self.emb_dim // 2  # 计算嵌入维度的一半

        emb = math.log(10000) / (half_dim - 1)  # 计算频率缩放因子

        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)  # 计算频率
        emb = torch.arange(num_emb, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)  # 计算位置编码
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_emb, -1)  # 将正弦和余弦组合

        if self.emb_dim % 2 == 1:
            # zero pad  如果嵌入维度是奇数，则进行零填充
            emb = torch.cat([emb, torch.zeros(num_emb, 1)], dim=1)

        if self.padding_index is not None:
            emb[self.padding_index, :] = 0  # 将填充标记的位置编码设置为0

        return emb


class Predictor(torch.nn.Module):
    def __init__(self, in_dim=-1, hidden_dim=2048, out_dim=-1, dropout=0.1):
        super().__init__()
        self.linear1 = Linear(in_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(dropout)
        self.linear2 = Linear(hidden_dim, hidden_dim)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.linear3 = Linear(hidden_dim, out_dim)

    def forward(self, emb):
        out_emb = self.linear1(emb)
        out_emb = self.relu(out_emb)
        out_emb = self.dropout1(out_emb)
        out_emb = self.linear2(out_emb)
        out_emb = self.relu(out_emb)
        out_emb = self.dropout2(out_emb)
        out_emb = self.linear3(out_emb)

        return out_emb


class SequenceRegressorConv(torch.nn.Module):
    def __init__(self, input_dim=128, output_dim=3, kernel_size=3):
        # 可选 可作为一种Predictor
        super(SequenceRegressorConv, self).__init__()
        self.conv = torch.nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        # x.shape = (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        x = self.conv(x)  # (batch_size, output_dim, seq_len)
        x = x.transpose(1, 2)  # (batch_size, seq_len, output_dim)
        return x


class RobertaLMHead(torch.nn.Module):
    def __init__(self, emb_dim, out_dim, weight):
        super().__init__()
        # 可选 可作为一种Predictor

        """用于掩码语言建模的头部"""
        self.dense = torch.nn.Linear(emb_dim, emb_dim)  # 全连接层
        # self.layer_norm = ESM1bLayerNorm(emb_dim)  # 层归一化
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=emb_dim)
        self.weight = weight  # 词汇表嵌入权重
        self.bias = torch.nn.Parameter(torch.zeros(out_dim))  # 偏置参数

    def forward(self, x, masked_tokens=None):
        # 仅在训练时投影掩码标记，节省内存和计算 # 仅处理掩码标记
        if masked_tokens is not None:  x = x[masked_tokens, :]

        x = self.dense(x)  # 全连接层
        x = self.gelu(x)  # GELU激活函数
        x = self.layer_norm(x)  # 层归一化

        # 投影回词汇表大小，并加上偏置
        x = torch.nn.functional.linear(x, self.weight) + self.bias  # 线性层

        return x

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class RNATransformer(torch.nn.Module, RNADataset):
    def __init__(self,
                 num_token_type,
                 emb_dim=512,
                 hidden_dim=256,
                 out_dim=3,
                 num_head=8,
                 num_encoder_layers=6,
                 ffn_dim=2048,
                 dropout=0.1,
                 eps=1e-5):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_head = num_head
        self.num_encoder_layers = num_encoder_layers
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.eps = eps

        self.token2emb = torch.nn.Embedding(num_embeddings=num_token_type, embedding_dim=emb_dim)
        self.position_emb =

        self.encoder_layer_list = torch.nn.ModuleList()
        for layer_index in range(self.num_encoder_layers):
            encoder_layer = EncoderLayer(emb_dim=emb_dim,
                                         num_head=num_head,
                                         ffn_dim=ffn_dim,
                                         dropout=dropout,
                                         eps=eps)
            self.encoder_layer_list.append(encoder_layer)

        self.predictor = Predictor(in_dim=emb_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=dropout)

    def forward(self,
                source_token,
                target_token=None,
                source_emb=None,
                target_emb=None,
                source_pad_mask=None,
                target_pad_mask=None,
                source_attention_mask=None,
                target_attention_mask=None,
                is_casual=False):
        # batch first:
        # source_emb (num_batch, num_token, emb_dim)
        # target_emb (num_batch, num_token, emb_dim)
        # sequence_length = num_token

        source_emb = self.token2emb.forward(input=source_token)

        encoder_emb = source_emb.clone()
        for encoder_layer in self.encoder_layer_list:
            encoder_emb = encoder_layer.forward(source_emb=encoder_emb,
                                                source_pad_mask=source_pad_mask,
                                                attention_mask=source_attention_mask,
                                                is_casual=is_casual)

        out = self.predictor(encoder_emb)

        return out
