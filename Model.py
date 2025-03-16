import pandas as pd
import torch
import numpy as np
import random
import os
import torch_geometric
import torch
import math


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
        # -> (num_target_token, batch_size, emb_dim)
        out_emb = torch.bmm(attention, v_emb)
        out_emb = out_emb.transpose(dim0=0, dim1=1).contiguous()
        out_emb = out_emb.view(num_target_token, batch_size, num_head, head_dim)
        out_emb = out_emb.view(num_target_token, batch_size, emb_dim)
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
