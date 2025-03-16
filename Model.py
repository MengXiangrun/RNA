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


    def forward(self, source_emb, target_emb,
                source_pad_mask=None, target_pad_mask=None, attention_mask=None, is_casual=False):


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


        return out_emb, attention

    def merge_mask(self,
                   num_source_token, source_pad_mask,
                   num_target_token, target_pad_mask,
                   attention_mask, batch_size, num_head, is_casual):


        return attention_mask


class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, in_dim=-1, hidden_dim=2048, out_dim=-1, bias=True, dropout=0.1):


    def forward(self, emb):


        return out_emb


class EncoderLayer(torch.nn.Module):
    def __init__(self,
                 emb_dim: int,
                 num_head: int,
                 ffn_dim: int = 2048,
                 dropout: float = 0.1,
                 eps: float = 1e-5):
        super().__init__()


    def forward(self, source_emb, source_pad_mask, attention_mask, is_casual):


        return ffn_emb


class Transformer(torch.nn.Module, RNADataset):
    def __init__(self,
                 num_token_type,
                 emb_dim=512,
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

        self.encoder_layer_list = torch.nn.ModuleList()
        for layer_index in range(self.num_encoder_layers):
            encoder_layer = EncoderLayer(emb_dim=emb_dim,
                                         num_head=num_head,
                                         ffn_dim=ffn_dim,
                                         dropout=dropout,
                                         eps=eps)
            self.encoder_layer_list.append(encoder_layer)

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
            encoder_emb = encoder_layer.forward(source_emb=source_emb,
                                                source_pad_mask=source_pad_mask,
                                                attention_mask=source_attention_mask,
                                                is_casual=is_casual)

        return encoder_emb
