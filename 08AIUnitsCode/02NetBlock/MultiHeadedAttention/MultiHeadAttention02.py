# https://github.com/jamesYu365/Transfomer-example/blob/a867f72f539de9746668da411f524dab45ddf12f/utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
from typing import Optional, List


class PrepareForMultiHeadAttention(nn.Module):
    """
    ## Prepare for multi-head attention
    这个linear transform作用是把query，key，value映射到同一个低维空间内
    This module does a linear transformation and splits the vector into given
    number of heads for multi-head attention.
    This is used to transform **key**, **query**, and **value** vectors.
    """

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super(PrepareForMultiHeadAttention, self).__init__()

        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        # Input has shape [seq_len, batch_size, d_model] or [batch_size, d_model].

        # [seq_len, batch_size]
        head_shape = x.shape[:-1]

        x = self.linear(x)

        # Split last dimension into heads
        x = x.view(*head_shape, self.heads, self.d_k)

        # Output has shape [seq_len, batch_size, heads, d_k] or [batch_size, d_model]
        return x


class MultiHeadAttention(nn.Module):
    r"""
    This computes scaled multi-headed attention for given query, key and value vectors.
    compute similatiry between query and key, use this as attention efficient multiply value
    It uses dot-product of query and key as the indicator of how matching they are.
    Before taking the $softmax$ the dot-products are scaled by $\frac{1}{\sqrt{d_k}}$.
    This is done to avoid large dot-product values causing softmax to
    give very small gradients when $d_k$ is large.
    Softmax is calculated along the axis of of the sequence (or time).
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        """
        * heads is the number of heads.
        * d_model is the number of features in the query, key and value vectors.
        """

        super(MultiHeadAttention, self).__init__()

        # Number of features per head
        self.d_k = d_model // heads
        # Number of heads
        self.heads = heads

        # query , key and value have shape [seq_len, batch_size, d_model]-> [seq_len, batch_size, heads,d_k]
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

        # query和key计算完点积之后的shapes[seq_len_q,seq_len_k,batch,heads]
        self.softmax = nn.Softmax(dim=1)

        # Output layer
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        # Scaling factor before the softmax
        self.scale = 1 / math.sqrt(self.d_k)

        # We store attentions so that it can be used for logging, or other computations if needed
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        ### Calculate scores between queries and keys，使用的是点积的方法
        还可以有cosine，MLP等计算相似度的方法
        """
        return torch.einsum('ibhd,jbhd->ijbh', query, key)

    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        """
        mask has shape [seq_len_q, seq_len_k, batch_size], where first dimension is the query dimension.
        If the query dimension is equal to $1$ it will be broadcasted.
        """
        # assert表达式为真程序继续执行
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        # Same mask applied to all heads.做运算时自动广播
        mask = mask.unsqueeze(-1)

        # resulting mask has shape [seq_len_q, seq_len_k, batch_size, heads]
        return mask

    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):

        # query, key and value  have shape [seq_len, batch_size, d_model]
        seq_len, batch_size, _ = query.shape

        # mask has shape [seq_len_q, seq_len_k, batch_size, heads]
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)

        # Prepare query, key and value for attention computation.
        # These will then have shape [seq_len, batch_size, heads, d_k].
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # This gives a tensor of shape [seq_len_q, seq_len_k, batch_size, heads].
        # rescale scores
        scores = self.get_scores(query, key)
        scores = scores * self.scale

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # softmax attention along the key sequence dimension
        attn = self.softmax(scores)

        # Apply dropout
        attn = self.dropout(attn)

        # Multiply by values，最后出来的是query的大小
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)

        # Save attentions for any other calculations
        self.attn = attn.detach()

        # Concatenate multiple heads
        x = x.reshape(seq_len, batch_size, -1)

        # Output layer
        return self.output(x)
