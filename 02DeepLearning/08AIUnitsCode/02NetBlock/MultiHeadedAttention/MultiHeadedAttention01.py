### https://zhuanlan.zhihu.com/p/348146980?utm_medium=social&utm_oi=643096634247090176
import torch
import torch.nn as nn
from typing import Optional

import torch.nn.functional as F
import numpy as np
import math
import copy
from typing import Optional, List
class MultiHeadAttention(nn.Module):
    r"""
    ## Multi-Head Attention Module
    This computes scaled multi-headed attention for given `query`, `key` and `value` vectors.
    $$\mathop{Attention}(Q, K, V) = \underset{seq}{\mathop{softmax}}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$
    In simple terms, it finds keys that matches the query, and get the values of
     those keys.
    It uses dot-product of query and key as the indicator of how matching they are.
    Before taking the $softmax$ the dot-products are scaled by $\frac{1}{\sqrt{d_k}}$.
    This is done to avoid large dot-product values causing softmax to
    give very small gradients when $d_k$ is large.
    Softmax is calculate along the axis of of the sequence (or time).
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True,
                 mask_type: str = 'softmax'):
        """
        * `heads` is the number of heads.
        * `d_model` is the number of features in the `query`, `key` and `value` vectors.
        """

        super().__init__()

        # Number of features per head
        self.d_k = d_model // heads
        # Number of heads
        self.heads = heads

        # These transform the `query`, `key` and `value` vectors for multi-headed attention.
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

        # Softmax for attention along the time dimension of `key`
        if mask_type == 'softmax':
            self.selector = nn.Softmax(dim=1)
        else:
            raise NotImplemented()

        # Output layer
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        # Scaling factor before the softmax
        self.scale = 1 / math.sqrt(self.d_k)

        # We store attentions so that it can used for logging, or other computations if needed
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        ### Calculate scores between queries and keys
        This method can be overridden for other variations like relative attention.
        """

        # Calculate $Q K^\top$
        return torch.einsum('bihd,bjhd->bijh', query, key)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        `query`, `key` and `value` are the tensors that store
        collection of*query*, *key* and *value* vectors.
        They have shape `[batch_size, seq_len, d_model]`.
        `mask` has shape `[batch_size, seq_len, seq_len]` and indicates
        `mask[b, i, j]` indicates whether for batch `b`,
        query at position `i` has access to key-value at position `j`.
        """

        # `query`, `key` and `value`  have shape `[batch_size, seq_len, d_model]`
        batch_size, seq_len, _ = query.shape

        if mask is not None:
            # `mask` has shape `[batch_size, seq_len, seq_len]`,
            # where first dimension is the query dimension.
            # If the query dimension is equal to $1$ it will be broadcasted
            assert mask.shape[1] == 1 or mask.shape[1] == mask.shape[2]

            # Same mask applied to all heads.
            mask = mask.unsqueeze(-1)

        # Prepare `query`, `key` and `value` for attention computation
        # These will then have shape `[batch_size, seq_len, heads, d_k]`
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Compute attention scores
        # Results in a tensor of shape `[batch_size, seq_len, seq_len, heads]`
        scores = self.get_scores(query, key)

        # Scale scores
        scores *= self.scale

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # softmax attention along the key sequence dimension
        attn = self.selector(scores)

        # Apply dropout
        attn = self.dropout(attn)
        # Multiply by values

        x = torch.einsum("bijh,bjhd->bihd", attn, value)
        # Save attentions for any other calculations

        self.attn = attn.detach()

        # Concatenate multiple heads
        x = x.reshape(batch_size, seq_len, -1)

        # Output layer
        return self.output(x)

import onnx
if __name__ == '__main__':
    model2 = SwinTransformerLayer_v2(32, 2)
    #model2 = WindowAttention_v2(32,(7,7), 2)
    #dummy = torch.zeros(1, 32, 128, 128)
    # torch.Size([361, 49, 32])

    dummy = torch.zeros(361, 49, 32)
    onnx_name = "D:/000/WindowAttention0.onnx"
    model2 = model2.attn
    onnx_model = torch.onnx.export(model2, (dummy,),
                      onnx_name,
                      opset_version=11)
    onnx_model = onnx.load(onnx_name)
    graph = onnx_model.graph
    nodes = graph.node
    inputs = graph.input
    outputs = graph.output
