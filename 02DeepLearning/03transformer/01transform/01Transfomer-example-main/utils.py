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
        super(PrepareForMultiHeadAttention,self).__init__()

        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        # Input has shape [seq_len, batch_size, d_model] or [batch_size, d_model].

        # [seq_len, batch_size]
        head_shape = x.shape[:-1]
        head_shape = map(int, head_shape)
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

        super(MultiHeadAttention,self).__init__()

        # Number of features per head
        self.d_k = d_model // heads
        # Number of heads
        self.heads = heads

        #query , key and value have shape [seq_len, batch_size, d_model]-> [seq_len, batch_size, heads,d_k]
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
        #assert表达式为真程序继续执行
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        # Same mask applied to all heads.做运算时自动广播
        mask = mask.unsqueeze(-1)

        # resulting mask has shape [seq_len_q, seq_len_k, batch_size, heads]
        return mask

    def forward(self, *,
                query: torch.Tensor,        # torch.Size([100, 10, 20])
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):

        # query, key and value  have shape [seq_len, batch_size, d_model]
        seq_len, batch_size, _ = map(int, query.shape)

        #mask has shape [seq_len_q, seq_len_k, batch_size, heads]
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)

        # Prepare query, key and value for attention computation.
        # These will then have shape [seq_len, batch_size, heads, d_k].
        query = self.query(query)   # torch.Size([100, 10, 4, 5])
        key = self.key(key)         # torch.Size([100, 10, 4, 5])
        value = self.value(value)   # torch.Size([100, 10, 4, 5])

        # This gives a tensor of shape [seq_len_q, seq_len_k, batch_size, heads].
        #rescale scores
        scores = self.get_scores(query, key)    # torch.Size([100, 100, 10, 4])
        scores =scores* self.scale              # torch.Size([100, 100, 10, 4])

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # softmax attention along the key sequence dimension
        attn = self.softmax(scores) # torch.Size([100, 100, 10, 4])

        # Apply dropout
        attn = self.dropout(attn)   # torch.Size([100, 100, 10, 4])

        # Multiply by values，最后出来的是query的大小 100 100 10 4   100 10 4 5 -> 100 10 4 5
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)    # torch.Size([100, 10, 4, 5])

        # Save attentions for any other calculations 
        self.attn = attn.detach()

        # Concatenate multiple heads
        x = x.reshape(seq_len, batch_size, -1)

        # Output layer
        return self.output(x)



class FeedForward(nn.Module):
    """
    ## FFN module
    """

    def __init__(self, d_model: int, d_ff: int,
                 dropout: float = 0.1,
                 activation=nn.ReLU(),
                 is_gated: bool = False,
                 bias: bool = True,
                 bias_gate: bool = True):
        """
        * d_model is the number of features
        * d_ff is the number of features in the hidden layer of the FFN
        * dropout is dropout probability for the hidden layer
        * is_gated specifies whether the hidden layer is gated
        * bias1 specified whether the first fully connected layer should have a learnable bias
        * bias2 specified whether the second fully connected layer should have a learnable bias
        * bias_gate specified whether the fully connected layer for the gate should have a learnable bias
        """
        super(FeedForward,self).__init__()

        self.layer1 = nn.Linear(d_model, d_ff, bias=bias)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias)
        # Hidden layer dropout
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        # Whether there is a gate
        self.is_gated = is_gated
        if is_gated:
            # If there is a gate the linear layer to transform inputs to be multiplied by the gate
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: torch.Tensor):

        g = self.activation(self.layer1(x))

        if self.is_gated:
            x = g * self.linear_v(x)
        # Otherwise
        else:
            x = g
        # Apply dropout
        x = self.dropout(x)

        return self.layer2(x)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout_prob: float, max_len: int = 5000):
        super(PositionalEncoding,self).__init__()
        
        self.dropout = nn.Dropout(dropout_prob)

        self.register_buffer('positional_encodings', get_positional_encoding(d_model, max_len), False)

    def forward(self, x: torch.Tensor):
        # seq_len可能不一样长所以使用:x.shape[0],可以取到最后一个因为索引是从0开始的
        pe = self.positional_encodings[:x.shape[0]].detach().requires_grad_(False)
        x = x + pe
        x = self.dropout(x)
        return x



def get_positional_encoding(d_model: int, max_len: int = 5000):

    encodings = torch.zeros(max_len, d_model)
    # Position indexes，sin和cos的相位分布
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

    #sin和cos的频率分布，随着two_i变大而变小
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))

    #在d_model的不同维度下用相同频率的波函数的position相位值编码在整个句子内的位置信息。
    #index上越高维度的波函数震动频率越小
    encodings[:, 0::2] = torch.sin(position * div_term)
    encodings[:, 1::2] = torch.cos(position * div_term)

    # Add batch dimension
    encodings = encodings.unsqueeze(1).requires_grad_(False)

    return encodings

   # 10 100 20
def get_training_data(batch_size, input_sequence_length, output_sequence_length):
    """
    generate data shape is seq_len,batch.feature_size
    """
    i = input_sequence_length + output_sequence_length
    b=torch.linspace(-10,0,i).unsqueeze(1).repeat(1,batch_size)
    t=torch.zeros(1,batch_size).uniform_(0,2).int()
    b=t+b
    s=torch.sigmoid(b.float())  # torch.Size([120, 10])

    return s[:input_sequence_length,:].unsqueeze(-1), s[-output_sequence_length:,:]


def get_testing_data(input_sequence_length, output_sequence_length):
    """
    generate data shape is seq_len,batch,feature_size
    """
    i = input_sequence_length + output_sequence_length
    b=torch.linspace(-5,0,input_sequence_length).unsqueeze(1)
    s=torch.sigmoid(b.float())
    
    c=torch.linspace(0,5,output_sequence_length)
    u=torch.sigmoid(c.float())

    return s.unsqueeze(-1), u.unsqueeze(-1)


def clone_module_list(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



def subsequent_mask(seq_len1,seq_len2):
    """
    ## Subsequent mask to mask out data from future (subsequent) time steps
    因为是下三角矩阵，所以每行表示不同时间之间的注意力。
    """
    mask = torch.ones(seq_len1, seq_len2)           # torch.Size([20, 20])
    mask = torch.tril(mask)                         # 返回矩阵(二维张量)或矩阵批次的下三角部分
    mask = mask.to(torch.bool).unsqueeze(-1)        # torch.Size([20, 20, 1])
    return mask

