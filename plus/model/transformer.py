# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# - Dong-Hyun Lee's code (https://github.com/dhlee347/pytorchic-bert)

""" Transformer model classes """

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    """ Transformer with self-attention blocks """
    def __init__(self, cfg):
        super(Transformer, self).__init__()
        self.embed = Embeddings(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])

    def forward(self, x, seg, mask):
        h = self.embed(x, seg)
        for block in self.blocks:
            h = block(h, mask)
        return h


class Embeddings(nn.Module):
    """ embedding module from word, position and token_type embeddings """
    def __init__(self, cfg):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.input_dim, cfg.hidden_dim)         # token embedding
        self.seg_embed = nn.Embedding(2, cfg.hidden_dim)                      # segment embedding
        if cfg.pos_encode:
            max_len = cfg.max_len if cfg.max_len is not None else 5000
            self.pos_embed = PositionalEncoding(max_len + 3, cfg.hidden_dim)  # position encoding
        else:
            self.pos_embed = nn.Embedding(cfg.max_len + 3, cfg.hidden_dim)        # position embedding

        self.norm = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x)  # (S,) -> (B, S)
        e = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.drop(self.norm(e))


class PositionalEncoding(nn.Module):
    """ sinusoidal positional encoding """
    def __init__(self, max_len, dim):
        super(PositionalEncoding, self).__init__()
        # compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class LayerNorm(nn.Module):
   """ layer normalization """
   def __init__(self, cfg, eps=1e-12):
       super().__init__()
       self.gamma = nn.Parameter(torch.ones(cfg.hidden_dim))
       self.beta  = nn.Parameter(torch.zeros(cfg.hidden_dim))
       self.eps = eps

   def forward(self, x):
       mean = x.mean(-1, keepdim=True)
       std = x.std(-1, keepdim=True)
       x = (x - mean) / torch.sqrt(std + self.eps)
       return self.gamma * x + self.beta


class Block(nn.Module):
    """ Transformer block """
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, mask):
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h


class MultiHeadedSelfAttention(nn.Module):
    """ multi-headed scaled dot product attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.proj_k = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.proj_v = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.scores = None  # for visualization
        self.num_heads = cfg.num_heads

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(num_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.num_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """ feedForward neural networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_dim, cfg.feedforward_dim)
        self.fc2 = nn.Linear(cfg.feedforward_dim, cfg.hidden_dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


def gelu(x):
    """ gelu activation function """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def split_last(x, shape):
    """ split the last dimension to given shape """
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    """ merge the last n_dims to a dimension """
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)