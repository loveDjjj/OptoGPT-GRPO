"""Transformer building blocks used by the original OptoGPT checkpoints.

This module intentionally keeps the legacy class and function names because the
serialized checkpoints reference them directly. The implementation is a compact
Transformer stack with two model variants:

- ``make_model``:
  Spectrum encoder followed by a classifier/regressor head.
- ``make_model_I``:
  Spectrum-to-structure model used by OptoGPT. The spectrum is projected into a
  decoder memory tensor and the decoder autoregressively predicts structure
  tokens.

The code below keeps the original tensor flow unchanged so old checkpoints stay
loadable, but the comments are updated to make the architecture explicit.
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def clones(module, N):
    """Return ``N`` independent deep-copied layers."""

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Module):
    """Token embedding layer with the standard ``sqrt(d_model)`` scaling."""

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding used by the original Transformer paper."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe_pos = torch.mul(position, div_term)
        pe[:, 0::2] = torch.sin(pe_pos)
        pe[:, 1::2] = torch.cos(pe_pos)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # ``pe`` is registered as a buffer so it follows the module device.
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    """Scaled dot-product attention."""

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """Standard multi-head attention with shared code for self/cross attention."""

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        # Linear layers for Q, K, V and the final output projection.
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Expand mask so it is broadcast across all attention heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # Project to Q/K/V, then reshape to [batch, heads, seq, head_dim].
        query, key, value = [
            linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # Merge heads back to [batch, seq, d_model] before the output projection.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class LayerNorm(nn.Module):
    """Simple learnable layer normalization."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x_zscore = (x - mean) / torch.sqrt(std**2 + self.eps)
        return self.a_2 * x_zscore + self.b_2


class SublayerConnection(nn.Module):
    """Pre-norm residual wrapper used around attention/feed-forward blocks."""

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class FullyConnectedLayers(nn.Module):
    """Two-layer MLP used to map spectrum features into model memory/output."""

    def __init__(self, input_dim, out_dim):
        super(FullyConnectedLayers, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, out_dim)
        self.norm = LayerNorm(input_dim)

    def forward(self, x):
        return self.fc2(self.norm(self.fc1(x)))


class PositionwiseFeedForward(nn.Module):
    """Feed-forward block applied independently at each sequence position."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x)))


class Encoder(nn.Module):
    """Stack of identical encoder layers."""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """One encoder block: self-attention followed by feed-forward."""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Transformer(nn.Module):
    """Encoder-only spectrum model with a final MLP head."""

    def __init__(self, encoder, fc, src_embed):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.fc = fc
        self.src_embed = src_embed

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def forward(self, src, src_mask):
        # The original implementation keeps only the first encoded position.
        encoded = self.encode(src, src_mask)
        encoded = encoded[:, 0, :]
        return self.fc(encoded)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Build the encoder-only spectrum model variant."""

    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    fc = FullyConnectedLayers(d_model, tgt_vocab)
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        fc,
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class Decoder(nn.Module):
    """Stack of autoregressive decoder layers."""

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """One decoder block: self-attention, cross-attention, feed-forward."""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    """Mask out future positions for autoregressive decoding."""

    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0


class Transformer_I(nn.Module):
    """OptoGPT spectrum-to-structure decoder model.

    ``fc`` maps the input spectrum directly into a decoder memory tensor. The
    decoder then autoregressively predicts structure tokens conditioned on that
    memory.
    """

    def __init__(self, fc, decoder, tgt_embed, generator):
        super(Transformer_I, self).__init__()
        self.fc = fc
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.fc(src), src_mask, tgt, tgt_mask)


class Generator(nn.Module):
    """Projection from decoder hidden states to log-probabilities over tokens."""

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def make_model_I(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Build the OptoGPT inverse-design model used by the checkpoints."""

    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    fc = FullyConnectedLayers(src_vocab, d_model)
    model = Transformer_I(
        fc,
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
