import numpy as np
import os
import re
import string
import random

import torch
import torch.nn as nn
import torch.functional as F

vocab_size = 20000  # Only consider the top 20k words
maxlen = 80  # Max sequence size
embed_dim = 256  # Embedding size for each token
num_heads = 2  # Number of attention heads
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer

def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = torch.arange(n_dest)[:, None]
    j = torch.arange(n_src)
    m = i >= j - n_src + n_dest
    mask = m.type(dtype)
    mask = mask.reshape(1, n_dest, n_src)
    mult = torch.cat(
        [torch.expand_dims(batch_size, -1), torch.tensor([1, 1], dtype=torch.int32)], 0
    )
    return torch.tile(mask, mult)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
    
    def forward(self, x):
        input_shape = x.shape
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, torch.bool)
        attention_output = self.att(x, x, causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(x + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(maxlen, embed_dim)

    def forward(self, x):
        maxlen = x.size(1)
        positions = torch.arange(0, maxlen, dtype=torch.long)
        positions = positions.unsqueeze(0).expand_as(x)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class Transformer(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim, num_heads, feed_forward_dim):
        super(Transformer, self).__init__()
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
        self.dense = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.transformer_block(x)
        x = self.dense(x)
        return x

# Create model
model = Transformer(maxlen, vocab_size, embed_dim, num_heads, feed_forward_dim)
