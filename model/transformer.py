# model/transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, context_length):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_length, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4*embed_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

        self.context_length = context_length

    def forward(self, x):
        B, T = x.size()
        token_emb = self.token_embedding(x)
        position_ids = torch.arange(T, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(position_ids)

        x = token_emb + pos_emb
        x = self.transformer(x)
        x = self.ln(x)
        logits = self.head(x)
        return logits
